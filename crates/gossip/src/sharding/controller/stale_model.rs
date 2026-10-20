//! Exhaustive stale-view model check of the re-check rules, driving the
//! real decision functions.
//!
//! A port of `spec/timed_check.py` from
//! <https://github.com/ValiChord/polite-shrink>, with one difference that
//! matters: the decision at execute time is made by this module's own
//! [`shrink_recheck`] (V3) and [`patient_recheck`] (V6), not by a
//! re-implementation of them. So these tests check the shipped code, and
//! the exact schedule counts below — copied from the Python checker —
//! check that the two models agree.
//!
//! One sector, `n` nodes all holding it. A node that announces at tick
//! `a` executes at `a + d`, reading the world as it was `lag` ticks ago
//! (per node, 1..=`max_lag`): holders as of then and intents active then.
//! Executes due in the same tick run in id order and cannot see each
//! other. Announcements are unconstrained, so safety does not lean on
//! the trigger being right. Optionally one node dies (leaves without
//! announcing) at any tick, visible to every viewer `detect` ticks later
//! regardless of gossip staleness. Safety: no *shrink* takes the sector
//! below `r` (a death alone can).
//!
//! Every combination of announce tick (or never), per-node lag and death
//! is enumerated: an exhaustive search, not a sample.

use super::*;

/// Which re-check runs at execute time.
#[derive(Clone, Copy, Debug)]
enum Rule {
    /// V3: lower-id intenders count as gone ([`shrink_recheck`]).
    TieBreak,
    /// V5: every intender counts as gone; stand down on failure
    /// ([`patient_recheck`] with `Wait` treated as `StandDown`).
    CountAll,
    /// V6: [`patient_recheck`], including waiting.
    Patient,
}

struct Outcome {
    /// Tick at which a shrink took the sector below `r`, if any.
    violation: Option<usize>,
    /// How many nodes vacated by shrinking.
    departed: u32,
}

struct World {
    n: usize,
    r: u32,
    d: usize,
    max_lag: usize,
    detect: usize,
    rule: Rule,
    ids: Vec<AgentId>,
    /// Count only intenders the viewer still sees as holders, as the
    /// Python checker does. The real controller does not: a peer marked
    /// unresponsive drops out of coverage, but its intent stays in the
    /// table until it expires, so it is discounted twice. That is the
    /// conservative direction; this switch exists to show it is the only
    /// difference between the two models.
    python_view: bool,
}

const VACATE: (u32, u32) = (0, 1);

fn bit(i: usize) -> u32 {
    1 << i
}

impl World {
    fn new(
        n: usize,
        r: u32,
        max_lag: usize,
        d: usize,
        rule: Rule,
        detect: usize,
    ) -> Self {
        // Single-byte ids order exactly like the model's node numbers.
        let ids = (0..n)
            .map(|i| AgentId::from(bytes::Bytes::from(vec![i as u8 + 1])))
            .collect();
        Self {
            n,
            r,
            d,
            max_lag,
            detect,
            rule,
            ids,
            python_view: false,
        }
    }

    fn python_view(mut self) -> Self {
        self.python_view = true;
        self
    }

    fn run(
        &self,
        ann: &[Option<usize>],
        lags: &[usize],
        crash: Option<(usize, usize)>,
    ) -> Outcome {
        let n = self.n;
        let full = bit(n) - 1;
        let far = Timestamp::now() + Duration::from_secs(3600);
        let mut holds = full;
        let mut intent: Vec<Option<usize>> = vec![None; n];
        let mut departed = 0u32;
        let mut died_at: Option<(usize, usize)> = None;

        let mut horizon =
            ann.iter().flatten().copied().max().unwrap_or(0) + self.d + 2;
        if matches!(self.rule, Rule::Patient) {
            // Room for waiting.
            horizon += n * (self.d + 2 * self.max_lag + 2);
        }
        if let Some((_, tc)) = crash {
            horizon = horizon.max(tc + 2);
        }
        // history[t] = (holders, intenders) at the end of tick t.
        let mut history: Vec<(u32, u32)> = Vec::with_capacity(horizon);

        for t in 0..horizon {
            // 1. Due executes, in id order, each on its own stale view.
            for i in 0..n {
                if intent[i] != Some(t) || holds & bit(i) == 0 {
                    continue;
                }
                let (mut hs, its) = match t.checked_sub(lags[i]) {
                    Some(s) => history[s],
                    None => (full, 0),
                };
                // A death is visible exactly `detect` ticks after it
                // happens, independently of gossip staleness: undetected,
                // still counted even in a view taken after the death;
                // detected, removed even from a view taken before it.
                if let Some((m, td)) = died_at {
                    if t - td < self.detect {
                        hs |= bit(m);
                    } else {
                        hs &= !bit(m);
                    }
                }
                let cov = [(hs & !bit(i)).count_ones()];
                let its = if self.python_view { its & hs } else { its };
                let live: Vec<(AgentId, ShrinkIntent)> = (0..n)
                    .filter(|&j| its & bit(j) != 0)
                    .map(|j| {
                        (
                            self.ids[j].clone(),
                            ShrinkIntent {
                                vacate: VACATE,
                                expires_at: far,
                            },
                        )
                    })
                    .collect();
                let me = &self.ids[i];
                let go = match self.rule {
                    Rule::TieBreak => {
                        shrink_recheck(&cov, &live, me, VACATE, self.r)
                    }
                    Rule::CountAll => {
                        patient_recheck(&cov, &live, me, VACATE, self.r)
                            == Recheck::Proceed
                    }
                    Rule::Patient => {
                        match patient_recheck(&cov, &live, me, VACATE, self.r) {
                            Recheck::Proceed => true,
                            Recheck::StandDown => false,
                            Recheck::Wait => {
                                intent[i] = Some(t + 1);
                                continue;
                            }
                        }
                    }
                };
                intent[i] = None;
                if go {
                    holds &= !bit(i);
                    departed |= bit(i);
                    if holds.count_ones() < self.r {
                        return Outcome {
                            violation: Some(t),
                            departed: departed.count_ones(),
                        };
                    }
                }
            }
            // 2. Death.
            if let Some((c, tc)) = crash
                && tc == t
                && holds & bit(c) != 0
            {
                holds &= !bit(c);
                intent[c] = None;
                died_at = Some((c, t));
            }
            // 3. Announcements.
            for i in 0..n {
                if ann[i] == Some(t)
                    && holds & bit(i) != 0
                    && intent[i].is_none()
                    && departed & bit(i) == 0
                {
                    intent[i] = Some(t + self.d);
                }
            }
            // 4. Publish.
            let intenders = (0..n)
                .filter(|&j| intent[j].is_some())
                .fold(0, |m, j| m | bit(j));
            history.push((holds, intenders));
        }
        Outcome {
            violation: None,
            departed: departed.count_ones(),
        }
    }

    /// Enumerate every schedule; call `f` on each.
    fn for_each_schedule(
        &self,
        crash_mode: bool,
        mut f: impl FnMut(&[Option<usize>], &[usize], Outcome),
    ) {
        let n = self.n;
        // Covers every relative offset that matters.
        let span = self.d + 2 * self.max_lag + 1;
        let crashes: Vec<Option<(usize, usize)>> = if crash_mode {
            (0..n)
                .flat_map(|c| (0..span + self.d).map(move |t| Some((c, t))))
                .collect()
        } else {
            vec![None]
        };
        // Odometers: announce option 0 = never, k = tick k - 1.
        let mut a_idx = vec![0usize; n];
        loop {
            if a_idx.iter().any(|&a| a != 0) {
                let ann: Vec<Option<usize>> =
                    a_idx.iter().map(|&a| a.checked_sub(1)).collect();
                let mut l_idx = vec![0usize; n];
                loop {
                    let lags: Vec<usize> =
                        l_idx.iter().map(|&l| l + 1).collect();
                    for &cr in &crashes {
                        f(&ann, &lags, self.run(&ann, &lags, cr));
                    }
                    if !odometer(&mut l_idx, self.max_lag) {
                        break;
                    }
                }
            }
            if !odometer(&mut a_idx, span + 1) {
                break;
            }
        }
    }

    /// (schedules, violations) over the whole space.
    fn check(&self, crash_mode: bool) -> (u64, u64) {
        let (mut count, mut bad) = (0, 0);
        self.for_each_schedule(crash_mode, |_, _, o| {
            count += 1;
            bad += o.violation.is_some() as u64;
        });
        (count, bad)
    }
}

/// Advance a base-`base` counter; false once it wraps to all zeros.
fn odometer(digits: &mut [usize], base: usize) -> bool {
    for d in digits.iter_mut() {
        *d += 1;
        if *d < base {
            return true;
        }
        *d = 0;
    }
    false
}

/// The staggered cascade, replayed on the real V3 code: nodes announce in
/// *descending* id order one tick apart, all two ticks stale, with a wait
/// of 2x lag. Each counts a higher-id announcer as its survivor; that
/// announcer leaves inside its blind window; the last copy goes. Found
/// by the Python checker as the first counterexample at N=4, R=1, L=2,
/// D=4. V6, on the same schedule, keeps the copy.
#[test]
fn staggered_cascade_breaks_v3_not_v6() {
    let ann = [Some(3), Some(2), Some(1), Some(0)];
    let lags = [2, 2, 2, 2];
    let v3 = World::new(4, 1, 2, 4, Rule::TieBreak, 0).run(&ann, &lags, None);
    assert_eq!(Some(7), v3.violation, "V3 should lose the last copy");
    let v6 = World::new(4, 1, 2, 4, Rule::Patient, 0).run(&ann, &lags, None);
    assert_eq!(None, v6.violation);
    assert!(v6.departed >= 1, "V6 should still let someone leave");
}

/// The real V3 code fails the same number of schedules as the Python
/// checker, at the configurations the report records as violated.
#[test]
fn v3_violation_counts_match_python_model() {
    // (n, r, max_lag, d) -> (schedules, violations) from timed_check.py.
    for ((n, r, l, d), expect) in [
        ((3, 1, 2, 2), (4_088, 126)),
        ((4, 1, 2, 4), (159_984, 6)),
        ((4, 2, 2, 4), (159_984, 0)),
    ] {
        let got = World::new(n, r, l, d, Rule::TieBreak, 0).check(false);
        assert_eq!(expect, got, "V3 at N={n} R={r} L={l} D={d}");
    }
}

/// V6 is safe at a wait of 1x lag over every schedule, where V3 is not
/// (126 failures at N=3, R=1, above). Counts match the Python checker.
#[test]
fn v6_safe_over_every_schedule_at_one_lag_wait() {
    for ((n, r, l, d), schedules) in [
        ((3, 1, 2, 2), 4_088),
        ((4, 1, 2, 2), 65_520),
        ((4, 2, 2, 2), 65_520),
    ] {
        let got = World::new(n, r, l, d, Rule::Patient, 0).check(false);
        assert_eq!((schedules, 0), got, "V6 at N={n} R={r} L={l} D={d}");
    }
}

/// One death at any tick, detected before the next decision: V3 loses
/// the sector in 2,820 of 110,376 schedules; V6 in none.
///
/// The Python checker counts 2,856 for V3. The 36 extra are schedules
/// where a dead node's intent outlives the detection of its death: the
/// real code discounts it twice, the Python model once. With the Python
/// view the counts agree exactly.
#[test]
fn prompt_death_breaks_v3_not_v6() {
    let v3 = World::new(3, 1, 2, 2, Rule::TieBreak, 0).check(true);
    assert_eq!((110_376, 2_820), v3);
    let v3_py = World::new(3, 1, 2, 2, Rule::TieBreak, 0)
        .python_view()
        .check(true);
    assert_eq!((110_376, 2_856), v3_py);
    let v6 = World::new(3, 1, 2, 2, Rule::Patient, 0).check(true);
    assert_eq!((110_376, 0), v6);
}

/// Progress. Whenever a node announces and the sector has a copy to
/// spare, V6 ends with at least one node vacated — the lowest racer
/// waits instead of cancelling. Counting every intender without the
/// wait (V5) stalls: everybody cancels in a large share of schedules.
#[test]
fn v6_always_makes_progress_where_count_all_stalls() {
    for (n, r, l, d) in [(3, 1, 2, 2), (4, 2, 2, 2), (4, 1, 2, 2)] {
        let mut v6_stalls = 0u64;
        World::new(n, r, l, d, Rule::Patient, 0).for_each_schedule(
            false,
            |ann, _, o| {
                if ann.iter().any(Option::is_some) && o.departed == 0 {
                    v6_stalls += 1;
                }
            },
        );
        assert_eq!(0, v6_stalls, "V6 stalled at N={n} R={r} L={l} D={d}");

        let mut v5_stalls = 0u64;
        World::new(n, r, l, d, Rule::CountAll, 0).for_each_schedule(
            false,
            |ann, _, o| {
                if ann.iter().any(Option::is_some) && o.departed == 0 {
                    v5_stalls += 1;
                }
            },
        );
        assert!(v5_stalls > 0, "expected V5 to stall at N={n} R={r}");
    }
}

/// The large configuration the report quotes: N=4, R=2, L=2, wait 2.5x
/// lag, one death at any tick, detected at once — 14,054,400 schedules,
/// all safe under V6; V3 fails. Slow in debug builds; run with
/// `cargo test --release -- --ignored`.
#[test]
#[ignore = "slow: 14M schedules; run with --release --ignored"]
fn v6_safe_with_prompt_death_large() {
    let v6 = World::new(4, 2, 2, 5, Rule::Patient, 0).check(true);
    assert_eq!((14_054_400, 0), v6);
    let v3 = World::new(4, 2, 2, 5, Rule::TieBreak, 0).check(true);
    assert!(v3.1 > 0, "V3 should fail some schedule");
}
