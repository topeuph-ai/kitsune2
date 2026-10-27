//! Table of announced shrink intents.
//!
//! Holds both remote intents received on the `k2sharding` channel and
//! this node's own local agents' pending intents (local agents must see
//! each other's announcements the same way remote peers do). Intents
//! expire at the timestamp their announcer attached; an executed shrink
//! shows up as the shrunk arc in the peer store well before that, and an
//! abandoned one ages out or is withdrawn by its announcer.
//!
//! Remote intents are keyed by the transport peer they arrived from as
//! well as the agent they name, and only the entry from the agent's own
//! declared peer URL is consumed (see [`retain_declared`]). Without that,
//! a later announcement replacing an earlier one would let anyone
//! displace a genuine intent with a short-lived forged one in the same
//! agent's name; once the forgery expired, peers would count a node that
//! is really leaving as staying, which is the unsafe direction.

use super::blocks::NUM_SECTORS;
use kitsune2_api::{AgentId, DhtArc, Timestamp, Url};
use kitsune2_dht::SECTOR_SIZE;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

/// One announced shrink intent.
#[derive(Debug, Clone)]
pub(crate) struct ShrinkIntent {
    /// Half-open sector range being vacated.
    pub vacate: (u32, u32),
    /// When this intent is void.
    pub expires_at: Timestamp,
    /// The transport peer it arrived from; `None` for this node's own
    /// local agents.
    pub from: Option<Url>,
}

/// Shared, expiring table of announced shrink intents, keyed by intender
/// and by the peer the announcement came from.
///
/// A later announcement from the same agent and peer replaces the earlier
/// one; an agent can only have one shrink in flight.
#[derive(Debug, Default)]
pub(crate) struct IntentTable {
    inner: Mutex<HashMap<(AgentId, Option<Url>), ShrinkIntent>>,
}

impl IntentTable {
    pub fn insert(&self, agent: AgentId, intent: ShrinkIntent) {
        let key = (agent, intent.from.clone());
        self.inner.lock().expect("poisoned").insert(key, intent);
    }

    /// Remove a local agent's own intent.
    pub fn remove(&self, agent: &AgentId) {
        self.inner
            .lock()
            .expect("poisoned")
            .remove(&(agent.clone(), None));
    }

    /// Apply a withdrawal: remove the intent `agent` announced from
    /// `from` only if it is the one being withdrawn, identified by its
    /// expiry. A withdrawal that arrives after a newer announcement (with
    /// a later expiry) leaves the newer one in place.
    pub fn withdraw(
        &self,
        agent: AgentId,
        from: Option<Url>,
        expires_at: Timestamp,
    ) -> bool {
        let mut lock = self.inner.lock().expect("poisoned");
        let key = (agent, from);
        if lock.get(&key).is_some_and(|i| i.expires_at == expires_at) {
            lock.remove(&key);
            return true;
        }
        false
    }

    /// Drop expired intents and return the live ones.
    pub fn live(&self, now: Timestamp) -> Vec<(AgentId, ShrinkIntent)> {
        let mut lock = self.inner.lock().expect("poisoned");
        lock.retain(|_, i| i.expires_at > now);
        lock.iter()
            .map(|((a, _), i)| (a.clone(), i.clone()))
            .collect()
    }
}

/// Receiver-side range-validation: keep only intents whose vacate range
/// lies inside the announcer's currently *declared* arc, as this node
/// sees it in its peer store. Local agents bypass the check — their
/// intents are made here, not received, and their declared arc may
/// briefly lag their own executed shrink.
///
/// Why: intents are consumed in the coverage-lowering direction
/// (growers subtract announced vacates; executing shrinkers count
/// higher-priority intenders as gone), so a forged intent can talk
/// declared coverage down over sectors its claimed sender never held.
/// That cannot orphan data, but it is a cheap resource lever — the
/// polite-shrink defense study measured a whole-ring forgery pinning
/// arcs wide at 2.3x sync cost, and this filter reducing the residual
/// to ~4%. The filter also drops stale intents whose announcer already
/// executed the shrink (the vacated range has left their declared arc),
/// which would otherwise be double-counted against coverage.
///
/// An intent from an agent absent from the peer store cannot be
/// validated and is dropped, as is one that did not arrive from the
/// agent's own declared peer URL (see the module docs). Local agents'
/// intents are kept only from this node itself.
pub(crate) fn retain_declared(
    live: &mut Vec<(AgentId, ShrinkIntent)>,
    declared: &HashMap<AgentId, (DhtArc, Option<Url>)>,
    own_ids: &HashSet<AgentId>,
) {
    live.retain(|(agent, intent)| {
        if own_ids.contains(agent) {
            return intent.from.is_none();
        }
        declared.get(agent).is_some_and(|(arc, url)| {
            url.is_some()
                && *url == intent.from
                && arc_contains_sectors(arc, intent.vacate)
        })
    });
}

/// Keep only grow intents that come from their agent's declared URL and
/// name the half adjoining the arc that agent declares: the only range an
/// agent can grow into. Intents from agents in `gone` (marked
/// unresponsive) are dropped: a dead grower brings no copy. Our own
/// agents' intents are kept as recorded.
pub(crate) fn retain_growers(
    live: &mut Vec<(AgentId, ShrinkIntent)>,
    declared: &HashMap<AgentId, (DhtArc, Option<Url>)>,
    own_ids: &HashSet<AgentId>,
    gone: &HashSet<AgentId>,
) {
    live.retain(|(agent, intent)| {
        if own_ids.contains(agent) {
            return intent.from.is_none();
        }
        if gone.contains(agent) {
            return false;
        }
        declared.get(agent).is_some_and(|(arc, url)| {
            url.is_some()
                && *url == intent.from
                && is_sibling_of(arc, intent.vacate)
        })
    });
}

/// True if `range` is the half adjoining `arc` within the aligned block
/// twice its size: the sibling an agent holding `arc` would grow into.
fn is_sibling_of(arc: &DhtArc, (start, end): (u32, u32)) -> bool {
    let DhtArc::Arc(a, b) = *arc else {
        return false;
    };
    if b < a || start >= end || end > NUM_SECTORS {
        return false;
    }
    let (s, e) = (a / SECTOR_SIZE, b / SECTOR_SIZE + 1);
    let len = e - s;
    if end - start != len || !len.is_power_of_two() {
        return false;
    }
    let parent = len * 2;
    let lo = s.min(start);
    (start == e || end == s) && lo % parent == 0
}

/// True if every sector in the half-open range `[start, end)` lies fully
/// within `arc`. Handles wrapping arcs by checking per sector, the same
/// containment rule the coverage counters use.
fn arc_contains_sectors(arc: &DhtArc, (start, end): (u32, u32)) -> bool {
    if start >= end || end > NUM_SECTORS {
        return false;
    }
    (start..end).all(|sector| {
        let lo = sector * SECTOR_SIZE;
        arc.contains(lo) && arc.contains(lo + (SECTOR_SIZE - 1))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn agent(b: &'static [u8]) -> AgentId {
        AgentId::from(bytes::Bytes::from_static(b))
    }

    /// An intent is live strictly before its expiry, not at it.
    #[test]
    fn intent_is_gone_at_its_expiry_instant() {
        let table = IntentTable::default();
        let now = Timestamp::now();
        table.insert(
            agent(b"a"),
            ShrinkIntent {
                vacate: (0, 8),
                expires_at: now,
                from: None,
            },
        );
        assert!(table.live(now).is_empty());
    }

    #[test]
    fn expired_intents_age_out() {
        let table = IntentTable::default();
        let now = Timestamp::now();
        table.insert(
            agent(b"a"),
            ShrinkIntent {
                vacate: (0, 8),
                expires_at: now + Duration::from_secs(10),
                from: None,
            },
        );
        table.insert(
            agent(b"b"),
            ShrinkIntent {
                vacate: (8, 16),
                expires_at: (now - Duration::from_secs(1)).unwrap(),
                from: None,
            },
        );

        let live = table.live(now);
        assert_eq!(1, live.len());
        assert_eq!(agent(b"a"), live[0].0);
    }

    fn url(s: &str) -> Option<Url> {
        Some(Url::from_str(format!("ws://{s}:80")).unwrap())
    }

    /// A remote intent, as received from `url("peer")`.
    fn intent(vacate: (u32, u32)) -> ShrinkIntent {
        ShrinkIntent {
            vacate,
            expires_at: Timestamp::now() + Duration::from_secs(60),
            from: url("peer"),
        }
    }

    fn local(vacate: (u32, u32)) -> ShrinkIntent {
        ShrinkIntent {
            from: None,
            ..intent(vacate)
        }
    }

    #[test]
    fn range_validation_drops_forged_stale_and_unknown_intents() {
        let sector = |s: u32| s * SECTOR_SIZE;
        // "small" declares sectors [0, 16); "ghost" is not in the store.
        let declared: HashMap<AgentId, (DhtArc, Option<Url>)> = [(
            agent(b"small"),
            (DhtArc::Arc(0, sector(16) - 1), url("peer")),
        )]
        .into();
        let own: HashSet<AgentId> = [agent(b"me")].into();

        let mut live = vec![
            // inside the declared arc: kept
            (agent(b"small"), intent((8, 16))),
            // whole-ring forgery in small's name: dropped
            (agent(b"small"), intent((0, NUM_SECTORS))),
            // outside the declared arc (forged, or stale after an
            // executed shrink): dropped
            (agent(b"small"), intent((16, 24))),
            // announcer unknown to the peer store: dropped
            (agent(b"ghost"), intent((0, 8))),
            // own agents bypass validation
            (agent(b"me"), local((0, 8))),
        ];
        retain_declared(&mut live, &declared, &own);

        let kept: Vec<_> =
            live.iter().map(|(a, i)| (a.clone(), i.vacate)).collect();
        assert_eq!(
            vec![(agent(b"small"), (8, 16)), (agent(b"me"), (0, 8))],
            kept
        );
    }

    #[test]
    fn grow_intents_must_name_the_adjoining_half() {
        let sector = |s: u32| s * SECTOR_SIZE;
        // "small" declares the aligned block [16, 24); its sibling within
        // the parent block [16, 32) is [24, 32).
        let declared: HashMap<AgentId, (DhtArc, Option<Url>)> = [(
            agent(b"small"),
            (DhtArc::Arc(sector(16), sector(24) - 1), url("peer")),
        )]
        .into();
        let own: HashSet<AgentId> = [agent(b"me")].into();
        let mut live = vec![
            // the sibling half: kept
            (agent(b"small"), intent((24, 32))),
            // adjoining but in the wrong parent block ([8, 16) belongs
            // with [0, 8)): dropped
            (agent(b"small"), intent((8, 16))),
            // the right place but the wrong size: dropped
            (agent(b"small"), intent((24, 40))),
            // its own arc (not a grow at all): dropped
            (agent(b"small"), intent((16, 24))),
            // unknown announcer: dropped
            (agent(b"ghost"), intent((24, 32))),
            // own agents bypass validation
            (agent(b"me"), local((0, 8))),
        ];
        retain_growers(&mut live, &declared, &own, &HashSet::new());
        let kept: Vec<_> =
            live.iter().map(|(a, i)| (a.clone(), i.vacate)).collect();
        assert_eq!(
            vec![(agent(b"small"), (24, 32)), (agent(b"me"), (0, 8))],
            kept
        );
        // The lower half of a parent block grows upward too: [0, 8) -> [8, 16).
        assert!(is_sibling_of(&DhtArc::Arc(0, sector(8) - 1), (8, 16)));
        // A full arc has nothing to grow into.
        assert!(!is_sibling_of(&DhtArc::FULL, (0, NUM_SECTORS)));
        // A wrong sender is dropped.
        let mut wrong = vec![(
            agent(b"small"),
            ShrinkIntent {
                from: url("other"),
                ..intent((24, 32))
            },
        )];
        retain_growers(&mut wrong, &declared, &own, &HashSet::new());
        assert!(wrong.is_empty());
        // A grower marked unresponsive is dropped; our own agents (which
        // the controller also puts in `gone`) are not.
        let mut dead = vec![
            (agent(b"small"), intent((24, 32))),
            (agent(b"me"), local((0, 8))),
        ];
        let gone: HashSet<AgentId> = [agent(b"small"), agent(b"me")].into();
        retain_growers(&mut dead, &declared, &own, &gone);
        assert_eq!(
            vec![agent(b"me")],
            dead.into_iter().map(|(a, _)| a).collect::<Vec<_>>()
        );
    }

    #[test]
    fn range_validation_handles_wrapping_arcs() {
        let sector = |s: u32| s * SECTOR_SIZE;
        // wraps: sectors [500, 512) and [0, 4)
        let declared: HashMap<AgentId, (DhtArc, Option<Url>)> = [(
            agent(b"w"),
            (DhtArc::Arc(sector(500), sector(4) - 1), url("peer")),
        )]
        .into();
        let own = HashSet::new();

        let mut live = vec![
            (agent(b"w"), intent((500, 512))), // fully inside: kept
            (agent(b"w"), intent((2, 6))),     // straddles the gap: dropped
        ];
        retain_declared(&mut live, &declared, &own);
        assert_eq!(1, live.len());
        assert_eq!((500, 512), live[0].1.vacate);
    }

    #[test]
    fn reannouncement_replaces() {
        let table = IntentTable::default();
        let now = Timestamp::now();
        for vacate in [(0, 8), (8, 16)] {
            table.insert(
                agent(b"a"),
                ShrinkIntent {
                    vacate,
                    expires_at: now + Duration::from_secs(10),
                    from: None,
                },
            );
        }
        let live = table.live(now);
        assert_eq!(1, live.len());
        assert_eq!((8, 16), live[0].1.vacate);
    }
    /// The displacement attack: a forged, short-lived intent in a leaving
    /// agent's name, sent from another peer. Keyed by sender, it cannot
    /// replace the genuine intent, and it is not consumed because it did
    /// not come from the agent's declared URL. Once it expires, the
    /// genuine intent is still there.
    #[test]
    fn forged_intent_cannot_displace_genuine_one() {
        let sector = |s: u32| s * SECTOR_SIZE;
        let table = IntentTable::default();
        let now = Timestamp::now();
        table.insert(
            agent(b"leaver"),
            ShrinkIntent {
                vacate: (8, 16),
                expires_at: now + Duration::from_secs(60),
                from: url("leaver"),
            },
        );
        table.insert(
            agent(b"leaver"),
            ShrinkIntent {
                vacate: (8, 16),
                expires_at: now + Duration::from_millis(1),
                from: url("forger"),
            },
        );
        let declared: HashMap<AgentId, (DhtArc, Option<Url>)> = [(
            agent(b"leaver"),
            (DhtArc::Arc(0, sector(16) - 1), url("leaver")),
        )]
        .into();

        let mut live = table.live(now);
        retain_declared(&mut live, &declared, &HashSet::new());
        assert_eq!(1, live.len());
        assert_eq!(url("leaver"), live[0].1.from);

        let later = now + Duration::from_secs(1);
        let mut live = table.live(later);
        retain_declared(&mut live, &declared, &HashSet::new());
        assert_eq!(1, live.len(), "genuine intent must survive the forgery");
    }

    /// A withdrawal removes exactly the intent it names: not one from
    /// another peer, and not a newer announcement it arrived after.
    #[test]
    fn withdrawal_removes_only_the_named_intent() {
        let table = IntentTable::default();
        let now = Timestamp::now();
        let old = now + Duration::from_secs(10);
        let new = now + Duration::from_secs(20);
        let announce = |expires_at| ShrinkIntent {
            vacate: (8, 16),
            expires_at,
            from: url("leaver"),
        };

        // Forged withdrawal from another peer: no effect.
        table.insert(agent(b"leaver"), announce(old));
        table.withdraw(agent(b"leaver"), url("forger"), old);
        assert_eq!(1, table.live(now).len());

        // Late withdrawal of the old intent after a newer one: no effect.
        table.insert(agent(b"leaver"), announce(new));
        table.withdraw(agent(b"leaver"), url("leaver"), old);
        assert_eq!(new, table.live(now)[0].1.expires_at);

        // Withdrawal of the current intent from its sender: removed.
        table.withdraw(agent(b"leaver"), url("leaver"), new);
        assert!(table.live(now).is_empty());
    }
}
