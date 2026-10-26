//! The staggered cascade, forced on a real network.
//!
//! The stale-view model check (`spec/timed_check.py` in
//! ValiChord/polite-shrink, and its Rust port in the controller's tests)
//! finds that the lower-id tie-break (V3) can lose the last copy of a
//! sector when nodes announce in *descending* id order, a little apart,
//! on views older than the gaps between them. Each node counts a
//! higher-id announcer as its survivor; that announcer has already left
//! inside the node's blind window. Random runs rarely hit the schedule,
//! so this test forces it, over real gossip on the in-memory transport:
//!
//! - four nodes, one agent each, whose home sectors all lie in the same
//!   half of the ring, so all four would vacate the *other* half;
//! - target redundancy 1, so that half may lose three of its four
//!   holders, but not all four;
//! - every node acts on a view of the network 2 s old
//!   ([`set_sharding_view_delay`]);
//! - each agent announces exactly once, in descending id order, 1 s
//!   apart ([`set_sharding_announce_schedule`]), and waits 4 s before
//!   re-checking.
//!
//! In the model this exact schedule (4 nodes, R = 1, lag 2, wait 4,
//! announces at 3, 2, 1, 0) loses the sector under V3 and not under V6.
//! Each step has about a second of margin, so tick jitter cannot change
//! the outcome.
//!
//! The real controller has a defence the model lacks: growers count
//! announced vacates as already gone, so the first node to leave sees the
//! later intents and grows straight back. That rescues V3 only if
//! re-growing finishes before the last holder leaves, about 3 s here. In
//! this test re-growing is near instant (a handful of tiny ops to sync);
//! in a real network, re-acquiring half the ring's data is not. So
//! re-growth speed is a parameter: grow persistence times a fixed 0.5 s
//! lag is the delay before a leaver starts to re-grow.
//!
//! Controls: V3 on fresh views (no cascade without staleness), and V3
//! with fast re-growth (the grow rule masks the cascade).
#![cfg(feature = "sharding")]

mod common;

use common::*;
use kitsune2_api::{AgentId, DhtArc, Timestamp};
use kitsune2_dht::SECTOR_SIZE;
use kitsune2_gossip::harness::{
    K2GossipFunctionalTestFactory, set_sharding_announce_schedule,
    set_sharding_view_delay,
};
use kitsune2_gossip::{K2GossipConfig, K2ShardingConfig};
use kitsune2_test_utils::space::TEST_SPACE_ID;
use std::time::Duration;

const NUM_NODES: usize = 4;
const STALENESS: Duration = Duration::from_millis(2_000);
const WAIT: Duration = Duration::from_millis(4_000);
const STAGGER: Duration = Duration::from_millis(1_000);
const WATCH: Duration = Duration::from_secs(20);

#[derive(Debug)]
struct Outcome {
    /// Lowest declared coverage of any sector of the shared vacate half
    /// while watched.
    min_coverage: u32,
    /// How many of the four executed their shrink.
    executed: usize,
    stood_down: usize,
    waited: usize,
    grew: usize,
    /// Agent ids in schedule order.
    order: Vec<String>,
    /// This run's sharding events for these agents, in order.
    log: Vec<(String, String)>,
}

async fn cascade_run(
    patient_lowest: bool,
    stale: bool,
    withdraw: bool,
    grow_persistence: f64,
) -> Outcome {
    cascade_run_ordered(
        patient_lowest,
        stale,
        withdraw,
        grow_persistence,
        false,
    )
    .await
}

async fn cascade_run_ordered(
    patient_lowest: bool,
    stale: bool,
    withdraw: bool,
    grow_persistence: f64,
    ascending: bool,
) -> Outcome {
    let events = events();
    events.lock().unwrap().clear();
    set_sharding_view_delay(Duration::ZERO);
    // Nobody announces until the schedule says so.
    set_sharding_announce_schedule(Some(Vec::new()));

    let factory = K2GossipFunctionalTestFactory::create(
        TEST_SPACE_ID,
        // No bootstrap: nodes discarded below must stay unknown to the
        // others, or their full arcs would count as extra holders.
        false,
        Some(K2GossipConfig {
            initiate_interval_ms: 250,
            min_initiate_interval_ms: 250,
            initial_initiate_interval_ms: 100,
            initiate_jitter_ms: 100,
            ..Default::default()
        }),
    )
    .await
    .with_sharding_config(K2ShardingConfig {
        target_redundancy: 1,
        clamp_min_peers: 2,
        check_interval_ms: 100,
        grow_persistence,
        // The schedule, not hysteresis, decides when to announce.
        shrink_persistence: 0.0,
        intent_wait: 0.0,
        intent_min_wait_ms: WAIT.as_millis() as u32,
        // Fixed lag, so grow persistence maps to a fixed re-grow delay.
        lag_floor_ms: 500,
        lag_ceiling_ms: 500,
        agentinfo_encoding: false,
        patient_lowest,
        withdraw_on_stand_down: withdraw,
        require_recent_contact: recent_contact(),
        announce_alongside: true,
        coordinated_growth: coordinated_growth(),
        critical_repair: true,
        pinned: false,
    });

    // Keep creating single-agent nodes until four have their home sector
    // in the same half of the ring.
    let half = NUM_SECTORS / 2;
    let mut buckets: [Vec<_>; 2] = [Vec::new(), Vec::new()];
    let nodes = loop {
        let node = factory.new_instance().await;
        let info = node.join_local_agent(DhtArc::FULL).await;
        let side = ((info.agent.loc() / SECTOR_SIZE) >= half) as usize;
        buckets[side].push((node, info));
        if buckets[side].len() == NUM_NODES {
            break std::mem::take(&mut buckets[side]);
        }
    };
    drop(buckets);
    // All four vacate the half their homes are not in.
    let home_side = (nodes[0].1.agent.loc() / SECTOR_SIZE) >= half;
    let (vs, ve) = if home_side {
        (0, half)
    } else {
        (half, NUM_SECTORS)
    };

    for (node, _) in &nodes {
        for (_, info) in &nodes {
            node.space
                .peer_store()
                .insert(vec![info.clone()])
                .await
                .unwrap();
        }
    }
    let harnesses = nodes.iter().map(|(n, _)| n).collect::<Vec<_>>();

    // Everyone at full arc, and the (stale) views have caught up.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    loop {
        let cov = coverage(&agents(&harnesses).await);
        if cov.iter().all(|&c| c == NUM_NODES as u32) {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "network never reached full arcs"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    if stale {
        set_sharding_view_delay(STALENESS);
    }
    tokio::time::sleep(STALENESS + Duration::from_secs(1)).await;

    // Descending id order, STAGGER apart.
    let mut ids: Vec<AgentId> =
        nodes.iter().map(|(_, i)| i.agent.clone()).collect();
    ids.sort();
    if !ascending {
        ids.reverse();
    }
    let t0 = Timestamp::now() + Duration::from_millis(500);
    let schedule = ids
        .iter()
        .enumerate()
        .map(|(k, id)| (id.clone(), t0 + STAGGER * k as u32))
        .collect();
    set_sharding_announce_schedule(Some(schedule));

    let mut min_coverage = u32::MAX;
    let end = tokio::time::Instant::now() + WATCH;
    while tokio::time::Instant::now() < end {
        let cov = coverage(&agents(&harnesses).await);
        min_coverage = min_coverage
            .min(*cov[vs as usize..ve as usize].iter().min().unwrap());
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    set_sharding_announce_schedule(None);
    set_sharding_view_delay(Duration::ZERO);

    let ours: std::collections::HashSet<String> =
        ids.iter().map(|id| format!("{id:?}")).collect();
    let log = events.lock().unwrap().clone();
    let count = |m: &str| {
        log.iter()
            .filter(|(a, msg)| ours.contains(a) && msg == m)
            .count()
    };
    let announced = count(ANNOUNCED);
    if !ascending {
        assert_eq!(
            NUM_NODES, announced,
            "the schedule should make each agent announce exactly once"
        );
    }
    Outcome {
        min_coverage,
        executed: count(EXECUTED),
        stood_down: count(STOOD_DOWN),
        waited: count(WAITED),
        grew: count(GREW),
        order: ids.iter().map(|id| format!("{id:?}")).collect(),
        log: log
            .iter()
            .filter(|(a, _)| ours.contains(a))
            .cloned()
            .collect(),
    }
}

/// Re-grow delay that outlasts the cascade: 16 x 0.5 s = 8 s.
const SLOW_REGROW: f64 = 16.0;
/// Near-instant re-grow: 1 x 0.5 s.
const FAST_REGROW: f64 = 1.0;

/// With re-growth slower than the cascade, V3 loses the last copy: all
/// four leave the half.
#[tokio::test(flavor = "multi_thread")]
async fn forced_cascade_breaks_tiebreak_v3() {
    let _serial = SERIAL.lock().await;
    let o = cascade_run(false, true, false, SLOW_REGROW).await;
    println!("V3, stale views, slow re-grow: {o:?}");
    assert_eq!(
        0, o.min_coverage,
        "expected the forced cascade to orphan the half under V3: {o:?}"
    );
}

/// The same schedule and re-grow delay under patient lowest (V6): the
/// half keeps a copy without relying on anyone growing back.
#[tokio::test(flavor = "multi_thread")]
async fn forced_cascade_spares_patient_lowest_v6() {
    let _serial = SERIAL.lock().await;
    let o = cascade_run(true, true, true, SLOW_REGROW).await;
    println!("V6, stale views, slow re-grow: {o:?}");
    assert!(
        o.min_coverage >= 1,
        "V6 lost the last copy on the forced cascade: {o:?}"
    );
    assert!(o.executed >= 1, "V6 should still let someone leave: {o:?}");
}

/// Control: the same schedule on fresh views. Without staleness each
/// node sees the departures before it re-checks, so V3 keeps a copy:
/// the cascade is a stale-view failure.
#[tokio::test(flavor = "multi_thread")]
async fn forced_cascade_needs_stale_views() {
    let _serial = SERIAL.lock().await;
    let o = cascade_run(false, false, false, SLOW_REGROW).await;
    println!("V3, fresh views, slow re-grow: {o:?}");
    assert!(
        o.min_coverage >= 1,
        "V3 lost the last copy even on fresh views: {o:?}"
    );
}

/// Control: with near-instant re-growth the grow rule masks the cascade
/// under V3 — the first leavers grow back before the last one goes.
#[tokio::test(flavor = "multi_thread")]
async fn forced_cascade_masked_by_fast_regrow() {
    let _serial = SERIAL.lock().await;
    let o = cascade_run(false, true, false, FAST_REGROW).await;
    println!("V3, stale views, fast re-grow: {o:?}");
    assert!(
        o.min_coverage >= 1 && o.grew >= 1,
        "expected fast re-growth to mask the cascade: {o:?}"
    );
}

/// The announce-time yield. Fresh views, V6, announcements scheduled in
/// ASCENDING id order 1 s apart: once the lowest has announced, the others
/// must not announce until its intent has resolved.
#[tokio::test(flavor = "multi_thread")]
async fn higher_ids_yield_while_a_lower_id_is_announcing() {
    let _serial = SERIAL.lock().await;
    let o = cascade_run_ordered(true, false, true, SLOW_REGROW, true).await;
    let lowest = &o.order[0];
    let mut open = false;
    for (a, m) in &o.log {
        if a == lowest && m == ANNOUNCED {
            open = true;
        } else if a == lowest && (m == EXECUTED || m == STOOD_DOWN) {
            open = false;
        } else if open && m == ANNOUNCED {
            panic!("agent {a} announced while the lowest id was announcing");
        }
    }
    assert!(
        o.log.iter().any(|(a, m)| a == lowest && m == ANNOUNCED),
        "the lowest id never announced: {o:?}"
    );
}

/// Sweep the re-grow delay for V3 and V6 on stale views, and print one
/// line per run. Slow; run with `--ignored --nocapture`, `CASCADE_RUNS`
/// runs per cell (default 3).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "slow sweep; run with --ignored --nocapture"]
async fn forced_cascade_regrow_sweep() {
    let _serial = SERIAL.lock().await;
    let runs: usize = std::env::var("CASCADE_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    println!(
        "cascade,arm,regrow_delay_s,run,min_coverage,executed,stood_down,\
         waited,grew"
    );
    for run in 1..=runs {
        for gp in [1.0, 2.0, 4.0, 6.0, 8.0, 16.0] {
            for (arm, patient, withdraw) in
                [("V3", false, false), ("V6", true, true)]
            {
                let o = cascade_run(patient, true, withdraw, gp).await;
                println!(
                    "cascade,{arm},{:.1},{run},{},{},{},{},{}",
                    gp * 0.5,
                    o.min_coverage,
                    o.executed,
                    o.stood_down,
                    o.waited,
                    o.grew
                );
            }
        }
    }
}
