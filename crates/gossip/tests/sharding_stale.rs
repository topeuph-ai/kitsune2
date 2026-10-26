//! Functional test of the shrink re-check under stale views, over real
//! gossip on the in-memory transport.
//!
//! The in-memory transport is near instant, so in the other functional
//! tests every node acts on a fresh view and the stale-view race that
//! breaks the lower-id tie-break (V3) cannot happen. Here every node acts
//! on a view of the network a fixed `STALENESS` old
//! ([`set_sharding_view_delay`]): peer arcs and announced intents both
//! from that instant.
//!
//! The model check (`spec/timed_check.py` in ValiChord/polite-shrink, and
//! its Rust port in the controller's tests) predicts that with an
//! announce-to-execute wait between one and two times the staleness:
//! - patient lowest (V6) never takes a sector below target (it is safe
//!   for any wait of at least one staleness);
//! - the tie-break (V3) can (it needs about twice the staleness).
//!
//! A third arm runs V6 with a wait *shorter* than the staleness, where no
//! rule is safe: if that arm never fails, the test has no teeth.
//!
//! Six nodes, one agent each, target 4, no deaths: from full arcs every
//! agent sees 5 copies of the half it would vacate, so only two of any
//! group vacating the same half may go.
#![cfg(feature = "sharding")]

mod common;

use common::*;
use kitsune2_api::DhtArc;
use kitsune2_gossip::harness::{
    K2GossipFunctionalTestFactory, set_sharding_view_delay,
};
use kitsune2_gossip::{K2GossipConfig, K2ShardingConfig};
use kitsune2_test_utils::space::TEST_SPACE_ID;
use std::time::Duration;

const NUM_NODES: usize = 6;
const TARGET_REDUNDANCY: u32 = 4;
const STALENESS: Duration = Duration::from_millis(2_000);
/// 1.5x the staleness: V6 should be safe, V3 need not be.
const WAIT_SAFE_FOR_V6: Duration = Duration::from_millis(3_000);
/// 0.5x the staleness: no rule is safe.
const WAIT_TOO_SHORT: Duration = Duration::from_millis(1_000);
/// How long to watch after every sector first reaches the target.
const WATCH: Duration = Duration::from_secs(45);

#[derive(Debug, Default)]
struct RunStats {
    /// Lowest declared coverage of any sector while watched.
    min_coverage: u32,
    /// Samples (every 50 ms) with some sector below target.
    samples_below: usize,
    /// Sector-samples below target, summed.
    sector_samples_below: usize,
    announced: usize,
    stood_down: usize,
    waited: usize,
    executed: usize,
    grew: usize,
    final_copies: u32,
}

async fn stale_run(
    patient_lowest: bool,
    wait: Duration,
    withdraw: bool,
) -> RunStats {
    let events = events();
    events.lock().unwrap().clear();
    set_sharding_view_delay(STALENESS);

    let factory = K2GossipFunctionalTestFactory::create(
        TEST_SPACE_ID,
        true,
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
        target_redundancy: TARGET_REDUNDANCY,
        clamp_min_peers: 2,
        check_interval_ms: 100,
        grow_persistence: 1.0,
        shrink_persistence: 4.0,
        // The measured gossip lag cannot see the artificial staleness,
        // so the wait is set outright through its floor.
        intent_wait: 0.0,
        intent_min_wait_ms: wait.as_millis() as u32,
        lag_floor_ms: 250,
        lag_ceiling_ms: 500,
        agentinfo_encoding: false,
        patient_lowest,
        withdraw_on_stand_down: withdraw,
        require_recent_contact: recent_contact(),
        announce_alongside: true,
        coordinated_growth: coordinated_growth(),
        critical_repair: true,
        pinned: false,
        probe_sibling_holders: true,
        probe_interval_ms: 1_000,
    });

    let mut nodes = Vec::with_capacity(NUM_NODES);
    let first = factory.new_instance().await;
    seed_ops(&first).await;
    first.join_local_agent(DhtArc::FULL).await;
    nodes.push(first);
    for _ in 1..NUM_NODES {
        let node = factory.new_instance().await;
        let info = node.join_local_agent(DhtArc::FULL).await;
        for other in &nodes {
            other
                .space
                .peer_store()
                .insert(vec![info.clone()])
                .await
                .unwrap();
        }
        nodes.push(node);
    }
    let all = nodes.iter().collect::<Vec<_>>();

    // Arm once every sector first reaches the target; watch from there.
    let arm_deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    loop {
        let cov = coverage(&agents(&all).await);
        if *cov.iter().min().unwrap() >= TARGET_REDUNDANCY {
            break;
        }
        assert!(
            tokio::time::Instant::now() < arm_deadline,
            "network never reached the target redundancy"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let armed_log_len = events.lock().unwrap().len();

    let mut stats = RunStats {
        min_coverage: u32::MAX,
        ..Default::default()
    };
    let watch_end = tokio::time::Instant::now() + WATCH;
    let mut cov = Vec::new();
    while tokio::time::Instant::now() < watch_end {
        cov = coverage(&agents(&all).await);
        stats.min_coverage = stats.min_coverage.min(*cov.iter().min().unwrap());
        let below = cov.iter().filter(|&&c| c < TARGET_REDUNDANCY).count();
        if below > 0 {
            stats.samples_below += 1;
            stats.sector_samples_below += below;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    stats.final_copies = cov.iter().sum();

    let ours: std::collections::HashSet<String> =
        agents(&all).await.into_iter().map(|(id, ..)| id).collect();
    for (i, (agent, msg)) in events.lock().unwrap().iter().enumerate() {
        if !ours.contains(agent) || (msg == GREW && i < armed_log_len) {
            continue;
        }
        match msg.as_str() {
            ANNOUNCED => stats.announced += 1,
            WAITED => stats.waited += 1,
            EXECUTED => stats.executed += 1,
            STOOD_DOWN => stats.stood_down += 1,
            GREW => stats.grew += 1,
            _ => {}
        }
    }
    set_sharding_view_delay(Duration::ZERO);
    stats
}

/// V6 with a wait of 1.5x the staleness never takes a sector below
/// target, and still shards down: once without and once with intent
/// withdrawal (the default), which must not weaken safety.
#[tokio::test(flavor = "multi_thread")]
async fn stale_views_patient_lowest_stays_safe() {
    let _serial = SERIAL.lock().await;
    for (run, withdraw) in [(1, false), (2, true)] {
        let s = stale_run(true, WAIT_SAFE_FOR_V6, withdraw).await;
        println!("V6 stale views, run {run}: {s:?}");
        assert_eq!(
            0, s.samples_below,
            "V6 took a sector below target on a stale view: {s:?}"
        );
        assert!(s.executed >= 1, "V6 never shrank: {s:?}");
    }
}

/// V3 and V6 side by side under stale views, plus V6 with a wait too
/// short for the staleness as a control that the test can see a
/// failure. Prints one CSV line per run. Slow; run with
/// `--ignored --nocapture`, `STALE_RUNS` runs per arm (default 5).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "slow comparison; run with --ignored --nocapture"]
async fn stale_views_compare() {
    let _serial = SERIAL.lock().await;
    let runs: usize = std::env::var("STALE_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    println!(
        "stale,arm,run,min_coverage,samples_below,sector_samples_below,\
         announced,stood_down,waited,executed,grew,final_copies"
    );
    for run in 1..=runs {
        for (arm, patient, wait, withdraw) in [
            ("V3", false, WAIT_SAFE_FOR_V6, false),
            ("V3+withdraw", false, WAIT_SAFE_FOR_V6, true),
            ("V6", true, WAIT_SAFE_FOR_V6, false),
            ("V6+withdraw", true, WAIT_SAFE_FOR_V6, true),
            ("V6-short-wait", true, WAIT_TOO_SHORT, false),
        ] {
            let s = stale_run(patient, wait, withdraw).await;
            println!(
                "stale,{arm},{run},{},{},{},{},{},{},{},{},{}",
                s.min_coverage,
                s.samples_below,
                s.sector_samples_below,
                s.announced,
                s.stood_down,
                s.waited,
                s.executed,
                s.grew,
                s.final_copies
            );
        }
    }
}
