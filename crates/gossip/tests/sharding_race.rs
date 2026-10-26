//! Functional test for the patient-lowest (V6) re-check under a forced
//! shrink race, over real gossip on the in-memory transport.
//!
//! The 8-node storm test never produces a race thin enough to make V6's
//! lowest racer wait: two intenders rarely overlap, and when they do the
//! margin covers both. This test forces one.
//!
//! - **Simultaneity.** Six agents share one node. A controller tick takes
//!   one snapshot of the intent table and then decides for each local
//!   agent in turn, so agents that announce in the same tick cannot see
//!   each other's intents: a true race, which the announce-time yield
//!   cannot prevent. Two more agents on a second node give the first
//!   node a gossip partner, and see the race over the wire.
//! - **A thin margin.** Eight agents, target redundancy 6. From full
//!   arcs every agent sees 7 copies of the half it would vacate, so the
//!   shrink condition holds, but only one of any three overlapping
//!   intenders can go. By pigeonhole one half has at least four would-be
//!   vacaters.
//!
//! What is asserted, with no peer deaths anywhere:
//! - **Safety.** Once every sector has reached the target, no sector
//!   ever drops below it.
//! - **Progress.** The network settles, at least one shrink executes, and
//!   at the end no agent has a shrink left that it could safely take (no
//!   stall, the failure of counting every intender without waiting).
//! - **The race resolves as designed.** Every agent that waited at
//!   re-check later executed its shrink.
//!
//! Agent ids are random, so a run is not guaranteed to race thinly
//! enough to make anyone wait; the V6 test repeats the scenario until a
//! wait is observed, asserting safety and progress on every run.
#![cfg(feature = "sharding")]

mod common;

use common::*;
use kitsune2_api::{AgentInfoSigned, DhtArc};
use kitsune2_gossip::harness::K2GossipFunctionalTestFactory;
use kitsune2_gossip::{K2GossipConfig, K2ShardingConfig};
use kitsune2_test_utils::space::TEST_SPACE_ID;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
const AGENTS_ON_A: usize = 6;
const AGENTS_ON_B: usize = 2;
const NUM_AGENTS: u32 = (AGENTS_ON_A + AGENTS_ON_B) as u32;
const TARGET_REDUNDANCY: u32 = 6;
const MAX_V6_RUNS: usize = 6;
/// Declared coverage unchanged this long counts as settled: longer than
/// the slowest path from a stand-down to the next decision (shrink
/// persistence 4 x the 2 s lag ceiling, then an intent wait of 2.5 x it),
/// so an agent that could still shrink has had time to.
const SETTLE: Duration = Duration::from_secs(15);

#[derive(Debug, Default)]
struct RunStats {
    announced: usize,
    waited: usize,
    executed: usize,
    stood_down: usize,
    /// Grow decisions after every sector first reached the target.
    grew: usize,
    /// ... of which the sibling half was short on declared coverage.
    grew_short: usize,
    /// ... of which only announced (not yet executed) vacates made it
    /// look short.
    grew_intents: usize,
    /// Seconds from every sector first reaching the target to the last
    /// change in declared coverage.
    settle_secs: f64,
    /// Total declared sector-copies at the end, against 8 x the ring.
    final_copies: u32,
    /// False if the network was still changing at the deadline (only
    /// tolerated in the comparison, which records it).
    settled: bool,
}

/// Build the network, let it shard down under a forced race, and assert
/// safety, progress, and (if anyone waited) that waiters went on to
/// execute.
async fn race_run(
    patient_lowest: bool,
    withdraw: bool,
    require_settle: bool,
) -> RunStats {
    let events = events();
    events.lock().unwrap().clear();

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
        intent_wait: 2.5,
        intent_min_wait_ms: 1_000,
        lag_floor_ms: 250,
        lag_ceiling_ms: 2_000,
        agentinfo_encoding: false,
        patient_lowest,
        withdraw_on_stand_down: withdraw,
        require_recent_contact: recent_contact(),
        announce_alongside: true,
        coordinated_growth: coordinated_growth(),
        critical_repair: true,
        pinned: false,
        probe_sibling_holders: probe_holders(),
        probe_interval_ms: 1_000,
    });

    let a = factory.new_instance().await;
    seed_ops(&a).await;
    let b = factory.new_instance().await;

    let mut infos_a: Vec<Arc<AgentInfoSigned>> = Vec::new();
    for _ in 0..AGENTS_ON_A {
        infos_a.push(a.join_local_agent(DhtArc::FULL).await);
    }
    let mut infos_b: Vec<Arc<AgentInfoSigned>> = Vec::new();
    for _ in 0..AGENTS_ON_B {
        infos_b.push(b.join_local_agent(DhtArc::FULL).await);
    }
    // Help discovery along rather than waiting for bootstrap.
    a.space.peer_store().insert(infos_b).await.unwrap();
    b.space.peer_store().insert(infos_a).await.unwrap();

    let nodes = [&a, &b];

    // Safety, from the moment every sector first reaches the target to
    // the end of the run.
    let mut armed = false;
    let mut armed_at = tokio::time::Instant::now();
    let mut armed_log_len = 0;
    let mut last = Vec::new();
    let mut stable_since = tokio::time::Instant::now();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(300);
    let (settled, did_settle) = loop {
        let now_agents = agents(&nodes).await;
        let cov = coverage(&now_agents);
        let min = *cov.iter().min().unwrap();
        if armed {
            assert!(
                min >= TARGET_REDUNDANCY,
                "a sector fell to {min} declared holders, below the \
                 target {TARGET_REDUNDANCY}, with no peer deaths"
            );
        } else if min >= TARGET_REDUNDANCY {
            armed = true;
            armed_at = tokio::time::Instant::now();
            armed_log_len = events.lock().unwrap().len();
        }

        let executed =
            events.lock().unwrap().iter().any(|(_, m)| m == EXECUTED);
        if cov != last {
            last = cov;
            stable_since = tokio::time::Instant::now();
        } else if executed && stable_since.elapsed() >= SETTLE {
            break (now_agents, true);
        }
        if tokio::time::Instant::now() >= deadline {
            assert!(
                !require_settle,
                "network never settled after shrinking (executed any: \
                 {executed})"
            );
            break (now_agents, false);
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    };

    assert_eq!(NUM_AGENTS as usize, settled.len());

    // Progress: nobody is left holding a shrink it could safely take.
    let stuck = if did_settle {
        could_shrink(&settled, TARGET_REDUNDANCY)
    } else {
        Vec::new()
    };
    if !stuck.is_empty() {
        let log = events.lock().unwrap().clone();
        for id in &stuck {
            let mine: Vec<&String> = log
                .iter()
                .filter(|(a, _)| a == id)
                .map(|(_, m)| m)
                .collect();
            println!("could still shrink: {id}: {mine:?}");
        }
    }
    // The stall check is for V6. V3 is a comparison control here: it churns
    // (up to about a minute to settle in this race), so a 15 s quiet spell
    // can fall mid-churn with shrinks still to come; on the published V6
    // branch this check failed for V3 in 2 of 6 runs. Its safety is still
    // asserted above.
    if patient_lowest {
        assert!(
            stuck.is_empty(),
            "settled with {} agent(s) still able to shrink safely: a stall",
            stuck.len()
        );
    } else if !stuck.is_empty() {
        println!(
            "V3: {} agent(s) could still shrink when the network went quiet",
            stuck.len()
        );
    }

    // Tally, and check that every waiter went on to execute.
    let ours: std::collections::HashSet<String> =
        settled.iter().map(|(id, ..)| id.clone()).collect();
    let log = events.lock().unwrap().clone();
    let mut stats = RunStats {
        settle_secs: (stable_since - armed_at).as_secs_f64(),
        final_copies: coverage(&settled).iter().sum(),
        settled: did_settle,
        ..Default::default()
    };
    let mut waited_then: HashMap<String, bool> = HashMap::new();
    for (i, (agent, msg)) in log.iter().enumerate() {
        let grow_msg = msg == GREW || msg == GREW_SHORT || msg == GREW_INTENTS;
        if !ours.contains(agent) || (grow_msg && i < armed_log_len) {
            continue;
        }
        match msg.as_str() {
            ANNOUNCED => stats.announced += 1,
            WAITED => {
                stats.waited += 1;
                waited_then.entry(agent.clone()).or_insert(false);
            }
            EXECUTED => {
                stats.executed += 1;
                if let Some(done) = waited_then.get_mut(agent) {
                    *done = true;
                }
            }
            STOOD_DOWN => stats.stood_down += 1,
            GREW => stats.grew += 1,
            GREW_SHORT => stats.grew_short += 1,
            GREW_INTENTS => stats.grew_intents += 1,
            _ => {}
        }
    }
    assert!(stats.executed >= 1, "no shrink executed");
    // A waiting agent must go on to execute the same intent: its next
    // announce / execute / stand-down event after a wait is the execute.
    // (A wait that silently dropped the intent would show up here as a
    // fresh announcement instead.)
    for agent in waited_then.keys() {
        let mut after_wait = false;
        for (a, m) in &log {
            if a != agent {
                continue;
            }
            if m == WAITED {
                after_wait = true;
            } else if after_wait
                && (m == ANNOUNCED || m == EXECUTED || m == STOOD_DOWN)
            {
                assert_eq!(
                    EXECUTED, m,
                    "agent {agent} waited, then did not execute that intent"
                );
                after_wait = false;
            }
        }
    }
    // With withdrawal on, a stand-down by an agent on one node must be
    // applied by the other node.
    if withdraw && stats.stood_down > 0 {
        assert!(
            log.iter()
                .any(|(a, m)| ours.contains(a) && m == WITHDRAWAL_APPLIED),
            "agents stood down but no node applied a withdrawal"
        );
    }
    // Remote intents must pass the sender check: if a peer's URL as seen
    // by the transport ever differed from its declared URL, every remote
    // intent would be silently dropped.
    assert!(
        !log.iter().any(|(_, m)| m == SENDER_MISMATCH),
        "a remote intent arrived from a URL other than its agent's"
    );
    for (agent, done) in &waited_then {
        assert!(done, "agent {agent} waited at re-check but never executed");
    }
    if !patient_lowest {
        assert_eq!(0, stats.waited, "V3 never waits");
    }
    stats
}

#[tokio::test(flavor = "multi_thread")]
async fn forced_race_patient_lowest_waits_and_stays_safe() {
    let _serial = SERIAL.lock().await;
    let mut all = Vec::new();
    for run in 1..=MAX_V6_RUNS {
        let stats = race_run(true, true, true).await;
        println!("V6 forced race, run {run}: {stats:?}");
        let waited = stats.waited > 0;
        all.push(stats);
        if waited {
            return;
        }
    }
    panic!(
        "no V6 agent waited in {MAX_V6_RUNS} forced-race runs; the race \
         was never thin enough to exercise the wait: {all:?}"
    );
}

/// The same forced race under the original tie-break (V3), for
/// comparison: with views this fresh it is also safe, and resolves by
/// letting the lowest intender go at once rather than waiting.
#[tokio::test(flavor = "multi_thread")]
async fn forced_race_tiebreak_v3_stays_safe() {
    let _serial = SERIAL.lock().await;
    let stats = race_run(false, false, true).await;
    println!("V3 forced race: {stats:?}");
}

/// V3 against V6 over repeated forced-race runs, for the record: how
/// much announcing, standing down and waiting each rule does, how long
/// it takes to settle, and how far it shards down. Every run asserts
/// safety and progress. Slow; run with `--ignored --nocapture`.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "slow comparison; run with --ignored --nocapture"]
async fn forced_race_compare_v3_v6() {
    let _serial = SERIAL.lock().await;
    let runs: usize = std::env::var("RACE_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    for run in 1..=runs {
        for (name, patient, withdraw) in [
            ("V3", false, false),
            ("V3+withdraw", false, true),
            ("V6", true, false),
            ("V6+withdraw", true, true),
        ] {
            let s = race_run(patient, withdraw, false).await;
            println!(
                "compare,{name},{run},{},{},{},{},{},{},{},{:.1},{},{}",
                s.announced,
                s.stood_down,
                s.waited,
                s.executed,
                s.grew,
                s.grew_short,
                s.grew_intents,
                s.settle_secs,
                s.final_copies,
                s.settled
            );
        }
    }
}
