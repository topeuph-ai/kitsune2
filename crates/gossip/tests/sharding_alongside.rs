//! Functional test of `announce_alongside`, over real gossip on the
//! in-memory transport.
//!
//! Without it, an agent never announces a shrink while a lower id is
//! announcing for an overlapping range, so agents that share a range
//! shrink one at a time. Starting from full arcs, every agent's first
//! shrink drops one of the same two halves of the ring, which makes that
//! queue as long as it can be. The simulator showed this slowing sharding
//! down several-fold at 200 agents (polite-shrink `reports/REPORT_v7.md`).
//!
//! Sixteen nodes with one agent each start at full arcs, target 3. One
//! agent per node matters: agents on the same node decide together from
//! one snapshot and never see each other's announcements in time to
//! queue, so the queue only forms across nodes. The same scenario runs
//! with `announce_alongside` off and on. Asserted for both, with no peer
//! deaths:
//! - once every sector reaches the target, none drops below it;
//! - the network settles, and no agent is left able to shrink safely.
//!
//! And with it on, more agents announce at the same time (the queue is
//! gone). The time to settle is printed for both.
#![cfg(feature = "sharding")]

mod common;

use common::*;
use kitsune2_api::{AgentInfoSigned, DhtArc};
use kitsune2_gossip::harness::K2GossipFunctionalTestFactory;
use kitsune2_gossip::{K2GossipConfig, K2ShardingConfig};
use kitsune2_test_utils::space::TEST_SPACE_ID;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

const NODES: usize = 16;
const TARGET: u32 = 3;
/// As in the race test: longer than the slowest path from a stand-down to
/// the next decision.
const SETTLE: Duration = Duration::from_secs(15);

#[derive(Debug)]
#[allow(dead_code)] // the counts are read through Debug, in the printout
struct Run {
    /// Most agents with a shrink announced and not yet resolved, at once.
    max_concurrent: usize,
    announced: usize,
    executed: usize,
    stood_down: usize,
    /// Seconds from the start until declared coverage last changed.
    settle_secs: f64,
    final_copies: u32,
}

async fn run(alongside: bool) -> Run {
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
        target_redundancy: TARGET,
        clamp_min_peers: 2,
        check_interval_ms: 100,
        grow_persistence: 1.0,
        shrink_persistence: 4.0,
        intent_wait: 2.5,
        intent_min_wait_ms: 1_000,
        lag_floor_ms: 250,
        lag_ceiling_ms: 2_000,
        agentinfo_encoding: false,
        patient_lowest: true,
        withdraw_on_stand_down: true,
        require_recent_contact: recent_contact(),
        announce_alongside: alongside,
    });

    let mut harnesses = Vec::new();
    let mut infos: Vec<Arc<AgentInfoSigned>> = Vec::new();
    for _ in 0..NODES {
        let node = factory.new_instance().await;
        seed_ops(&node).await;
        infos.push(node.join_local_agent(DhtArc::FULL).await);
        harnesses.push(node);
    }
    // Introduce everyone to everyone rather than waiting for bootstrap.
    for node in &harnesses {
        node.space.peer_store().insert(infos.clone()).await.unwrap();
    }
    let nodes: Vec<_> = harnesses.iter().collect();

    let start = tokio::time::Instant::now();
    let mut armed = false;
    let mut last = Vec::new();
    let mut stable_since = tokio::time::Instant::now();
    let mut outstanding: HashSet<String> = HashSet::new();
    let mut max_concurrent = 0;
    let mut seen = 0;
    let deadline = start + Duration::from_secs(300);
    let settled = loop {
        // Follow announcements as they arrive.
        {
            let log = events.lock().unwrap();
            for (agent, msg) in &log[seen..] {
                match msg.as_str() {
                    ANNOUNCED => {
                        outstanding.insert(agent.clone());
                    }
                    EXECUTED | STOOD_DOWN => {
                        outstanding.remove(agent);
                    }
                    _ => {}
                }
            }
            seen = log.len();
        }
        max_concurrent = max_concurrent.max(outstanding.len());

        let now = agents(&nodes).await;
        let cov = coverage(&now);
        let min = *cov.iter().min().unwrap();
        if armed {
            assert!(
                min >= TARGET,
                "a sector fell to {min} declared holders, below the target \
                 {TARGET}, with no peer deaths (alongside {alongside})"
            );
        } else if min >= TARGET {
            armed = true;
        }
        let executed =
            events.lock().unwrap().iter().any(|(_, m)| m == EXECUTED);
        if cov != last {
            last = cov;
            stable_since = tokio::time::Instant::now();
        } else if executed && stable_since.elapsed() >= SETTLE {
            break now;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "never settled (alongside {alongside})"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    let stuck = could_shrink(&settled, TARGET);
    assert!(
        stuck.is_empty(),
        "settled with agents still able to shrink (alongside {alongside}): \
         {stuck:?}"
    );
    let log = events.lock().unwrap().clone();
    let count = |what: &str| log.iter().filter(|(_, m)| m == what).count();
    Run {
        max_concurrent,
        announced: count(ANNOUNCED),
        executed: count(EXECUTED),
        stood_down: count(STOOD_DOWN),
        settle_secs: (stable_since - start).as_secs_f64(),
        final_copies: coverage(&settled).iter().sum(),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn alongside_lets_agents_sharing_a_range_announce_together() {
    let _serial = SERIAL.lock().await;
    let off = run(false).await;
    println!("announce_alongside off: {off:?}");
    let on = run(true).await;
    println!("announce_alongside on:  {on:?}");
    assert!(
        on.max_concurrent > off.max_concurrent,
        "alongside should let more agents announce at once: on {} vs off {}",
        on.max_concurrent,
        off.max_concurrent
    );
}
