//! Functional tests for networks that are not uniform: what a real rollout
//! of dynamic sharding looks like.
//!
//! No live network runs an earlier version of this controller, so the
//! realistic starting point is nodes without dynamic sharding: they keep
//! a fixed (here full) arc and never shrink. Such a node does not know the
//! sharding module at all; kitsune2's transport silently drops module
//! messages for a module that is not registered (the `K2WireType::Module`
//! branch of `TxImpHnd`), so V6 intents cost it nothing. Here it is
//! modelled as a node whose clamp is never reached (`clamp_min_peers`
//! above any network size): full arc, never shrinks, never announces.
//!
//! - **Fixed-arc nodes alongside V6.** Two never-shrinking nodes and six
//!   V6 nodes; no deaths. Once every sector reaches the target, none
//!   ever drops below it; the fixed nodes stay full; V6 nodes shrink and
//!   the network settles with no V6 agent left able to shrink.
//! - **A rollout.** Six fixed-arc nodes are replaced one at a time: a new
//!   V6 node joins, runs alongside the rest for a while, then an old node
//!   goes away. Each step is a death the V6 nodes must absorb while they
//!   are shrinking. No sector is ever left without a declared holder
//!   (outside the death-detection window, as in the storm test), and at
//!   the end, with only V6 nodes, every sector is back at the target.
//! - **Mixed V6 settings.** Eight V6 nodes, half with one set of timing
//!   settings and half with another (wait, check interval, persistence,
//!   lag ceiling), as a network upgraded by different operators would
//!   be. Same assertions as the first test.
#![cfg(feature = "sharding")]

mod common;

use common::*;
use kitsune2_api::{AgentInfoSigned, DhtArc};
use kitsune2_gossip::harness::{
    K2GossipFunctionalTestFactory, K2GossipFunctionalTestHarness,
};
use kitsune2_gossip::{K2GossipConfig, K2ShardingConfig};
use kitsune2_test_utils::space::TEST_SPACE_ID;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

/// Coverage unchanged this long counts as settled (as in the race test)
/// for the default settings here: longer than the slowest path from a
/// stand-down to the next decision (shrink persistence 4 x the 2 s lag
/// ceiling, then an intent wait of 2.5 x it).
const SETTLE: Duration = Duration::from_secs(15);

/// The same for a given config, with 2 s to spare.
fn settle_for(c: &K2ShardingConfig) -> Duration {
    let lag = c.lag_ceiling_ms as f64 / 1000.0;
    let wait = (c.intent_wait * lag).max(c.intent_min_wait_ms as f64 / 1000.0);
    Duration::from_secs_f64(c.shrink_persistence * lag + wait + 2.0)
}

/// A running node and the agent it hosts.
type Node = (K2GossipFunctionalTestHarness, Arc<AgentInfoSigned>);

fn v6(target: u32) -> K2ShardingConfig {
    K2ShardingConfig {
        target_redundancy: target,
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
    }
}

/// A node without dynamic sharding: the clamp is never reached, so it
/// keeps its full arc and never shrinks or announces.
fn fixed(target: u32) -> K2ShardingConfig {
    K2ShardingConfig {
        clamp_min_peers: u32::MAX,
        ..v6(target)
    }
}

async fn factory() -> K2GossipFunctionalTestFactory {
    K2GossipFunctionalTestFactory::create(
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
}

/// Start a node with one full-arc agent and introduce it to the running
/// nodes, and them to it (rather than waiting for bootstrap).
async fn add_node(
    factory: &mut K2GossipFunctionalTestFactory,
    config: K2ShardingConfig,
    others: &[&Node],
) -> Node {
    factory.set_sharding_config(config);
    let node = factory.new_instance().await;
    seed_ops(&node).await;
    let info = node.join_local_agent(DhtArc::FULL).await;
    for (other, _) in others {
        other
            .space
            .peer_store()
            .insert(vec![info.clone()])
            .await
            .unwrap();
    }
    let known: Vec<Arc<AgentInfoSigned>> =
        others.iter().map(|(_, i)| i.clone()).collect();
    if !known.is_empty() {
        node.space.peer_store().insert(known).await.unwrap();
    }
    (node, info)
}

async fn view(nodes: &[&Node]) -> Vec<AgentView> {
    let hs: Vec<&K2GossipFunctionalTestHarness> =
        nodes.iter().map(|(h, _)| h).collect();
    agents(&hs).await
}

/// Wait until every agent has grown to its full starting arc: an
/// established, over-provisioned network, where shrinking becomes correct.
async fn wait_full(nodes: &[&Node]) {
    for (h, _) in nodes {
        h.wait_for_full_arc_for_all(Duration::from_secs(120)).await;
    }
}

fn id_of(node: &Node) -> String {
    format!("{:?}", node.1.agent)
}

/// Run a network with no deaths until it settles, asserting that once
/// every sector reaches `target` none drops below it, that agents in
/// `fixed_ids` keep a full arc and never announce, and that at the end no
/// other agent could still shrink safely.
async fn settle_no_deaths(
    nodes: &[&Node],
    target: u32,
    fixed_ids: &HashSet<String>,
    settle: Duration,
) {
    let events = events();
    let mut armed = false;
    let mut last = Vec::new();
    let mut stable_since = tokio::time::Instant::now();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(300);
    let settled = loop {
        let now = view(nodes).await;
        for (id, _, range) in &now {
            if fixed_ids.contains(id) {
                assert_eq!(
                    Some((0, NUM_SECTORS)),
                    *range,
                    "a fixed-arc agent's arc changed"
                );
            }
        }
        let cov = coverage(&now);
        let min = *cov.iter().min().unwrap();
        if armed {
            assert!(
                min >= target,
                "a sector fell to {min} declared holders, below the target \
                 {target}, with no peer deaths"
            );
        } else if min >= target {
            armed = true;
        }
        let executed =
            events.lock().unwrap().iter().any(|(_, m)| m == EXECUTED);
        if cov != last {
            last = cov;
            stable_since = tokio::time::Instant::now();
        } else if executed && stable_since.elapsed() >= settle {
            break now;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "never settled (executed any: {executed})"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    let stuck: Vec<String> = could_shrink(&settled, target)
        .into_iter()
        .filter(|id| !fixed_ids.contains(id))
        .collect();
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
    assert!(
        stuck.is_empty(),
        "V6 agents still able to shrink: {stuck:?}"
    );
    let log = events.lock().unwrap().clone();
    assert!(
        !log.iter()
            .any(|(a, m)| fixed_ids.contains(a) && m == ANNOUNCED),
        "a fixed-arc agent announced a shrink"
    );
    let count = |what: &str| log.iter().filter(|(_, m)| m == what).count();
    println!(
        "settled: {} announced, {} waited, {} executed, {} stood down; \
         final copies {} of {}",
        count(ANNOUNCED),
        count(WAITED),
        count(EXECUTED),
        count(STOOD_DOWN),
        coverage(&settled).iter().sum::<u32>(),
        NUM_SECTORS * nodes.len() as u32
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn fixed_arc_nodes_alongside_v6() {
    let _serial = SERIAL.lock().await;
    events().lock().unwrap().clear();
    const TARGET: u32 = 4;
    let mut f = factory().await;
    let mut nodes: Vec<Node> = Vec::new();
    for i in 0..8 {
        let cfg = if i < 2 { fixed(TARGET) } else { v6(TARGET) };
        let refs: Vec<&Node> = nodes.iter().collect();
        let node = add_node(&mut f, cfg, &refs).await;
        nodes.push(node);
    }
    let fixed_ids: HashSet<String> = nodes[..2].iter().map(id_of).collect();
    let refs: Vec<&Node> = nodes.iter().collect();
    wait_full(&refs).await;
    settle_no_deaths(&refs, TARGET, &fixed_ids, SETTLE).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn mixed_v6_settings() {
    let _serial = SERIAL.lock().await;
    events().lock().unwrap().clear();
    const TARGET: u32 = 4;
    let other = K2ShardingConfig {
        check_interval_ms: 250,
        shrink_persistence: 2.0,
        intent_wait: 4.0,
        intent_min_wait_ms: 1_500,
        lag_ceiling_ms: 3_000,
        ..v6(TARGET)
    };
    let mut f = factory().await;
    let mut nodes: Vec<Node> = Vec::new();
    for i in 0..8 {
        let cfg = if i % 2 == 0 {
            v6(TARGET)
        } else {
            other.clone()
        };
        let refs: Vec<&Node> = nodes.iter().collect();
        let node = add_node(&mut f, cfg, &refs).await;
        nodes.push(node);
    }
    let refs: Vec<&Node> = nodes.iter().collect();
    wait_full(&refs).await;
    let settle = settle_for(&other).max(SETTLE);
    settle_no_deaths(&refs, TARGET, &HashSet::new(), settle).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn rollout_replaces_fixed_arc_nodes() {
    let _serial = SERIAL.lock().await;
    events().lock().unwrap().clear();
    const TARGET: u32 = 3;
    const OLD: usize = 6;
    let mut f = factory().await;
    let mut old: Vec<Node> = Vec::new();
    for _ in 0..OLD {
        let refs: Vec<&Node> = old.iter().collect();
        let node = add_node(&mut f, fixed(TARGET), &refs).await;
        old.push(node);
    }
    wait_full(&old.iter().collect::<Vec<_>>()).await;
    let mut new: Vec<Node> = Vec::new();

    let no_holes = |cov: &[u32], when: &str| {
        let holes = cov.iter().filter(|&&c| c == 0).count();
        assert_eq!(0, holes, "{holes} sectors with no holder {when}");
    };

    for step in 1..=OLD {
        // A new V6 node joins ...
        let refs: Vec<&Node> = old.iter().chain(new.iter()).collect();
        let node = add_node(&mut f, v6(TARGET), &refs).await;
        new.push(node);
        // ... and runs alongside the rest for a while, shrinking.
        let until = tokio::time::Instant::now() + Duration::from_secs(10);
        while tokio::time::Instant::now() < until {
            let refs: Vec<&Node> = old.iter().chain(new.iter()).collect();
            no_holes(&coverage(&view(&refs).await), "before a removal");
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        // Then an old node goes away (dropping the harness aborts its
        // tasks and closes its transport).
        drop(old.pop());
        // As in the storm test, a shrink announced just before the death
        // can execute before the death is detected; a hole in that first
        // second is reported, not failed. After it, none is allowed.
        tokio::time::sleep(Duration::from_secs(1)).await;
        let refs: Vec<&Node> = old.iter().chain(new.iter()).collect();
        let cov = coverage(&view(&refs).await);
        let holes = cov.iter().filter(|&&c| c == 0).count();
        if holes > 0 {
            println!(
                "note: step {step}: {holes} sectors uncovered inside the \
                 death-detection window"
            );
        }
        println!(
            "step {step}: {} old, {} V6, min coverage {}",
            old.len(),
            new.len(),
            cov.iter().min().unwrap()
        );
    }

    // Only V6 nodes remain: every sector must get back to the target and
    // stay there, with no hole on the way.
    let refs: Vec<&Node> = new.iter().collect();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    let mut reached = false;
    let mut since = tokio::time::Instant::now();
    loop {
        let cov = coverage(&view(&refs).await);
        no_holes(&cov, "after the rollout");
        let min = *cov.iter().min().unwrap();
        if reached {
            assert!(min >= TARGET, "fell back below target to {min}");
            if since.elapsed() >= SETTLE {
                break;
            }
        } else if min >= TARGET {
            reached = true;
            since = tokio::time::Instant::now();
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "never got back to target after the rollout (min {min})"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let log = events().lock().unwrap().clone();
    let count = |what: &str| log.iter().filter(|(_, m)| m == what).count();
    println!(
        "rollout done: {} shrinks, {} grows; final copies {} of {}",
        count(EXECUTED),
        count(GREW),
        coverage(&view(&refs).await).iter().sum::<u32>(),
        NUM_SECTORS * refs.len() as u32
    );
}
