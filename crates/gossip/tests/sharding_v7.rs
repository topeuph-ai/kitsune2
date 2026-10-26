//! Functional tests for V7: coordinated growth and pinned nodes, over real
//! gossip on the in-memory transport.
//!
//! - **The herd.** Twelve nodes, one agent each, shard down from full arcs
//!   and settle; then three nodes die at once (chosen so no sector loses
//!   its last holder). Without coordination every agent that sees a short
//!   sector grows; with `coordinated_growth`, agents announce the range,
//!   wait, and grow only if it is still short counting lower-id announced
//!   growers. Asserted both ways: no sector is left without a holder after
//!   the death-detection window, and every sector gets back to the target.
//!   With coordination, fewer agents grow to repair the same deaths.
//! - **Pinned.** Eight nodes, one pinned: it keeps the full arc throughout
//!   while the others shard down; once every sector reaches the target none
//!   drops below it, and no unpinned agent is left able to shrink.
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

const SETTLE: Duration = Duration::from_secs(15);
/// After a kill: time to mark the dead unresponsive, as in the storm test.
const GRACE: Duration = Duration::from_secs(2);

fn config(target: u32, coordinated: bool, pinned: bool) -> K2ShardingConfig {
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
        require_recent_contact: recent_contact(),
        announce_alongside: true,
        coordinated_growth: coordinated,
        critical_repair: true,
        pinned,
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

type Node = (K2GossipFunctionalTestHarness, Arc<AgentInfoSigned>);

/// Start `configs.len()` nodes, one full-arc agent each, all introduced.
async fn network(configs: Vec<K2ShardingConfig>) -> Vec<Node> {
    let mut f = factory().await;
    let mut nodes: Vec<Node> = Vec::new();
    for c in configs {
        f.set_sharding_config(c);
        let h = f.new_instance().await;
        seed_ops(&h).await;
        let info = h.join_local_agent(DhtArc::FULL).await;
        nodes.push((h, info));
    }
    let infos: Vec<_> = nodes.iter().map(|(_, i)| i.clone()).collect();
    for (h, _) in &nodes {
        h.space.peer_store().insert(infos.clone()).await.unwrap();
    }
    nodes
}

async fn view(nodes: &[Node]) -> Vec<AgentView> {
    let hs: Vec<&K2GossipFunctionalTestHarness> =
        nodes.iter().map(|(h, _)| h).collect();
    agents(&hs).await
}

/// Wait until declared coverage has not changed for SETTLE, with every
/// sector at or above `target` throughout once it first gets there.
async fn settle(nodes: &[Node], target: u32, what: &str) -> Vec<AgentView> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(300);
    let mut last = Vec::new();
    let mut since = tokio::time::Instant::now();
    let mut armed = false;
    loop {
        let now = view(nodes).await;
        let cov = coverage(&now);
        let min = *cov.iter().min().unwrap();
        if armed {
            assert!(
                min >= target,
                "{what}: a sector fell to {min} (target {target})"
            );
        } else if min >= target {
            armed = true;
        }
        if cov != last {
            last = cov;
            since = tokio::time::Instant::now();
        } else if armed && since.elapsed() >= SETTLE {
            return now;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "{what}: never settled"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[derive(Debug)]
#[allow(dead_code)] // read through Debug, in the printout
struct Herd {
    grows_after_kill: usize,
    grow_announced: usize,
    grow_stood_down: usize,
    critical: usize,
    holes_in_grace: usize,
    secs_to_target: f64,
}

async fn herd(coordinated: bool) -> Herd {
    const TARGET: u32 = 3;
    const NODES: usize = 12;
    const KILL: usize = 3;
    let events = events();
    events.lock().unwrap().clear();
    let mut nodes = network(
        (0..NODES)
            .map(|_| config(TARGET, coordinated, false))
            .collect(),
    )
    .await;
    let settled = settle(&nodes, TARGET, "shard down").await;

    // Victims that are never the last holder of a sector, together.
    let mut cov = coverage(&settled);
    let mut victims = Vec::new();
    for (i, (_, info)) in nodes.iter().enumerate() {
        if victims.len() == KILL {
            break;
        }
        let id = format!("{:?}", info.agent);
        let Some((_, _, Some((s, e)))) =
            settled.iter().find(|(a, ..)| *a == id)
        else {
            continue;
        };
        if cov[*s as usize..*e as usize].iter().all(|&c| c >= 2) {
            for c in &mut cov[*s as usize..*e as usize] {
                *c -= 1;
            }
            victims.push(i);
        }
    }
    assert_eq!(KILL, victims.len(), "no survivable set of victims");

    let mark = events.lock().unwrap().len();
    let killed_at = tokio::time::Instant::now();
    for i in victims.iter().rev() {
        drop(nodes.remove(*i));
    }

    // Recovery: no hole after the grace window; back to the target.
    let mut holes_in_grace = 0;
    let deadline = killed_at + Duration::from_secs(240);
    let secs_to_target = loop {
        let cov = coverage(&view(&nodes).await);
        let holes = cov.iter().filter(|&&c| c == 0).count();
        if killed_at.elapsed() < GRACE {
            holes_in_grace = holes_in_grace.max(holes);
        } else {
            assert_eq!(
                0, holes,
                "a sector has no holder after the grace window"
            );
        }
        if *cov.iter().min().unwrap() >= TARGET {
            break killed_at.elapsed().as_secs_f64();
        }
        assert!(tokio::time::Instant::now() < deadline, "never recovered");
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    // Let any announced grows resolve before counting.
    settle(&nodes, TARGET, "recover").await;
    let log = events.lock().unwrap();
    let after = &log[mark..];
    let count = |what: &str| after.iter().filter(|(_, m)| m == what).count();
    Herd {
        grows_after_kill: count(GREW),
        grow_announced: count(GROW_ANNOUNCED),
        grow_stood_down: count(GROW_STOOD_DOWN),
        critical: count(CRITICAL_REPAIR),
        holes_in_grace,
        secs_to_target,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn coordinated_growth_repairs_deaths_with_fewer_growers() {
    let _serial = SERIAL.lock().await;
    let off = herd(false).await;
    println!("coordinated growth off: {off:?}");
    let on = herd(true).await;
    println!("coordinated growth on:  {on:?}");
    assert!(on.grow_announced > 0, "coordinated growth never announced");
    assert!(
        on.grows_after_kill <= off.grows_after_kill,
        "coordination should not add growers: on {} vs off {}",
        on.grows_after_kill,
        off.grows_after_kill
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn pinned_node_keeps_everything_while_others_shard() {
    let _serial = SERIAL.lock().await;
    events().lock().unwrap().clear();
    const TARGET: u32 = 3;
    let nodes =
        network((0..8).map(|i| config(TARGET, true, i == 0)).collect()).await;
    let pinned_id = format!("{:?}", nodes[0].1.agent);
    // Watch the pinned agent while the network shards down.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(300);
    let mut last = Vec::new();
    let mut since = tokio::time::Instant::now();
    let mut armed = false;
    let settled = loop {
        let now = view(&nodes).await;
        if let Some((_, _, range)) = now.iter().find(|(a, ..)| *a == pinned_id)
        {
            if armed {
                assert_eq!(
                    Some((0, NUM_SECTORS)),
                    *range,
                    "the pinned agent shrank"
                );
            }
        }
        let cov = coverage(&now);
        let min = *cov.iter().min().unwrap();
        if armed {
            assert!(min >= TARGET, "a sector fell to {min}");
        } else if min >= TARGET {
            armed = true;
        }
        if cov != last {
            last = cov;
            since = tokio::time::Instant::now();
        } else if armed && since.elapsed() >= SETTLE {
            break now;
        }
        assert!(tokio::time::Instant::now() < deadline, "never settled");
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    let stuck: Vec<String> = could_shrink(&settled, TARGET)
        .into_iter()
        .filter(|id| *id != pinned_id)
        .collect();
    assert!(
        stuck.is_empty(),
        "unpinned agents still able to shrink: {stuck:?}"
    );
    let shrunk = settled
        .iter()
        .filter(|(a, _, r)| *a != pinned_id && *r != Some((0, NUM_SECTORS)))
        .count();
    assert!(shrunk > 0, "nobody else sharded");
    let log = events().lock().unwrap().clone();
    let ours: HashSet<String> = [pinned_id.clone()].into();
    assert!(
        !log.iter().any(|(a, m)| ours.contains(a) && m == ANNOUNCED),
        "the pinned agent announced a shrink"
    );
    println!(
        "pinned: {} of 7 others shrank; final copies {} of {}",
        shrunk,
        coverage(&settled).iter().sum::<u32>(),
        NUM_SECTORS * 8
    );
}
