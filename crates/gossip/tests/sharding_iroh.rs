//! The sharding storm over the real iroh transport, at several sizes.
//!
//! The other functional tests use the in-memory transport. This one runs the
//! same kind of storm over kitsune2's iroh transport (QUIC, with a local relay
//! from the test bootstrap server), so connection setup, real streams and
//! real unresponsive-peer marking are all in the loop. It is still one
//! machine, so latency is local; it tests the wiring and the controller over
//! a real network stack, not a wide-area network.
//!
//! Scenario, for N nodes with one agent each and patient lowest (V6) with
//! intent withdrawal: everyone starts at full arc; the network shards down
//! until declared copies are below 60% of full replication; it settles; a
//! quarter of the nodes are killed at once (chosen so the kill itself orphans
//! nothing); the rest recover. Asserted throughout: no sector is ever left
//! with no declared holder, except during the short death-detection window
//! right after the storm, which is reported, as in sharding_storm.rs.
#![cfg(feature = "sharding")]

mod common;

use common::*;
use kitsune2_api::DhtArc;
use kitsune2_gossip::harness::{
    K2GossipFunctionalTestFactory, K2GossipFunctionalTestHarness,
};
use kitsune2_gossip::{K2GossipConfig, K2ShardingConfig};
use kitsune2_test_utils::bootstrap::TestBootstrapSrv;
use kitsune2_test_utils::space::TEST_SPACE_ID;
use kitsune2_transport_iroh::{
    IrohTransportConfig, IrohTransportFactory, IrohTransportModConfig,
};
use std::sync::Arc;
use std::time::Duration;

const TARGET_REDUNDANCY: u32 = 2;

#[derive(Debug, Default)]
#[allow(dead_code)] // fields are read through Debug in the report line
struct StormStats {
    nodes: usize,
    killed: usize,
    copies_before: u32,
    full_copies: u32,
    transient_orphans: usize,
    below_target_after_kill: usize,
    min_after_recovery: u32,
    recovery_secs: f64,
}

async fn storm_over_iroh(num_nodes: usize) -> StormStats {
    let relay = TestBootstrapSrv::new(false).await;
    let relay_url = format!("{}/relay", relay.addr());

    let factory = K2GossipFunctionalTestFactory::create(
        TEST_SPACE_ID,
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
        target_redundancy: TARGET_REDUNDANCY,
        clamp_min_peers: 4,
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
    })
    .with_transport(
        IrohTransportFactory::create(),
        Arc::new(move |builder: &kitsune2_api::Builder| {
            builder
                .config
                .set_module_config(&IrohTransportModConfig {
                    iroh_transport: IrohTransportConfig {
                        relay_allow_plain_text: relay_url
                            .starts_with("http://"),
                        relay_url: Some(relay_url.clone()),
                        ..Default::default()
                    },
                })
                .unwrap();
        }),
    );

    let mut nodes: Vec<K2GossipFunctionalTestHarness> = Vec::new();
    let mut infos = Vec::new();
    for i in 0..num_nodes {
        let node = factory.new_instance().await;
        if i == 0 {
            seed_ops(&node).await;
        }
        infos.push(node.join_local_agent(DhtArc::FULL).await);
        nodes.push(node);
    }
    for node in &nodes {
        node.space.peer_store().insert(infos.clone()).await.unwrap();
    }

    let full_copies = num_nodes as u32 * NUM_SECTORS;
    let mut stats = StormStats {
        nodes: num_nodes,
        full_copies,
        ..Default::default()
    };

    // Reach full replication, then shard down below 60% with no orphan.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(300);
    let mut armed = false;
    loop {
        let all = nodes.iter().collect::<Vec<_>>();
        let cov = coverage(&agents(&all).await);
        let total: u32 = cov.iter().sum();
        if !armed {
            armed = cov.iter().all(|&c| c == num_nodes as u32);
        } else {
            assert!(
                cov.iter().all(|&c| c > 0),
                "a sector lost its last declared holder while sharding down"
            );
            if total * 10 <= full_copies * 6 {
                break;
            }
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "network never sharded down: {total}/{full_copies} copies"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    // Settle: unchanged for 6 s.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(180);
    let mut last = Vec::new();
    let mut since = tokio::time::Instant::now();
    loop {
        let all = nodes.iter().collect::<Vec<_>>();
        let cov = coverage(&agents(&all).await);
        assert!(cov.iter().all(|&c| c > 0), "orphan while settling");
        if cov != last {
            last = cov;
            since = tokio::time::Instant::now();
        } else if since.elapsed() >= Duration::from_secs(6) {
            break;
        }
        assert!(tokio::time::Instant::now() < deadline, "never settled");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    stats.copies_before = last.iter().sum();

    // Storm: kill a quarter, chosen greedily so the kill orphans nothing.
    let mut per_node = Vec::new();
    for node in &nodes {
        per_node.push(coverage(&agents(&[node]).await));
    }
    // Prefer victims that push sectors below the target (so recovery has to
    // grow), but never kill the last holder of any sector.
    let kill = num_nodes / 4;
    let mut cov_left = last.clone();
    let mut victims: Vec<usize> = Vec::new();
    while victims.len() < kill {
        let best = per_node
            .iter()
            .enumerate()
            .filter(|(i, own)| {
                !victims.contains(i)
                    && own.iter().zip(&cov_left).all(|(o, c)| c - o > 0)
            })
            .max_by_key(|(_, own)| {
                own.iter()
                    .zip(&cov_left)
                    .filter(|&(o, c)| *o > 0 && c - o < TARGET_REDUNDANCY)
                    .count()
            })
            .map(|(i, _)| i);
        let Some(i) = best else { break };
        for (c, o) in cov_left.iter_mut().zip(&per_node[i]) {
            *c -= o;
        }
        victims.push(i);
    }
    stats.below_target_after_kill =
        cov_left.iter().filter(|&&c| c < TARGET_REDUNDANCY).count();
    stats.killed = victims.len();
    let mut survivors = Vec::new();
    for (i, node) in nodes.into_iter().enumerate() {
        if victims.contains(&i) {
            drop(node);
        } else {
            survivors.push(node);
        }
    }

    // Death-detection window: observed, not asserted.
    tokio::time::sleep(Duration::from_secs(2)).await;
    {
        let all = survivors.iter().collect::<Vec<_>>();
        let cov = coverage(&agents(&all).await);
        stats.transient_orphans = cov.iter().filter(|&&c| c == 0).count();
    }

    // Recovery: once a sector has a holder again it must keep one; wait for
    // every sector to reach the target, or report how far it got.
    let start = tokio::time::Instant::now();
    let deadline = start + Duration::from_secs(240);
    let mut seen = vec![false; NUM_SECTORS as usize];
    loop {
        let all = survivors.iter().collect::<Vec<_>>();
        let cov = coverage(&agents(&all).await);
        for (s, &c) in cov.iter().enumerate() {
            if seen[s] {
                assert!(c > 0, "sector {s} re-covered and then orphaned again");
            } else if c > 0 {
                seen[s] = true;
            }
        }
        stats.min_after_recovery = *cov.iter().min().unwrap();
        if cov.iter().all(|&c| c >= TARGET_REDUNDANCY) {
            break;
        }
        if tokio::time::Instant::now() >= deadline {
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    stats.recovery_secs = start.elapsed().as_secs_f64();
    drop(relay);
    stats
}

#[tokio::test(flavor = "multi_thread")]
async fn storm_over_iroh_8_nodes() {
    let s = storm_over_iroh(8).await;
    println!("iroh storm: {s:?}");
    assert!(s.min_after_recovery >= 1, "data left uncovered: {s:?}");
}

/// Larger networks over iroh on one machine. Slow; run with
/// `--ignored --nocapture`, `IROH_SIZES` (default "16,24").
#[tokio::test(flavor = "multi_thread")]
#[ignore = "slow: larger iroh networks; run with --ignored --nocapture"]
async fn storm_over_iroh_larger() {
    let sizes: Vec<usize> = std::env::var("IROH_SIZES")
        .unwrap_or_else(|_| "16,24".into())
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();
    for n in sizes {
        let s = storm_over_iroh(n).await;
        println!("iroh storm: {s:?}");
        assert!(s.min_after_recovery >= 1, "data left uncovered: {s:?}");
    }
}
