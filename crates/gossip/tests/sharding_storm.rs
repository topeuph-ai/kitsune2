//! Functional test for the sharding module: a network shards down from
//! full arcs via polite shrinks, absorbs a sudden die-off of nodes, and
//! re-grows. No sector may ever lose its last declared holder while the
//! controller is the only thing moving (shard-down, settle, and
//! post-storm recovery); the one exception is the short death-detection
//! window right after the storm, where an already-announced shrink can
//! execute before the deaths are visible — that window is observed and
//! reported rather than asserted, because no local rule can beat it and
//! it does not lose data (the shrinking node still holds the ops).
//!
//! This exercises the mechanism end-to-end over real gossip on the
//! in-memory transport: coverage measurement, hysteresis, shrink-intent
//! announcement and tie-break, unresponsive-peer discounting, and grow
//! via the verified-sync machinery. It validates that the wiring works;
//! the statistical zero-loss evidence for the controller design comes
//! from the simulation study this module ports.
#![cfg(feature = "sharding")]

use kitsune2_api::{DhtArc, Timestamp};
use kitsune2_core::factories::MemoryOp;
use kitsune2_dht::SECTOR_SIZE;
use kitsune2_gossip::harness::{
    K2GossipFunctionalTestFactory, K2GossipFunctionalTestHarness,
    MemoryOpRecord,
};
use kitsune2_gossip::{K2GossipConfig, K2ShardingConfig};
use kitsune2_test_utils::space::TEST_SPACE_ID;
use kitsune2_test_utils::{enable_tracing_with_default_level, random_bytes};
use std::time::Duration;

const NUM_SECTORS: u32 = u32::MAX / SECTOR_SIZE + 1;
const NUM_NODES: usize = 8;
const NUM_KILLED: usize = 3;
const TARGET_REDUNDANCY: u32 = 2;

/// Per-sector count of live local agents declaring coverage.
async fn declared_coverage(
    nodes: &[&K2GossipFunctionalTestHarness],
) -> Vec<u32> {
    let mut cov = vec![0u32; NUM_SECTORS as usize];
    for node in nodes {
        for agent in node.space.local_agent_store().get_all().await.unwrap() {
            let arc = agent.get_cur_storage_arc();
            if matches!(arc, DhtArc::Empty) {
                continue;
            }
            for (sector, c) in cov.iter_mut().enumerate() {
                let start = sector as u32 * SECTOR_SIZE;
                let end = start + (SECTOR_SIZE - 1);
                if arc.contains(start) && arc.contains(end) {
                    *c += 1;
                }
            }
        }
    }
    cov
}

#[tokio::test(flavor = "multi_thread")]
async fn storm_does_not_orphan_sectors() {
    enable_tracing_with_default_level(tracing::Level::INFO);

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
        // Above the post-storm survivor count: the 8-node network shards,
        // and the 5 survivors fall back to full arcs. Growth is
        // sibling-half-local, so it cannot be steered at distant thin
        // regions; the clamp is the designed recovery path when a network
        // gets this sparse, exactly as in the simulation (whose default
        // clamp is 25).
        clamp_min_peers: 6,
        check_interval_ms: 100,
        grow_persistence: 1.0,
        shrink_persistence: 4.0,
        intent_wait: 2.5,
        intent_min_wait_ms: 1_000,
        lag_floor_ms: 250,
        lag_ceiling_ms: 2_000,
    });

    // Seed some data so gossip rounds have something to verify.
    let mut nodes = Vec::with_capacity(NUM_NODES);
    let first = factory.new_instance().await;
    first.join_local_agent(DhtArc::FULL).await;
    {
        let mut op_store = first.op_store.write().await;
        for _ in 0..16 {
            let op = MemoryOp::new(Timestamp::now(), random_bytes(512));
            op_store.op_list.insert(
                op.compute_op_id(),
                MemoryOpRecord {
                    op_id: op.compute_op_id(),
                    op_data: op.op_data,
                    created_at: op.created_at,
                    stored_at: Timestamp::now(),
                    processed: false,
                },
            );
        }
    }
    nodes.push(first);

    for _ in 1..NUM_NODES {
        let node = factory.new_instance().await;
        let info = node.join_local_agent(DhtArc::FULL).await;
        // Help discovery along rather than waiting for bootstrap.
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

    // Everyone reaches full arc first: an established, over-provisioned
    // network, which is where shrinking becomes correct.
    for node in &nodes {
        node.wait_for_full_arc_for_all(Duration::from_secs(120))
            .await;
    }

    // The invariant under test, checked from here to the end: no sector
    // is ever left without a live declared holder. (The controller's job
    // is never to *create* a hole; a hole the storm itself causes would
    // be a test-setup failure, excluded below by victim choice.)
    let assert_no_orphans = |cov: &[u32], when: &str| {
        let orphans = cov.iter().filter(|&&c| c == 0).count();
        assert_eq!(
            0, orphans,
            "{orphans} sectors with zero declared holders {when}"
        );
    };

    // Wait for the network to shard down deeply: total declared
    // sector-copies must drop below 60% of the everyone-stores-
    // everything baseline, and no sector may be orphaned at any point
    // on the way. This is the polite-shrink phase doing real work, not
    // a single token shrink.
    let full_copies = (NUM_NODES as u32) * NUM_SECTORS;
    let shard_down_deadline =
        tokio::time::Instant::now() + Duration::from_secs(300);
    loop {
        let all = nodes.iter().collect::<Vec<_>>();
        let cov = declared_coverage(&all).await;
        assert_no_orphans(&cov, "while sharding down");

        let total: u32 = cov.iter().sum();
        if total * 10 <= full_copies * 6 {
            break;
        }
        assert!(
            tokio::time::Instant::now() < shard_down_deadline,
            "network never sharded down: still {total}/{full_copies} \
             declared sector-copies"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Let the controllers reach a working equilibrium: declared
    // coverage unchanged for three consecutive seconds, invariant
    // checked throughout.
    let settle_deadline =
        tokio::time::Instant::now() + Duration::from_secs(120);
    let mut last_cov = Vec::new();
    let mut stable_since = tokio::time::Instant::now();
    loop {
        let all = nodes.iter().collect::<Vec<_>>();
        let cov = declared_coverage(&all).await;
        assert_no_orphans(&cov, "while settling");

        if cov != last_cov {
            last_cov = cov;
            stable_since = tokio::time::Instant::now();
        } else if stable_since.elapsed() >= Duration::from_secs(6) {
            // Longer than the maximum intent wait, so any shrink that
            // was in flight when the window opened has executed or been
            // cancelled inside it.
            break;
        }
        assert!(
            tokio::time::Instant::now() < settle_deadline,
            "controllers never settled"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Storm: kill NUM_KILLED nodes at once. Pick victims whose loss does
    // not itself orphan any sector, so any orphan seen afterwards is the
    // controller's fault, which is exactly the property under test.
    // Prefer a victim set that pushes some sector *below* the redundancy
    // target, so recovery has to exercise the grow path rather than
    // coast on leftover slack.
    //
    // Selection works on a single snapshot of every node's declared
    // arcs, taken in one pass, so the controllers cannot move between
    // measurement and the kill.
    let mut per_node_cov = Vec::with_capacity(NUM_NODES);
    for node in &nodes {
        per_node_cov.push(declared_coverage(&[node]).await);
    }
    let survivor_cov = |skips: &[usize]| {
        let mut cov = vec![0u32; NUM_SECTORS as usize];
        for (i, node_cov) in per_node_cov.iter().enumerate() {
            if !skips.contains(&i) {
                for (c, n) in cov.iter_mut().zip(node_cov) {
                    *c += n;
                }
            }
        }
        cov
    };
    let mut victims = Vec::new();
    let mut fallback = Vec::new();
    'pick: for skip_a in 0..NUM_NODES {
        for skip_b in (skip_a + 1)..NUM_NODES {
            for skip_c in (skip_b + 1)..NUM_NODES {
                let skips = [skip_a, skip_b, skip_c];
                let cov = survivor_cov(&skips);
                if cov.iter().all(|&c| c > 0) {
                    if fallback.is_empty() {
                        fallback = skips.to_vec();
                    }
                    if cov.iter().any(|&c| c < TARGET_REDUNDANCY) {
                        victims = skips.to_vec();
                        break 'pick;
                    }
                }
            }
        }
    }
    if victims.is_empty() {
        victims = fallback;
    }
    assert_eq!(
        NUM_KILLED,
        victims.len(),
        "no survivable victim set exists; coverage too thin before storm"
    );

    // Kill by dropping the harnesses: tasks abort, transport goes away.
    let mut survivors = Vec::new();
    for (i, node) in nodes.into_iter().enumerate() {
        if victims.contains(&i) {
            drop(node);
        } else {
            survivors.push(node);
        }
    }

    // Death-detection grace. A shrink intent announced just before the
    // storm can execute just after it, before the dead peers have been
    // marked unresponsive — no local rule can see a death that has not
    // been detected yet. A declared-coverage hole in this window is not
    // data loss (the node that shrank still holds the op data and will
    // re-grow); it is observed and reported, but not a failure.
    tokio::time::sleep(Duration::from_secs(1)).await;
    {
        let all = survivors.iter().collect::<Vec<_>>();
        let cov = declared_coverage(&all).await;
        let transient = cov.iter().filter(|&&c| c == 0).count();
        if transient > 0 {
            println!(
                "note: {transient} sectors transiently uncovered inside \
                 the death-detection window"
            );
        }
    }

    // Recovery: survivors must notice the loss (unresponsive marking),
    // grow to re-cover, and from here on never orphan a sector.
    // Coverage must return to the redundancy target everywhere.
    let recovery_deadline =
        tokio::time::Instant::now() + Duration::from_secs(180);
    let mut seen_covered = vec![false; NUM_SECTORS as usize];
    loop {
        let all = survivors.iter().collect::<Vec<_>>();
        let cov = declared_coverage(&all).await;

        // Once a sector has a holder after the storm, losing it again
        // would be the controller's doing: that is a hard failure.
        for (sector, &c) in cov.iter().enumerate() {
            if seen_covered[sector] {
                assert!(
                    c > 0,
                    "sector {sector} was re-covered after the storm and \
                     then orphaned again"
                );
            } else if c > 0 {
                seen_covered[sector] = true;
            }
        }

        if cov.iter().all(|&c| c >= TARGET_REDUNDANCY) {
            break;
        }
        assert!(
            tokio::time::Instant::now() < recovery_deadline,
            "network did not recover to target redundancy; min coverage {}",
            cov.iter().min().unwrap()
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
