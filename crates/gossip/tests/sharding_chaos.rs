//! A randomised soak ("chaos") run of V6 on the in-memory transport.
//!
//! For `CHAOS_SECS` (default 3600), every 2-6 s one of these happens at
//! random (seeded by `CHAOS_SEED`):
//! - a random node is killed, if the network has more than `MIN_NODES`
//!   and the node is not the last declared holder of any sector (no rule
//!   survives losing the last copy, so that kill would test nothing);
//! - a new node joins with a full arc, if there are fewer than
//!   `MAX_NODES`;
//! - the staleness of every node's view changes, to 0-1.5 s
//!   ([`set_sharding_view_delay`]); the wait is a fixed 2 s, so
//!   assumption A1 (wait >= staleness) holds throughout;
//! - nothing.
//!
//! Checked every 50 ms against the nodes' true declared arcs:
//! - **no shrink takes a sector below target**: a sector whose count
//!   falls, to below the target, with no node killed in the last
//!   `GRACE`, is a failure;
//! - **no sector is ever left without a holder** outside `GRACE` after a
//!   kill.
//!
//! Inside `GRACE` after a kill such drops are counted and reported, not
//! failed: a node can only discount a death once it has noticed it, and
//! here a dead node stays in other nodes' views for up to the staleness
//! plus the time to mark it unresponsive (assumption A3).
//!
//! `CHAOS_WAIT_MS` overrides the wait. Set below the largest staleness it
//! is a control: A1 no longer holds and the run is expected to fail.
//!
//! At the end the churn stops, staleness returns to zero, and every sector
//! must get back to the target and stay there.
//!
//! Slow; run with
//! `CHAOS_SECS=3600 cargo test -p kitsune2_gossip --features sharding
//! --test sharding_chaos -- --ignored --nocapture`.
#![cfg(feature = "sharding")]

mod common;

use common::*;
use kitsune2_api::{AgentInfoSigned, DhtArc};
use kitsune2_gossip::harness::{
    K2GossipFunctionalTestFactory, K2GossipFunctionalTestHarness,
    set_sharding_view_delay,
};
use kitsune2_gossip::{K2GossipConfig, K2ShardingConfig};
use kitsune2_test_utils::space::TEST_SPACE_ID;
use std::sync::Arc;
use std::time::Duration;

const TARGET: u32 = 3;
const START_NODES: usize = 10;
const MIN_NODES: usize = 7;
const MAX_NODES: usize = 14;
const WAIT_MS: u32 = 2_000;
const STALENESS_MS: [u64; 5] = [0, 250, 500, 1_000, 1_500];
/// After a kill: the largest staleness, plus time to notice the death,
/// with room to spare.
const GRACE: Duration = Duration::from_secs(5);

type Node = (K2GossipFunctionalTestHarness, Arc<AgentInfoSigned>);

/// A small deterministic generator (xorshift64*), so a failing seed can
/// be replayed without depending on a particular `rand` version.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn env(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn config(wait_ms: u32) -> K2ShardingConfig {
    K2ShardingConfig {
        target_redundancy: TARGET,
        clamp_min_peers: 2,
        check_interval_ms: 100,
        grow_persistence: 1.0,
        shrink_persistence: 4.0,
        // The measured gossip lag cannot see the artificial staleness, so
        // the wait is set outright through its floor (as in the stale-view
        // test).
        intent_wait: 0.0,
        intent_min_wait_ms: wait_ms,
        lag_floor_ms: 250,
        lag_ceiling_ms: 500,
        agentinfo_encoding: false,
        patient_lowest: true,
        withdraw_on_stand_down: true,
    }
}

async fn add_node(
    factory: &K2GossipFunctionalTestFactory,
    others: &[Node],
) -> Node {
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

async fn view(nodes: &[Node]) -> Vec<AgentView> {
    let hs: Vec<&K2GossipFunctionalTestHarness> =
        nodes.iter().map(|(h, _)| h).collect();
    agents(&hs).await
}

#[derive(Debug, Default)]
struct Tally {
    kills: u32,
    joins: u32,
    staleness_changes: u32,
    /// Sector-samples where a count fell below target inside GRACE.
    dips_after_death: u64,
    /// Sector-samples with no holder inside GRACE.
    holes_after_death: u64,
    announced: u64,
    waited: u64,
    executed: u64,
    stood_down: u64,
    grew: u64,
    min_nodes: usize,
    max_nodes: usize,
}

fn drain_events(t: &mut Tally) {
    let events = events();
    let mut log = events.lock().unwrap();
    for (_, m) in log.iter() {
        match m.as_str() {
            ANNOUNCED => t.announced += 1,
            WAITED => t.waited += 1,
            EXECUTED => t.executed += 1,
            STOOD_DOWN => t.stood_down += 1,
            GREW => t.grew += 1,
            _ => {}
        }
    }
    log.clear();
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "slow soak; run with CHAOS_SECS=.. --ignored --nocapture"]
async fn chaos_soak_v6() {
    let _serial = SERIAL.lock().await;
    events().lock().unwrap().clear();
    let secs = env("CHAOS_SECS", 3_600);
    let seed = env("CHAOS_SEED", 1);
    // A control: CHAOS_WAIT_MS below the largest staleness breaks A1, and
    // the run should then fail. If it never does, the checks have no teeth.
    let wait_ms = env("CHAOS_WAIT_MS", WAIT_MS as u64) as u32;
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    println!("chaos: {secs} s, seed {seed}, wait {wait_ms} ms");

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
    .with_sharding_config(config(wait_ms));

    let mut nodes: Vec<Node> = Vec::new();
    for _ in 0..START_NODES {
        let node = add_node(&factory, &nodes).await;
        nodes.push(node);
    }
    for (h, _) in &nodes {
        h.wait_for_full_arc_for_all(Duration::from_secs(120)).await;
    }

    let mut t = Tally {
        min_nodes: nodes.len(),
        max_nodes: nodes.len(),
        ..Default::default()
    };
    let start = tokio::time::Instant::now();
    let end = start + Duration::from_secs(secs);
    let mut last_kill: Option<tokio::time::Instant> = None;
    let mut prev: Vec<u32>;
    let mut next_report = start + Duration::from_secs(300);

    while tokio::time::Instant::now() < end {
        // One random action.
        match rng.below(4) {
            0 if nodes.len() > MIN_NODES => {
                let now = view(&nodes).await;
                let cov = coverage(&now);
                // Victims that are not the last holder of any sector.
                let ok: Vec<usize> = (0..nodes.len())
                    .filter(|&i| {
                        let id = format!("{:?}", nodes[i].1.agent);
                        now.iter().filter(|(a, ..)| *a == id).all(
                            |(_, _, r)| {
                                r.is_none_or(|(s, e)| {
                                    cov[s as usize..e as usize]
                                        .iter()
                                        .all(|&c| c >= 2)
                                })
                            },
                        )
                    })
                    .collect();
                if !ok.is_empty() {
                    let i = ok[rng.below(ok.len() as u64) as usize];
                    drop(nodes.remove(i));
                    last_kill = Some(tokio::time::Instant::now());
                    t.kills += 1;
                }
            }
            1 if nodes.len() < MAX_NODES => {
                let node = add_node(&factory, &nodes).await;
                nodes.push(node);
                t.joins += 1;
            }
            2 => {
                let ms =
                    STALENESS_MS[rng.below(STALENESS_MS.len() as u64) as usize];
                set_sharding_view_delay(Duration::from_millis(ms));
                t.staleness_changes += 1;
            }
            _ => {}
        }
        t.min_nodes = t.min_nodes.min(nodes.len());
        t.max_nodes = t.max_nodes.max(nodes.len());
        // Membership changed: compare from the new population.
        prev = coverage(&view(&nodes).await);

        // Watch until the next action.
        let until = tokio::time::Instant::now()
            + Duration::from_millis(2_000 + rng.below(4_000));
        while tokio::time::Instant::now() < until {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let cov = coverage(&view(&nodes).await);
            let in_grace = last_kill.is_some_and(|k| k.elapsed() < GRACE);
            for (s, (&c, &p)) in cov.iter().zip(prev.iter()).enumerate() {
                if c == 0 {
                    assert!(
                        in_grace,
                        "seed {seed}, {:.0} s: sector {s} has no holder, \
                         no kill in the last {GRACE:?}",
                        start.elapsed().as_secs_f64()
                    );
                    t.holes_after_death += 1;
                }
                if c < p && c < TARGET {
                    assert!(
                        in_grace,
                        "seed {seed}, {:.0} s: a shrink took sector {s} \
                         from {p} to {c} (target {TARGET}), no kill in the \
                         last {GRACE:?}",
                        start.elapsed().as_secs_f64()
                    );
                    t.dips_after_death += 1;
                }
            }
            prev = cov;
        }
        drain_events(&mut t);
        if tokio::time::Instant::now() >= next_report {
            println!(
                "chaos {:.0} s: {} nodes; {t:?}",
                start.elapsed().as_secs_f64(),
                nodes.len()
            );
            next_report += Duration::from_secs(300);
        }
    }

    // Calm: no more churn, fresh views. Every sector back at the target.
    set_sharding_view_delay(Duration::ZERO);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(180);
    let mut since: Option<tokio::time::Instant> = None;
    loop {
        let cov = coverage(&view(&nodes).await);
        let min = *cov.iter().min().unwrap();
        let grace = last_kill.is_some_and(|k| k.elapsed() < GRACE);
        assert!(min > 0 || grace, "a sector lost its last holder at calm");
        if min >= TARGET {
            let s = *since.get_or_insert_with(tokio::time::Instant::now);
            if s.elapsed() >= Duration::from_secs(15) {
                break;
            }
        } else {
            since = None;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "seed {seed}: never got back to target after the churn (min {min})"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    drain_events(&mut t);
    println!(
        "chaos done, seed {seed}: {} nodes at the end; {t:?}",
        nodes.len()
    );
}
