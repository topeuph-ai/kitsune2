//! Functional tests for V7: coordinated growth and pinned nodes, over real
//! gossip on the in-memory transport.
//!
//! - **The herd.** Twelve nodes, one agent each, shard down from full arcs
//!   and settle; then three nodes die at once (chosen so no sector loses
//!   its last holder). Without coordination every agent that sees a short
//!   sector grows; with `coordinated_growth`, agents announce the range,
//!   wait, and grow only if it is still short counting lower-id announced
//!   growers. Asserted both ways: no sector is left without a holder after
//!   the death-detection window, and every short sector that some agent
//!   could grow into is repaired within 60 s. (Full recovery to the target is
//!   not asserted: growth is sibling-half-local, so a thin sector with no
//!   neighbour to grow into stays thin; the clamp, or the simulator's
//!   expanding-ring repair, is the designed path for that.)
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
        probe_sibling_holders: probe_holders(),
        probe_interval_ms: 1_000,
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
    /// "grow cause" decisions logged (the grow condition met).
    grow_causes: usize,
    grows_after_kill: usize,
    grow_announced: usize,
    grow_executed: usize,
    grow_stood_down: usize,
    /// Grow withdrawals applied by receivers.
    grow_withdrawals: usize,
    critical: usize,
    holes_in_grace: usize,
    /// Fewest copies of any sector, 60 s after the kill.
    min_end: u32,
    /// Sectors still below the target then.
    short_end: usize,
    /// ... of which some agent's sibling half contains them.
    short_reachable: usize,
    /// Agents whose declared arc lags their target, with both ranges.
    mid_grow: Vec<String>,
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

    // Watch for 60 s: no hole after the grace window.
    let mut holes_in_grace = 0;
    let mut min_end = 0;
    let watch_until = killed_at + Duration::from_secs(60);
    while tokio::time::Instant::now() < watch_until {
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
        min_end = *cov.iter().min().unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    // Let grows announced near the end resolve before counting (a re-check
    // comes one wait, at most 2.5 x the 2 s lag ceiling, after announcing).
    tokio::time::sleep(Duration::from_secs(12)).await;
    let end_view = view(&nodes).await;
    // How out of date are the survivors' peer stores? For each survivor,
    // count live agents whose arc in that store differs from the arc they
    // actually declare now, and dead agents still listed and not marked
    // unresponsive.
    let live: std::collections::HashMap<String, Option<(u32, u32)>> =
        end_view.iter().map(|(a, _, r)| (a.clone(), *r)).collect();
    let mut stale_arcs = 0;
    let mut wider = 0;
    let mut dead_listed = 0;
    for (h, _) in &nodes {
        for info in h.space.peer_store().get_all().await.unwrap() {
            let id = format!("{:?}", info.agent);
            match live.get(&id) {
                Some(actual) => {
                    let seen = sectors(info.storage_arc);
                    if seen != *actual {
                        stale_arcs += 1;
                        let w =
                            |r: Option<(u32, u32)>| r.map_or(0, |(s, e)| e - s);
                        if w(seen) > w(*actual) {
                            wider += 1;
                        }
                    }
                }
                None => {
                    let unresponsive = match info.url.clone() {
                        Some(u) => matches!(
                            h.space.peer_meta_store().get_unresponsive(u).await,
                            Ok(Some(_))
                        ),
                        None => false,
                    };
                    if !info.is_tombstone && !unresponsive {
                        dead_listed += 1;
                    }
                }
            }
        }
    }
    println!(
        "peer stores: {stale_arcs} out-of-date arcs ({wider} wider than          actual), {dead_listed} dead agents listed and not marked"
    );
    // Agents whose declared arc has not caught up with their target: a
    // grow still syncing, or stuck. The controller makes no decision for
    // such an agent until it catches up.
    let mut mid_grow = Vec::new();
    for (h, _) in &nodes {
        for a in h.space.local_agent_store().get_all().await.unwrap() {
            let (cur, tgt) = (a.get_cur_storage_arc(), a.get_tgt_storage_arc());
            if cur != tgt {
                mid_grow.push(format!(
                    "cur {:?} tgt {:?}",
                    sectors(cur),
                    sectors(tgt)
                ));
            }
        }
    }
    let end_cov = coverage(&end_view);
    let short_end = end_cov.iter().filter(|&&c| c < TARGET).count();
    // How many short sectors lie in some agent's sibling half, the only
    // place an agent can grow into? The rest no local grow can reach.
    let reachable = (0..NUM_SECTORS)
        .filter(|&x| end_cov[x as usize] < TARGET)
        .filter(|&x| {
            end_view.iter().any(|(_, _, r)| {
                let Some((s, e)) = *r else { return false };
                let len = e - s;
                if len >= NUM_SECTORS {
                    return false;
                }
                let parent = (s / (2 * len)) * (2 * len);
                let (ps, pe) = (parent, parent + 2 * len);
                ps <= x && x < pe && !(s <= x && x < e)
            })
        })
        .count();
    let log = events.lock().unwrap();
    let after = &log[mark..];
    let count = |what: &str| after.iter().filter(|(_, m)| m == what).count();
    Herd {
        grow_causes: count(GREW_SHORT) + count(GREW_INTENTS),
        grows_after_kill: count(GREW),
        grow_announced: count(GROW_ANNOUNCED),
        grow_executed: count(GROW_EXECUTED),
        grow_stood_down: count(GROW_STOOD_DOWN),
        grow_withdrawals: count(GROW_WITHDRAWAL_APPLIED),
        critical: count(CRITICAL_REPAIR),
        holes_in_grace,
        min_end,
        short_end,
        short_reachable: reachable,
        mid_grow,
    }
}

/// Asserted for both settings: no hole after the grace window (inside
/// `herd`), and every short sector some agent could grow into has been
/// repaired within 60 s. With coordination, every announced grow is
/// resolved (grown or stood down). How many agents grow is printed but not
/// compared: each run draws new random agent ids, so the two runs face
/// different networks; the like-for-like comparison is the simulator's
/// (polite-shrink `reports/REPORT_v7.md`).
#[tokio::test(flavor = "multi_thread")]
async fn growth_repairs_deaths_with_and_without_coordination() {
    let _serial = SERIAL.lock().await;
    for coordinated in [false, true] {
        let h = herd(coordinated).await;
        println!("coordinated growth {coordinated}: {h:?}");
        assert_eq!(
            0, h.short_reachable,
            "short sectors that an agent could grow into were not repaired              (coordinated {coordinated}): {h:?}"
        );
        if coordinated {
            assert_eq!(
                h.grow_announced,
                h.grow_executed + h.grow_stood_down,
                "an announced grow was neither executed nor stood down: {h:?}"
            );
            // A grower that stands down tells the others, who drop its
            // intent at once instead of counting it as arriving.
            assert!(
                h.grow_stood_down == 0 || h.grow_withdrawals > 0,
                "grows stood down but no withdrawal was applied: {h:?}"
            );
        } else {
            assert_eq!(0, h.grow_announced, "V6 announced a grow: {h:?}");
        }
    }
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
            && armed
        {
            assert_eq!(
                Some((0, NUM_SECTORS)),
                *range,
                "the pinned agent shrank"
            );
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
