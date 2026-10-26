//! Diagnostic (ignored): how long do survivors take to mark killed peers
//! unresponsive on the in-memory transport? Run with --ignored --nocapture.
#![cfg(feature = "sharding")]

mod common;

use common::*;
use kitsune2_api::{AgentInfoSigned, DhtArc};
use kitsune2_gossip::harness::K2GossipFunctionalTestFactory;
use kitsune2_gossip::{K2GossipConfig, K2ShardingConfig};
use kitsune2_test_utils::space::TEST_SPACE_ID;
use std::sync::Arc;
use std::time::Duration;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "diagnostic"]
async fn time_to_mark_dead_peers_unresponsive() {
    let _serial = SERIAL.lock().await;
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
        target_redundancy: 3,
        clamp_min_peers: 2,
        check_interval_ms: 100,
        intent_min_wait_ms: 1_000,
        lag_floor_ms: 250,
        lag_ceiling_ms: 2_000,
        ..Default::default()
    });
    let mut nodes = Vec::new();
    let mut infos: Vec<Arc<AgentInfoSigned>> = Vec::new();
    for _ in 0..12 {
        let n = factory.new_instance().await;
        seed_ops(&n).await;
        infos.push(n.join_local_agent(DhtArc::FULL).await);
        nodes.push(n);
    }
    for n in &nodes {
        n.space.peer_store().insert(infos.clone()).await.unwrap();
    }
    tokio::time::sleep(Duration::from_secs(30)).await;
    let dead: Vec<_> =
        infos[..3].iter().map(|i| i.url.clone().unwrap()).collect();
    let survivors = nodes.split_off(3);
    drop(nodes);
    let t0 = tokio::time::Instant::now();
    let mut first_all: Option<f64> = None;
    let mut per_tick = Vec::new();
    while t0.elapsed() < Duration::from_secs(120) {
        let mut marked = 0;
        for s in &survivors {
            for url in &dead {
                if let Ok(Some(_)) = s
                    .space
                    .peer_meta_store()
                    .get_unresponsive(url.clone())
                    .await
                {
                    marked += 1;
                }
            }
        }
        per_tick.push((t0.elapsed().as_secs_f64(), marked));
        if marked == survivors.len() * dead.len() && first_all.is_none() {
            first_all = Some(t0.elapsed().as_secs_f64());
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    let total = survivors.len() * dead.len();
    for (t, m) in per_tick.iter().step_by(4) {
        println!("t={t:5.1}s  marked {m}/{total}");
    }
    println!("all marked after: {first_all:?}");
}
