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

use kitsune2_api::{AgentInfoSigned, DhtArc, Timestamp};
use kitsune2_core::factories::MemoryOp;
use kitsune2_dht::SECTOR_SIZE;
use kitsune2_gossip::harness::{
    K2GossipFunctionalTestFactory, K2GossipFunctionalTestHarness,
    MemoryOpRecord,
};
use kitsune2_gossip::{K2GossipConfig, K2ShardingConfig};
use kitsune2_test_utils::random_bytes;
use kitsune2_test_utils::space::TEST_SPACE_ID;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use tracing_subscriber::Layer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

const NUM_SECTORS: u32 = u32::MAX / SECTOR_SIZE + 1;
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

const ANNOUNCED: &str = "sharding: announcing shrink intent";
const WAITED: &str = "sharding: lowest racer waiting at re-check";
const EXECUTED: &str = "sharding: executed polite shrink";
const STOOD_DOWN: &str = "sharding: shrink intent cancelled at re-check";
const GREW: &str = "sharding: growing target arc";

/// Sharding-module log events, `(agent, message)`, in arrival order.
type Events = Arc<Mutex<Vec<(String, String)>>>;

/// Install (once per test binary) a subscriber that records the
/// sharding module's events, and return the shared record.
fn events() -> Events {
    static EVENTS: OnceLock<Events> = OnceLock::new();
    EVENTS
        .get_or_init(|| {
            let events: Events = Arc::default();
            let layer = Recorder(events.clone()).with_filter(
                tracing_subscriber::filter::filter_fn(|m| {
                    m.target().starts_with("kitsune2_gossip::sharding")
                }),
            );
            tracing_subscriber::registry().with(layer).init();
            events
        })
        .clone()
}

struct Recorder(Events);

impl<S: tracing::Subscriber> Layer<S> for Recorder {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        #[derive(Default)]
        struct Fields {
            agent: String,
            message: String,
        }
        impl tracing::field::Visit for Fields {
            fn record_debug(
                &mut self,
                field: &tracing::field::Field,
                value: &dyn std::fmt::Debug,
            ) {
                match field.name() {
                    "agent" => self.agent = format!("{value:?}"),
                    "message" => self.message = format!("{value:?}"),
                    _ => {}
                }
            }
        }
        let mut f = Fields::default();
        event.record(&mut f);
        self.0.lock().unwrap().push((f.agent, f.message));
    }
}

/// Tests in this binary share the global event record; run one at a time.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// One agent's declared arc as a half-open sector range.
fn sectors(arc: DhtArc) -> Option<(u32, u32)> {
    match arc {
        DhtArc::Empty => None,
        DhtArc::Arc(s, e) if s <= e => {
            Some((s / SECTOR_SIZE, e / SECTOR_SIZE + 1))
        }
        // Quantised blocks never wrap; the full ring is FULL.
        DhtArc::Arc(..) => Some((0, NUM_SECTORS)),
    }
}

/// One local agent: (id, home sector, declared sector range).
type AgentView = (String, u32, Option<(u32, u32)>);

/// Every local agent's view.
async fn agents(nodes: &[&K2GossipFunctionalTestHarness]) -> Vec<AgentView> {
    let mut out = Vec::new();
    for node in nodes {
        for agent in node.space.local_agent_store().get_all().await.unwrap() {
            out.push((
                format!("{:?}", agent.agent()),
                agent.agent().loc() / SECTOR_SIZE,
                sectors(agent.get_cur_storage_arc()),
            ));
        }
    }
    out
}

fn coverage(agents: &[AgentView]) -> Vec<u32> {
    let mut cov = vec![0u32; NUM_SECTORS as usize];
    for (_, _, range) in agents {
        if let Some((s, e)) = range {
            for c in &mut cov[*s as usize..*e as usize] {
                *c += 1;
            }
        }
    }
    cov
}

/// The agents that could still take a safe shrink: the half of their
/// block not containing home keeps more than the target without them —
/// exactly the controller's shrink condition.
fn could_shrink(agents: &[AgentView]) -> Vec<String> {
    let cov = coverage(agents);
    let mut out = Vec::new();
    for (id, home, range) in agents {
        let Some((s, e)) = *range else { continue };
        if e - s < 2 {
            continue;
        }
        let mid = s + (e - s) / 2;
        let (vs, ve) = if *home < mid { (mid, e) } else { (s, mid) };
        let without_me = cov[vs as usize..ve as usize]
            .iter()
            .map(|c| c - 1)
            .min()
            .unwrap();
        if without_me > TARGET_REDUNDANCY {
            out.push(id.clone());
        }
    }
    out
}

#[derive(Debug, Default)]
struct RunStats {
    announced: usize,
    waited: usize,
    executed: usize,
    stood_down: usize,
    /// Grow decisions after every sector first reached the target.
    grew: usize,
    /// Seconds from every sector first reaching the target to the last
    /// change in declared coverage.
    settle_secs: f64,
    /// Total declared sector-copies at the end, against 8 x the ring.
    final_copies: u32,
}

/// Build the network, let it shard down under a forced race, and assert
/// safety, progress, and (if anyone waited) that waiters went on to
/// execute.
async fn race_run(patient_lowest: bool) -> RunStats {
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
    });

    let a = factory.new_instance().await;
    {
        let mut op_store = a.op_store.write().await;
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
    let settled = loop {
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
            break now_agents;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "network never settled after shrinking (executed any: \
             {executed})"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    };

    assert_eq!(NUM_AGENTS as usize, settled.len());

    // Progress: nobody is left holding a shrink it could safely take.
    let stuck = could_shrink(&settled);
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
        "settled with {} agent(s) still able to shrink safely: a stall",
        stuck.len()
    );

    // Tally, and check that every waiter went on to execute.
    let ours: std::collections::HashSet<String> =
        settled.iter().map(|(id, ..)| id.clone()).collect();
    let log = events.lock().unwrap().clone();
    let mut stats = RunStats {
        settle_secs: (stable_since - armed_at).as_secs_f64(),
        final_copies: coverage(&settled).iter().sum(),
        ..Default::default()
    };
    let mut waited_then: HashMap<String, bool> = HashMap::new();
    for (i, (agent, msg)) in log.iter().enumerate() {
        if !ours.contains(agent) || (msg == GREW && i < armed_log_len) {
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
            _ => {}
        }
    }
    assert!(stats.executed >= 1, "no shrink executed");
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
        let stats = race_run(true).await;
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
    let stats = race_run(false).await;
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
        for (name, patient) in [("V3", false), ("V6", true)] {
            let s = race_run(patient).await;
            println!(
                "compare,{name},{run},{},{},{},{},{},{:.1},{}",
                s.announced,
                s.stood_down,
                s.waited,
                s.executed,
                s.grew,
                s.settle_secs,
                s.final_copies
            );
        }
    }
}
