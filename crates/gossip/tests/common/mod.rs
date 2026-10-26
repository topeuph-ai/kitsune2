//! Helpers shared by the sharding functional tests.
#![allow(dead_code)]

use kitsune2_api::{DhtArc, Timestamp};
use kitsune2_core::factories::MemoryOp;
use kitsune2_dht::SECTOR_SIZE;
use kitsune2_gossip::harness::{K2GossipFunctionalTestHarness, MemoryOpRecord};
use kitsune2_test_utils::random_bytes;
use std::sync::{Arc, Mutex, OnceLock};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

pub const NUM_SECTORS: u32 = u32::MAX / SECTOR_SIZE + 1;

pub const ANNOUNCED: &str = "sharding: announcing shrink intent";
pub const WAITED: &str = "sharding: lowest racer waiting at re-check";
pub const EXECUTED: &str = "sharding: executed polite shrink";
pub const STOOD_DOWN: &str = "sharding: shrink intent cancelled at re-check";
pub const GREW: &str = "sharding: growing target arc";
pub const GREW_SHORT: &str = "sharding: grow cause: sibling half short";
pub const GREW_INTENTS: &str = "sharding: grow cause: announced vacates only";
pub const WITHDRAWAL_APPLIED: &str =
    "sharding: applied a withdrawal from its sender";
pub const SENDER_MISMATCH: &str =
    "sharding: intents not from their agent's declared URL";

/// Whether the functional tests run with `require_recent_contact`: on
/// unless `SHARDING_RECENT_CONTACT=0`, so the whole suite can be compared
/// with and without it.
pub fn recent_contact() -> bool {
    std::env::var("SHARDING_RECENT_CONTACT").map_or(true, |v| v != "0")
}

/// Whether the functional tests run with V7's coordinated growth: off
/// unless `SHARDING_COORDINATED_GROWTH=1`, so the whole suite can be run
/// both ways.
pub fn coordinated_growth() -> bool {
    std::env::var("SHARDING_COORDINATED_GROWTH").is_ok_and(|v| v == "1")
}

pub const GROW_ANNOUNCED: &str = "sharding: announcing grow intent";
pub const GROW_EXECUTED: &str = "sharding: executed coordinated grow";
pub const GROW_STOOD_DOWN: &str =
    "sharding: grow intent stood down, already covered";
pub const CRITICAL_REPAIR: &str = "sharding: critical repair, growing at once";

/// Whether the functional tests probe sibling-half holders: on unless
/// `SHARDING_PROBE=0`.
pub fn probe_holders() -> bool {
    std::env::var("SHARDING_PROBE").map_or(true, |v| v != "0")
}

/// Sharding-module log events, `(agent, message)`, in arrival order.
pub type Events = Arc<Mutex<Vec<(String, String)>>>;

/// Install (once per test binary) a subscriber that records the
/// sharding module's events, and return the shared record.
pub fn events() -> Events {
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

pub struct Recorder(Events);

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
pub static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// One agent's declared arc as a half-open sector range.
pub fn sectors(arc: DhtArc) -> Option<(u32, u32)> {
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
pub type AgentView = (String, u32, Option<(u32, u32)>);

/// Every local agent's view.
pub async fn agents(
    nodes: &[&K2GossipFunctionalTestHarness],
) -> Vec<AgentView> {
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

pub fn coverage(agents: &[AgentView]) -> Vec<u32> {
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
pub fn could_shrink(agents: &[AgentView], target: u32) -> Vec<String> {
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
        if without_me > target {
            out.push(id.clone());
        }
    }
    out
}

/// Seed some ops so gossip rounds have something to verify.
pub async fn seed_ops(node: &K2GossipFunctionalTestHarness) {
    let mut op_store = node.op_store.write().await;
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
