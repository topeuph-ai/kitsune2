//! The sharding controller: decides target arcs and coordinates shrinks.

use super::blocks::{
    MAX_LEVEL, block_arc, home_sector, sibling_half, vacate_half,
};
use super::coverage::{min_over, others_coverage, subtract_sectors};
use super::intents::{
    IntentTable, ShrinkIntent, retain_declared, retain_growers,
};
use super::protocol::{
    K2ShardingMessage, K2ShardingProbeMessage, K2ShardingShrinkIntentMessage,
    deserialize_sharding_message, k2_sharding_message,
    serialize_sharding_message,
};
use super::{K2ShardingConfig, SHARDING_MOD_NAME};
use crate::gossip::DropAbortHandle;
use crate::peer_meta_store::K2PeerMetaStore;
use kitsune2_api::{
    AgentId, AgentInfoSigned, DhtArc, DynLocalAgent, DynLocalAgentStore,
    DynPeerStore, DynTransport, K2Result, SpaceId, Timestamp, TxBaseHandler,
    TxModuleHandler, Url, WeakDynTransport,
};
use kitsune2_dht::SECTOR_SIZE;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::sync::Mutex;

/// Per-local-agent controller state.
#[derive(Debug)]
struct AgentControl {
    /// The agent's home sector.
    home: u32,
    /// The block level of the arc this agent currently declares.
    declared_level: u8,
    /// The block level the agent is growing towards. Greater than
    /// `declared_level` while a grow is being synced by the gossip
    /// machinery; equal otherwise.
    target_level: u8,
    /// How long the grow condition has held continuously.
    grow_acc: Duration,
    /// How long the shrink condition has held continuously.
    shrink_acc: Duration,
    /// An announced, not yet executed, shrink.
    pending: Option<PendingShrink>,
    /// An announced, not yet executed, grow (V7 coordinated growth). Its
    /// `vacate` field is the range to take.
    pending_grow: Option<PendingShrink>,
    /// A coordinated grow that has been executed and is being synced. Its
    /// intent is kept announced until the declared arc covers the range,
    /// so peers do not grow into the same range while the data arrives.
    syncing_grow: Option<PendingShrink>,
}

#[derive(Debug)]
struct PendingShrink {
    /// When to re-check and execute.
    execute_at: Timestamp,
    /// Half-open sector range that would be vacated.
    vacate: (u32, u32),
    /// When the announced intent expires at peers.
    expires_at: Timestamp,
    /// The announce-to-execute wait this intent was given.
    wait: Duration,
}

/// The sharding module implementation.
///
/// Runs one controller tick per configured interval over every local
/// agent, and receives shrink-intent announcements from peers as a
/// transport module handler. See the `sharding` module source docs and
/// [`K2ShardingConfig`](crate::K2ShardingConfig) for the decision rules.
#[derive(Debug)]
pub struct K2Sharding {
    config: Arc<K2ShardingConfig>,
    space_id: SpaceId,
    peer_store: DynPeerStore,
    local_agent_store: DynLocalAgentStore,
    peer_meta_store: Arc<K2PeerMetaStore>,
    transport: WeakDynTransport,
    intents: IntentTable,
    /// Announced grows (V7 coordinated growth), same shape as shrinks.
    grow_intents: IntentTable,
    states: Mutex<HashMap<AgentId, AgentControl>>,
    /// Peers seen as gone (unresponsive or tombstoned) last tick, to
    /// detect fresh departures.
    last_gone: std::sync::Mutex<HashSet<AgentId>>,
    /// Per peer URL: the completed-rounds count last seen, and when it
    /// last went up (when this node last heard from the peer), if ever
    /// observed. See `require_recent_contact`.
    contact: std::sync::Mutex<HashMap<Url, (u32, Option<Timestamp>)>>,
    /// When each peer was last probed (see `probe_sibling_holders`).
    last_probe: std::sync::Mutex<HashMap<Url, Timestamp>>,
    /// Test-only artificial staleness (see [`view_delay`]).
    #[cfg(any(test, feature = "test-utils"))]
    view_delay: view_delay::ViewDelay,
    _check_task: OnceLock<DropAbortHandle>,
}

impl K2Sharding {
    /// Construct a new [K2Sharding] instance and start its check task.
    pub fn create(
        config: K2ShardingConfig,
        space_id: SpaceId,
        peer_store: DynPeerStore,
        local_agent_store: DynLocalAgentStore,
        peer_meta_store: Arc<K2PeerMetaStore>,
        transport: DynTransport,
    ) -> Arc<K2Sharding> {
        let sharding = Arc::new(K2Sharding {
            config: Arc::new(config),
            space_id: space_id.clone(),
            peer_store,
            local_agent_store,
            peer_meta_store,
            transport: Arc::downgrade(&transport),
            intents: IntentTable::default(),
            grow_intents: IntentTable::default(),
            states: Mutex::new(HashMap::new()),
            last_gone: std::sync::Mutex::new(HashSet::new()),
            contact: std::sync::Mutex::new(HashMap::new()),
            last_probe: std::sync::Mutex::new(HashMap::new()),
            #[cfg(any(test, feature = "test-utils"))]
            view_delay: Default::default(),
            _check_task: OnceLock::new(),
        });

        transport.register_module_handler(
            space_id,
            SHARDING_MOD_NAME.to_string(),
            sharding.clone(),
        );

        let weak = Arc::downgrade(&sharding);
        let interval = sharding.config.check_interval();
        let handle = tokio::task::spawn(async move {
            loop {
                tokio::time::sleep(interval).await;
                let Some(sharding) = weak.upgrade() else {
                    tracing::info!(
                        "Sharding instance dropped, stopping check task"
                    );
                    break;
                };
                if let Err(e) = sharding.check(interval).await {
                    tracing::warn!(?e, "Sharding check failed");
                }
            }
        })
        .abort_handle();
        sharding
            ._check_task
            .set(DropAbortHandle {
                name: "Sharding check task".to_string(),
                handle,
            })
            .expect("Check task set twice");

        sharding
    }

    /// One controller tick over all local agents.
    async fn check(&self, elapsed: Duration) -> K2Result<()> {
        let local_agents = self.local_agent_store.get_all().await?;
        if local_agents.is_empty() {
            return Ok(());
        }
        let peers = self.peer_store.get_all().await?;
        let now = Timestamp::now();
        let live_intents = self.intents.live(now);
        let live_grows = self.grow_intents.live(now);
        // Tests can make this node act on a view of the peer store and
        // the intent tables that is a fixed time old, all from the same
        // instant, to exercise the stale-view race on a fast transport.
        #[cfg(any(test, feature = "test-utils"))]
        let (peers, live_intents, live_grows) =
            self.view_delay.apply(now, peers, live_intents, live_grows);
        let mut live_grows = live_grows;

        let own_ids = local_agents
            .iter()
            .map(|a| a.agent().clone())
            .collect::<HashSet<_>>();

        // A dead peer's agent info lingers in the peer store until it
        // expires, long after gossip has marked the peer unresponsive.
        // Coverage must not count such ghosts, or the controller would
        // sit on its hands through exactly the die-off it exists to
        // absorb. Excluding them is how a disruption becomes visible
        // here, the analogue of departed arcs ageing out of a declared
        // snapshot.
        let mut gone = own_ids.clone();
        for peer in &peers {
            if peer.is_tombstone || gone.contains(&peer.agent) {
                continue;
            }
            let Some(url) = peer.url.clone() else {
                continue;
            };
            // Tests can simulate deaths that are never detected.
            #[cfg(any(test, feature = "test-utils"))]
            if view_delay::ignore_unresponsive() {
                continue;
            }
            if let Ok(Some(_)) =
                self.peer_meta_store.get_unresponsive(url).await
            {
                gone.insert(peer.agent.clone());
            }
        }

        // Redundancy provided by (responsive) remote peers only; each
        // agent's view adds this node's *other* local agents back in
        // below.
        let base_cov = others_coverage(&peers, &gone);

        let visible_peers = peers
            .iter()
            .filter(|p| !p.is_tombstone && !gone.contains(&p.agent))
            .count()
            + local_agents.len();

        let mut live_intents = live_intents;
        // Range-validation: only count an intent over sectors its
        // announcer actually declares (see retain_declared). Uses the
        // same peer snapshot as this tick's coverage, so intent and arc
        // are judged on one consistent view.
        let declared_arcs: HashMap<AgentId, (DhtArc, Option<Url>)> = peers
            .iter()
            .filter(|p| !p.is_tombstone)
            .map(|p| (p.agent.clone(), (p.storage_arc, p.url.clone())))
            .collect();
        let sender_mismatches = live_intents
            .iter()
            .filter(|(a, i)| {
                i.from.is_some()
                    && declared_arcs
                        .get(a)
                        .is_some_and(|(_, url)| *url != i.from)
            })
            .count();
        if sender_mismatches > 0 {
            tracing::debug!(
                sender_mismatches,
                "sharding: intents not from their agent's declared URL"
            );
        }
        retain_declared(&mut live_intents, &declared_arcs, &own_ids);
        retain_growers(&mut live_grows, &declared_arcs, &own_ids, &gone);
        // Per-peer bookkeeping only for peers still in the store.
        {
            let urls: HashSet<&Url> =
                peers.iter().filter_map(|p| p.url.as_ref()).collect();
            self.contact
                .lock()
                .expect("poisoned")
                .retain(|u, _| urls.contains(u));
            self.last_probe
                .lock()
                .expect("poisoned")
                .retain(|u, _| urls.contains(u));
        }
        let lag = self.lag_estimate(&peers, &gone, now).await;

        // Coverage for shrink decisions: with `require_recent_contact`,
        // only holders heard from within the current wait count.
        let shrink_base_cov = if self.config.require_recent_contact {
            let silent = self.silent_peers(&peers, &gone, lag, now).await;
            let mut skip = gone.clone();
            skip.extend(silent);
            others_coverage(&peers, &skip)
        } else {
            base_cov.clone()
        };
        // Coverage for grow decisions: every holder not marked
        // unresponsive, as under V6. (Reading silence as absence here, as
        // shrink decisions do, set off needless grows: gossip reaches peers
        // in rotation, so healthy peers routinely go unheard for a while.
        // And deaths are marked quickly: on the in-memory transport, all
        // killed peers within 12 s, most within 2 s; see sharding_detect.)
        let grow_base_cov = base_cov.clone();

        let mut states = self.states.lock().await;

        // Shrinks the storm brake cancels; peers are told after the loop.
        let mut cancelled = Vec::new();
        // Storm brake. An announced shrink's re-check can only account
        // for deaths it can see, and death visibility (unresponsive
        // marking) runs on a different clock than gossip staleness. If
        // any peer has newly become invisible since the last tick, the
        // evidence behind every pending intent is suspect: cancel them
        // all and let the persistence counters accumulate afresh. The
        // sim gets this for free because a single lag parameter bounds
        // both; the real system does not.
        {
            let mut last_gone = self.last_gone.lock().expect("poisoned");
            let newly_gone = gone.difference(&last_gone).any(|a| {
                // Own agents appearing in `gone` initially is not a
                // departure.
                !own_ids.contains(a)
            });
            *last_gone = gone.clone();
            if newly_gone {
                for (agent_id, ctl) in states.iter_mut() {
                    if let Some(p) = ctl.pending.take() {
                        cancelled.push((agent_id.clone(), p));
                        ctl.shrink_acc = Duration::ZERO;
                        self.intents.remove(agent_id);
                        if self.config.agentinfo_encoding {
                            // The announcement was a narrowed arc claim, so
                            // cancelling it means widening the claim back.
                            // Without this the agent is left declaring less
                            // than `target_level`, and the arc-match guard in
                            // `tick_agent` then blocks every later decision —
                            // including the growth this brake exists to allow.
                            if let Some(agent) = local_agents
                                .iter()
                                .find(|a| a.agent() == agent_id)
                            {
                                let arc =
                                    block_arc(ctl.home, ctl.declared_level);
                                agent.set_tgt_storage_arc_hint(arc);
                                agent.set_cur_storage_arc(arc);
                                agent.invoke_cb();
                            }
                        }
                        tracing::info!(
                            agent = ?agent_id,
                            "sharding: peer loss detected, cancelling \
                             pending shrink intent"
                        );
                    }
                }
            }
        }
        // Tell peers, so they stop counting the cancelled shrinks as
        // leaving (they would otherwise do so until the intents expire).
        if self.config.withdraw_on_stand_down && !self.config.agentinfo_encoding
        {
            for (agent_id, p) in cancelled {
                if let Some(agent) =
                    local_agents.iter().find(|a| a.agent() == &agent_id)
                {
                    self.withdraw_intent(agent, p.vacate, p.expires_at).await;
                }
            }
        }
        for agent in &local_agents {
            // The declared redundancy this agent sees around itself:
            // remote peers plus sibling local agents, excluding itself.
            let mut shrink_cov = shrink_base_cov.clone();
            let mut grow_cov = grow_base_cov.clone();
            for other in &local_agents {
                if other.agent() != agent.agent() {
                    add_cur_arc(&mut shrink_cov, &other.get_cur_storage_arc());
                    add_cur_arc(&mut grow_cov, &other.get_cur_storage_arc());
                }
            }

            let ctl =
                states.entry(agent.agent().clone()).or_insert_with(|| {
                    // Start at the full arc, matching kitsune2's current
                    // fixed behaviour; the controller shrinks from there.
                    agent.set_tgt_storage_arc_hint(DhtArc::FULL);
                    AgentControl {
                        home: home_sector(agent.agent().loc()),
                        declared_level: MAX_LEVEL,
                        target_level: MAX_LEVEL,
                        grow_acc: Duration::ZERO,
                        shrink_acc: Duration::ZERO,
                        pending: None,
                        pending_grow: None,
                        syncing_grow: None,
                    }
                });

            self.tick_agent(
                agent,
                ctl,
                shrink_cov,
                grow_cov,
                &live_intents,
                &live_grows,
                visible_peers,
                lag,
                now,
                elapsed,
            )
            .await;
        }

        // Forget agents that have left the space.
        states.retain(|id, _| own_ids.contains(id));

        if self.config.probe_sibling_holders {
            let halves: Vec<(u32, u32)> = states
                .values()
                .filter(|ctl| ctl.declared_level < MAX_LEVEL)
                .map(|ctl| sibling_half(ctl.home, ctl.declared_level))
                .collect();
            drop(states);
            self.probe_holders(&peers, &gone, &halves, now).await;
        }

        Ok(())
    }

    /// The decision rules for one agent. Direct port of the simulated
    /// controller: pending intents execute first, in-flight grows gate
    /// decisions, the small-network clamp overrides everything, then
    /// grow/shrink conditions accumulate against their persistence
    /// thresholds, with grow taking priority.
    #[allow(clippy::too_many_arguments)]
    async fn tick_agent(
        &self,
        agent: &DynLocalAgent,
        ctl: &mut AgentControl,
        shrink_cov: Vec<u32>,
        grow_cov: Vec<u32>,
        live_intents: &[(AgentId, ShrinkIntent)],
        live_grows: &[(AgentId, ShrinkIntent)],
        visible_peers: usize,
        lag: Duration,
        now: Timestamp,
        elapsed: Duration,
    ) {
        let cfg = &self.config;

        // Phase two of an announced shrink; no other decisions while an
        // intent is pending.
        if let Some(pending) = &ctl.pending {
            if now >= pending.execute_at {
                self.execute_intent(agent, ctl, &shrink_cov, live_intents, now)
                    .await;
            }
            return;
        }
        if let Some(pending) = &ctl.pending_grow {
            // A range that falls to one copy or none while we wait is
            // repaired at once; the announcement stays up while it syncs.
            if cfg.critical_repair
                && is_critical(
                    &grow_view(&grow_cov, live_intents, agent.agent()),
                    pending.vacate,
                )
            {
                tracing::debug!(
                    agent = ?agent.agent(),
                    "sharding: critical repair while a grow was pending"
                );
                let pending = ctl.pending_grow.take().expect("checked");
                self.start_grow(agent, ctl);
                self.keep_grow_announced(agent, ctl, pending);
                return;
            }
            if now >= pending.execute_at {
                self.execute_grow(
                    agent,
                    ctl,
                    &grow_cov,
                    live_intents,
                    live_grows,
                    now,
                )
                .await;
            }
            return;
        }

        // The declared arc must have caught up with the model before any
        // decision: a growing (or freshly joined, still empty) arc means
        // the verified sync is in flight, and deciding from an arc we do
        // not actually hold could shrink us onto data we never synced.
        if agent.get_cur_storage_arc() != block_arc(ctl.home, ctl.target_level)
        {
            // A coordinated grow stays announced until it has synced, so
            // peers do not take the same range while the data arrives.
            if let Some(g) = &mut ctl.syncing_grow
                && let Some(expires_at) = refreshed_expiry(
                    g.expires_at,
                    now + cfg.check_interval(),
                    g.wait,
                )
            {
                g.expires_at = expires_at;
                let range = g.vacate;
                self.publish_grow_intent(agent, range, expires_at).await;
            }
            return;
        }
        if ctl.syncing_grow.take().is_some() {
            // Synced: the declared arc now covers the range, and peers
            // drop the intent once they see it (retain_growers).
            self.grow_intents.remove(agent.agent());
        }
        ctl.declared_level = ctl.target_level;

        // Pinned: hold the full arc and never shrink.
        if cfg.pinned {
            if ctl.declared_level < MAX_LEVEL {
                self.start_grow(agent, ctl);
            }
            return;
        }

        // Small-network clamp: too few visible peers, hold a full arc.
        if (visible_peers as u32) < cfg.clamp_min_peers {
            if ctl.declared_level < MAX_LEVEL {
                self.start_grow(agent, ctl);
            }
            return;
        }

        // Grow check: would the sibling half be under-covered? Announced
        // vacates count as already gone, so we move before the hole opens.
        let mut grow_cond = false;
        // Whether the sibling half is short on declared coverage alone,
        // before announced vacates are subtracted. Only used to label the
        // grow decision in the logs.
        let mut grow_short_without_intents = false;
        if ctl.declared_level < MAX_LEVEL {
            let mut eff = grow_cov.clone();
            for (id, intent) in live_intents {
                if id != agent.agent() {
                    subtract_sectors(
                        &mut eff,
                        intent.vacate.0,
                        intent.vacate.1,
                    );
                }
            }
            let (ws, we) = sibling_half(ctl.home, ctl.declared_level);
            grow_cond = min_over(&eff, ws, we) < cfg.target_redundancy;
            grow_short_without_intents =
                min_over(&grow_cov, ws, we) < cfg.target_redundancy;
        }

        // Shrink check: does the half we would drop stay strictly above
        // the target without us?
        let mut shrink_cond = false;
        if ctl.declared_level > 0 {
            let (vs, ve) = vacate_half(ctl.home, ctl.declared_level);
            shrink_cond = min_over(&shrink_cov, vs, ve) > cfg.target_redundancy;
        }

        // Hysteresis scaled to our own measured staleness.
        ctl.grow_acc = if grow_cond {
            ctl.grow_acc + elapsed
        } else {
            Duration::ZERO
        };
        ctl.shrink_acc = if shrink_cond {
            ctl.shrink_acc + elapsed
        } else {
            Duration::ZERO
        };
        let grow_need = lag.mul_f64(cfg.grow_persistence);
        let shrink_need = lag.mul_f64(cfg.shrink_persistence);

        if grow_cond && ctl.grow_acc >= grow_need {
            // Safety first: growing wins over shrinking.
            if grow_short_without_intents {
                tracing::debug!(
                    agent = ?agent.agent(),
                    "sharding: grow cause: sibling half short"
                );
            } else {
                tracing::debug!(
                    agent = ?agent.agent(),
                    "sharding: grow cause: announced vacates only"
                );
            }
            self.grow(agent, ctl, &grow_cov, live_intents, lag, now)
                .await;
        } else if shrink_cond && ctl.shrink_acc >= shrink_need {
            // Patient lowest: hold off while a lower id is racing us for
            // the same sectors. The accumulator is kept, so we announce
            // as soon as it clears.
            if self.patient() {
                let (vs, ve) = vacate_half(ctl.home, ctl.declared_level);
                if lower_racer(live_intents, agent.agent(), (vs, ve))
                    && !(cfg.announce_alongside
                        && room_alongside(
                            &shrink_cov,
                            live_intents,
                            agent.agent(),
                            (vs, ve),
                            cfg.target_redundancy,
                        ))
                {
                    return;
                }
            }
            // Tests can dictate exactly when each agent may announce.
            #[cfg(any(test, feature = "test-utils"))]
            if !view_delay::announce_allowed(agent.agent(), now) {
                return;
            }
            self.announce_shrink(agent, ctl, lag, now).await;
        }
    }

    /// Widen the target arc by one level; the gossip sync machinery grows
    /// the declared arc to match as sectors verify.
    fn start_grow(&self, agent: &DynLocalAgent, ctl: &mut AgentControl) {
        ctl.target_level = ctl.declared_level + 1;
        ctl.grow_acc = Duration::ZERO;
        ctl.shrink_acc = Duration::ZERO;
        ctl.pending = None;
        ctl.pending_grow = None;
        ctl.syncing_grow = None;
        self.intents.remove(agent.agent());
        self.grow_intents.remove(agent.agent());
        agent.set_tgt_storage_arc_hint(block_arc(ctl.home, ctl.target_level));
        tracing::debug!(
            agent = ?agent.agent(),
            level = ctl.target_level,
            "sharding: growing target arc"
        );
    }

    /// The grow decision. V6: grow at once. V7 (`coordinated_growth`):
    /// announce the range and re-check after a wait, unless a sector of it
    /// is down to one copy or none, which is repaired at once.
    async fn grow(
        &self,
        agent: &DynLocalAgent,
        ctl: &mut AgentControl,
        cov: &[u32],
        live_intents: &[(AgentId, ShrinkIntent)],
        lag: Duration,
        now: Timestamp,
    ) {
        let cfg = &self.config;
        if !cfg.coordinated_growth {
            return self.start_grow(agent, ctl);
        }
        let range = sibling_half(ctl.home, ctl.declared_level);
        if cfg.critical_repair
            && is_critical(&grow_view(cov, live_intents, agent.agent()), range)
        {
            tracing::debug!(
                agent = ?agent.agent(),
                "sharding: critical repair, growing at once"
            );
            return self.start_grow(agent, ctl);
        }
        let wait = self.wait_for(lag);
        let (execute_at, expires_at) = intent_times(now, wait);
        let DhtArc::Arc(..) = super::blocks::sectors_to_arc(range.0, range.1)
        else {
            return self.start_grow(agent, ctl);
        };
        ctl.pending_grow = Some(PendingShrink {
            execute_at,
            vacate: range,
            expires_at,
            wait,
        });
        ctl.grow_acc = Duration::ZERO;
        tracing::debug!(
            agent = ?agent.agent(),
            ?range,
            ?wait,
            "sharding: announcing grow intent"
        );
        self.publish_grow_intent(agent, range, expires_at).await;
    }

    /// Record a grow intent for `range` and broadcast it.
    async fn publish_grow_intent(
        &self,
        agent: &DynLocalAgent,
        range: (u32, u32),
        expires_at: Timestamp,
    ) {
        let DhtArc::Arc(start, end) =
            super::blocks::sectors_to_arc(range.0, range.1)
        else {
            return;
        };
        self.grow_intents.insert(
            agent.agent().clone(),
            ShrinkIntent {
                vacate: range,
                expires_at,
                from: None,
            },
        );
        self.send_intent(agent, (start, end), expires_at, false, true)
            .await;
    }

    /// After `start_grow` from an announced grow: keep the intent up
    /// (locally now, re-broadcast before it lapses) until the sync is done.
    fn keep_grow_announced(
        &self,
        agent: &DynLocalAgent,
        ctl: &mut AgentControl,
        pending: PendingShrink,
    ) {
        self.grow_intents.insert(
            agent.agent().clone(),
            ShrinkIntent {
                vacate: pending.vacate,
                expires_at: pending.expires_at,
                from: None,
            },
        );
        ctl.syncing_grow = Some(pending);
    }

    /// Re-check an announced grow: take the range only if it is still
    /// short counting every lower-id announced grower as already there.
    async fn execute_grow(
        &self,
        agent: &DynLocalAgent,
        ctl: &mut AgentControl,
        cov: &[u32],
        live_intents: &[(AgentId, ShrinkIntent)],
        live_grows: &[(AgentId, ShrinkIntent)],
        now: Timestamp,
    ) {
        let Some(mut pending) = ctl.pending_grow.take() else {
            return;
        };
        let range = pending.vacate;
        match grow_recheck(
            cov,
            live_intents,
            live_grows,
            agent.agent(),
            range,
            self.config.target_redundancy,
        ) {
            Recheck::Proceed => {
                tracing::debug!(
                    agent = ?agent.agent(),
                    "sharding: executed coordinated grow"
                );
                self.start_grow(agent, ctl);
                self.keep_grow_announced(agent, ctl, pending);
            }
            Recheck::Wait => {
                // The lowest of several growers: keep the intent up and
                // look again next interval, as the higher ones stand down.
                pending.execute_at = now + self.config.check_interval();
                if let Some(expires_at) = refreshed_expiry(
                    pending.expires_at,
                    pending.execute_at,
                    pending.wait,
                ) {
                    pending.expires_at = expires_at;
                    self.publish_grow_intent(agent, range, expires_at).await;
                }
                tracing::debug!(
                    agent = ?agent.agent(),
                    "sharding: lowest grower waiting at re-check"
                );
                ctl.pending_grow = Some(pending);
            }
            Recheck::StandDown => {
                self.grow_intents.remove(agent.agent());
                ctl.grow_acc = Duration::ZERO;
                tracing::debug!(
                    agent = ?agent.agent(),
                    "sharding: grow intent stood down, already covered"
                );
                if let DhtArc::Arc(start, end) =
                    super::blocks::sectors_to_arc(range.0, range.1)
                {
                    self.send_intent(
                        agent,
                        (start, end),
                        pending.expires_at,
                        true,
                        true,
                    )
                    .await;
                }
            }
        }
    }

    /// Phase one of a shrink: announce the intent and start the clock.
    async fn announce_shrink(
        &self,
        agent: &DynLocalAgent,
        ctl: &mut AgentControl,
        lag: Duration,
        now: Timestamp,
    ) {
        let cfg = &self.config;
        let wait = self.wait_for(lag);
        // Keep the intent visible for a grace period past execution so
        // growers stay conservative while the shrink propagates.
        let (execute_at, expires_at) = intent_times(now, wait);

        let vacate = vacate_half(ctl.home, ctl.declared_level);
        if !matches!(
            super::blocks::sectors_to_arc(vacate.0, vacate.1),
            DhtArc::Arc(..)
        ) {
            return;
        }

        ctl.pending = Some(PendingShrink {
            execute_at,
            vacate,
            expires_at,
            wait,
        });

        if cfg.agentinfo_encoding {
            // The announcement IS the arc claim: publish the reduced arc
            // now and let the existing AgentInfo gossip carry it. No
            // message is sent, and no intent is recorded — peers will read
            // the narrowed claim and count us out, which is the
            // conservative direction. `declared_level` deliberately still
            // holds the pre-shrink level, so the arc-match guard in
            // `tick_agent` blocks further decisions until this resolves.
            let arc = block_arc(ctl.home, ctl.declared_level - 1);
            agent.set_tgt_storage_arc_hint(arc);
            agent.set_cur_storage_arc(arc);
            agent.invoke_cb();
            tracing::debug!(
                agent = ?agent.agent(),
                ?vacate,
                ?wait,
                "sharding: announcing shrink by narrowing the arc claim"
            );
            return;
        }

        tracing::debug!(
            agent = ?agent.agent(),
            ?vacate,
            ?wait,
            "sharding: announcing shrink intent"
        );
        self.publish_intent(agent, vacate, expires_at).await;
    }

    /// Record a shrink intent locally (sibling agents must see it too) and
    /// broadcast it. Re-publishing replaces the earlier intent at peers.
    async fn publish_intent(
        &self,
        agent: &DynLocalAgent,
        vacate: (u32, u32),
        expires_at: Timestamp,
    ) {
        let DhtArc::Arc(vacate_start, vacate_end) =
            super::blocks::sectors_to_arc(vacate.0, vacate.1)
        else {
            return;
        };
        self.intents.insert(
            agent.agent().clone(),
            ShrinkIntent {
                vacate,
                expires_at,
                from: None,
            },
        );

        self.send_intent(
            agent,
            (vacate_start, vacate_end),
            expires_at,
            false,
            false,
        )
        .await;
    }

    /// Broadcast a withdrawal of this agent's intent with the given
    /// expiry: it stood down and keeps holding the range.
    async fn withdraw_intent(
        &self,
        agent: &DynLocalAgent,
        vacate: (u32, u32),
        expires_at: Timestamp,
    ) {
        let DhtArc::Arc(vacate_start, vacate_end) =
            super::blocks::sectors_to_arc(vacate.0, vacate.1)
        else {
            return;
        };
        self.send_intent(
            agent,
            (vacate_start, vacate_end),
            expires_at,
            true,
            false,
        )
        .await;
    }

    async fn send_intent(
        &self,
        agent: &DynLocalAgent,
        (vacate_start, vacate_end): (u32, u32),
        expires_at: Timestamp,
        withdrawn: bool,
        grow: bool,
    ) {
        let msg = serialize_sharding_message(K2ShardingMessage {
            msg: Some(k2_sharding_message::Msg::ShrinkIntent(
                K2ShardingShrinkIntentMessage {
                    agent_id: agent.agent().0.0.clone(),
                    vacate_start,
                    vacate_end,
                    expires_at_us: expires_at.as_micros(),
                    withdrawn,
                    grow,
                },
            )),
        });
        self.broadcast(msg).await;
    }

    /// Whether the patient-lowest (V6) re-check is in force.
    fn patient(&self) -> bool {
        self.config.patient_lowest && !self.config.agentinfo_encoding
    }

    /// Phase two: re-check with fresher information. Under the default
    /// rule, announced intenders that outrank us (lower agent id) count as
    /// already gone; under patient lowest, every overlapping intender
    /// does, and the lowest racer may wait rather than stand down. If the
    /// check holds, drop the vacated half; otherwise stand down.
    async fn execute_intent(
        &self,
        agent: &DynLocalAgent,
        ctl: &mut AgentControl,
        cov: &[u32],
        live_intents: &[(AgentId, ShrinkIntent)],
        now: Timestamp,
    ) {
        let Some(mut pending) = ctl.pending.take() else {
            return;
        };

        let r = self.config.target_redundancy;
        let proceed = if self.patient() {
            match patient_recheck(
                cov,
                live_intents,
                agent.agent(),
                pending.vacate,
                r,
            ) {
                Recheck::Proceed => true,
                Recheck::StandDown => false,
                Recheck::Wait => {
                    // Keep the intent up and look again next interval.
                    // Peers must keep seeing it for as long as we may
                    // still act on it, so refresh it well before expiry.
                    pending.execute_at = now + self.config.check_interval();
                    if let Some(expires_at) = refreshed_expiry(
                        pending.expires_at,
                        pending.execute_at,
                        pending.wait,
                    ) {
                        pending.expires_at = expires_at;
                        self.publish_intent(
                            agent,
                            pending.vacate,
                            pending.expires_at,
                        )
                        .await;
                    }
                    tracing::debug!(
                        agent = ?agent.agent(),
                        "sharding: lowest racer waiting at re-check"
                    );
                    ctl.pending = Some(pending);
                    return;
                }
            }
        } else {
            shrink_recheck(cov, live_intents, agent.agent(), pending.vacate, r)
        };
        self.intents.remove(agent.agent());

        if proceed {
            ctl.declared_level -= 1;
            ctl.target_level = ctl.declared_level;
            ctl.grow_acc = Duration::ZERO;
            ctl.shrink_acc = Duration::ZERO;
            let arc = block_arc(ctl.home, ctl.declared_level);
            // Under the AgentInfo encoding this arc was already published
            // at announce time; re-publishing would double-count the
            // gossip cost the encoding is being measured for.
            if !self.config.agentinfo_encoding {
                agent.set_tgt_storage_arc_hint(arc);
                agent.set_cur_storage_arc(arc);
                // Re-sign and publish the shrunk declaration.
                agent.invoke_cb();
            }
            tracing::info!(
                agent = ?agent.agent(),
                level = ctl.declared_level,
                "sharding: executed polite shrink"
            );
        } else {
            // Someone who outranks us is going, or the world changed.
            ctl.shrink_acc = Duration::ZERO;
            if self.config.withdraw_on_stand_down
                && !self.config.agentinfo_encoding
            {
                self.withdraw_intent(agent, pending.vacate, pending.expires_at)
                    .await;
            }
            if self.config.agentinfo_encoding {
                // Stand down: re-publish the wider claim we never stopped
                // holding, so peers route to us again.
                let arc = block_arc(ctl.home, ctl.declared_level);
                agent.set_tgt_storage_arc_hint(arc);
                agent.set_cur_storage_arc(arc);
                agent.invoke_cb();
            }
            tracing::debug!(
                agent = ?agent.agent(),
                "sharding: shrink intent cancelled at re-check"
            );
        }
    }

    /// Our view's staleness: how long ago, at the 90th percentile, we
    /// last completed a gossip round with the peers we can see, clamped
    /// to the configured bounds. This is what the hysteresis and the
    /// intent wait scale from.
    /// The wait between announcing and re-checking, for a given lag.
    fn wait_for(&self, lag: Duration) -> Duration {
        std::cmp::max(
            lag.mul_f64(self.config.intent_wait),
            Duration::from_millis(self.config.intent_min_wait_ms as u64),
        )
    }

    /// Update the record of when each peer was last heard from, and
    /// return the peers not heard from within the current wait (or never
    /// observed to send anything). Peers already `gone` are skipped.
    async fn silent_peers(
        &self,
        peers: &[Arc<AgentInfoSigned>],
        gone: &HashSet<AgentId>,
        lag: Duration,
        now: Timestamp,
    ) -> HashSet<AgentId> {
        let lease = self.wait_for(lag);
        // Tests that make the view a fixed time old make contact that old
        // too: nothing in a view `d` old can be fresher than `d`.
        #[cfg(any(test, feature = "test-utils"))]
        let view_age = view_delay::current();
        #[cfg(not(any(test, feature = "test-utils")))]
        let view_age = Duration::ZERO;

        let mut counts = Vec::new();
        for peer in peers {
            if peer.is_tombstone || gone.contains(&peer.agent) {
                continue;
            }
            let Some(url) = peer.url.clone() else {
                continue;
            };
            let n = self
                .peer_meta_store
                .completed_rounds(url.clone())
                .await
                .ok()
                .flatten()
                .unwrap_or(0);
            counts.push((peer.agent.clone(), url, n));
        }

        let mut contact = self.contact.lock().expect("poisoned");
        let mut silent = HashSet::new();
        for (agent, url, n) in counts {
            let entry = contact.entry(url).or_insert((n, None));
            if n > entry.0 {
                *entry = (n, Some(now));
            }
            let fresh = entry.1.is_some_and(|heard| {
                let age = (now - heard).unwrap_or_default().max(view_age);
                age < lease
            });
            if !fresh {
                silent.insert(agent);
            }
        }
        silent
    }

    async fn lag_estimate(
        &self,
        peers: &[Arc<AgentInfoSigned>],
        exclude: &HashSet<AgentId>,
        now: Timestamp,
    ) -> Duration {
        let floor = Duration::from_millis(self.config.lag_floor_ms as u64);
        let ceiling = Duration::from_millis(self.config.lag_ceiling_ms as u64);

        let mut staleness = Vec::new();
        for peer in peers {
            if peer.is_tombstone || exclude.contains(&peer.agent) {
                continue;
            }
            let Some(url) = peer.url.clone() else {
                continue;
            };
            if let Ok(Some(ts)) =
                self.peer_meta_store.last_gossip_timestamp(url).await
            {
                staleness.push((now - ts).unwrap_or_default());
            }
        }
        if staleness.is_empty() {
            // No completed rounds yet: assume the worst.
            return ceiling;
        }
        staleness.sort_unstable();
        let idx = ((staleness.len() * 9) / 10).min(staleness.len() - 1);
        staleness[idx].clamp(floor, ceiling)
    }

    /// Probe the peers holding any sector of `halves`, each at most once
    /// per `probe_interval_ms`. A failed send marks the peer unresponsive.
    async fn probe_holders(
        &self,
        peers: &[Arc<AgentInfoSigned>],
        gone: &HashSet<AgentId>,
        halves: &[(u32, u32)],
        now: Timestamp,
    ) {
        if halves.is_empty() {
            return;
        }
        let Some(transport) = self.transport.upgrade() else {
            return;
        };
        let interval =
            Duration::from_millis(self.config.probe_interval_ms as u64);
        let mut due = Vec::new();
        {
            let mut last = self.last_probe.lock().expect("poisoned");
            for peer in peers {
                if peer.is_tombstone || gone.contains(&peer.agent) {
                    continue;
                }
                let Some(url) = peer.url.clone() else {
                    continue;
                };
                let holds = halves.iter().any(|&(s, e)| {
                    (s..e).any(|sector| {
                        let lo = sector * SECTOR_SIZE;
                        peer.storage_arc.contains(lo)
                            && peer.storage_arc.contains(lo + (SECTOR_SIZE - 1))
                    })
                });
                if !holds {
                    continue;
                }
                if !probe_due(last.get(&url), now, interval) {
                    continue;
                }
                last.insert(url.clone(), now);
                due.push(url);
            }
        }
        if due.is_empty() {
            return;
        }
        let msg = serialize_sharding_message(K2ShardingMessage {
            msg: Some(k2_sharding_message::Msg::Probe(
                K2ShardingProbeMessage {},
            )),
        });
        let space_id = self.space_id.clone();
        // Fire-and-forget, as for broadcasts: a send to a dead peer can
        // block for the transport's connect timeout.
        tokio::task::spawn(async move {
            let sends = due.into_iter().map(|url| {
                let transport = transport.clone();
                let space_id = space_id.clone();
                let msg = msg.clone();
                async move {
                    if let Err(e) = transport
                        .send_module(
                            url.clone(),
                            space_id,
                            SHARDING_MOD_NAME.to_string(),
                            msg,
                        )
                        .await
                    {
                        tracing::debug!(?e, %url, "sharding: probe failed");
                    }
                }
            });
            futures::future::join_all(sends).await;
        });
    }

    /// Send a sharding message to every reachable peer.
    ///
    /// Full fan-out is deliberate for now: shrink intents are rare, tiny,
    /// and everyone's grow decisions benefit from seeing them. A targeted
    /// send to peers overlapping the vacated range is the obvious
    /// optimisation once this graduates.
    async fn broadcast(&self, msg: bytes::Bytes) {
        let Some(transport) = self.transport.upgrade() else {
            return;
        };
        let Ok(peers) = self.peer_store.get_all().await else {
            return;
        };
        // Local agents already see the intent via the IntentTable insert
        // in announce_shrink; sending to our own URL is refused by the
        // transport ("Connecting to ourself") and only produces noise.
        let own_ids = match self.local_agent_store.get_all().await {
            Ok(agents) => agents
                .iter()
                .map(|a| a.agent().clone())
                .collect::<HashSet<_>>(),
            Err(_) => HashSet::new(),
        };
        // Fire-and-forget, all sends in parallel, in a detached task. A
        // send to a dead peer blocks for the transport's full connect
        // timeout; awaiting that here — inside the check task — froze the
        // controller for minutes under Wind Tunnel (an announce hitting 6
        // dead peers sequentially = 6 × 60 s with the pending intent's
        // execute time never re-checked). Delivery is best-effort by
        // design: the intent wait is sized to dominate non-delivery, and
        // the announcer's own intent is already in its local table.
        let peer_meta_store = self.peer_meta_store.clone();
        let space_id = self.space_id.clone();
        tokio::task::spawn(async move {
            let mut sends = Vec::new();
            for peer in peers {
                if peer.is_tombstone || own_ids.contains(&peer.agent) {
                    continue;
                }
                let Some(url) = peer.url.clone() else {
                    continue;
                };
                // A peer already marked unresponsive gets skipped: the
                // send would just burn a connect timeout to learn what
                // the meta store already knows.
                if let Ok(Some(_)) =
                    peer_meta_store.get_unresponsive(url.clone()).await
                {
                    continue;
                }
                let transport = transport.clone();
                let space_id = space_id.clone();
                let msg = msg.clone();
                sends.push(async move {
                    if let Err(e) = transport
                        .send_module(
                            url.clone(),
                            space_id,
                            SHARDING_MOD_NAME.to_string(),
                            msg,
                        )
                        .await
                    {
                        tracing::debug!(?e, %url, "sharding: intent send failed");
                    }
                });
            }
            futures::future::join_all(sends).await;
        });
    }
}

/// Phase two of a polite shrink: with coverage as currently seen
/// (excluding self), and this agent's announced vacate range, decide
/// whether the shrink may proceed.
///
/// Announced intenders with a *lower* agent id are counted as already
/// gone; intenders with a higher id are ignored, because they will count
/// *us* as gone and defer if the range cannot spare us both. This
/// asymmetry is what makes concurrent overlapping shrinks safe without
/// serialising them: symmetric deference (everyone yielding to any
/// overlapping intent) livelocks the whole network, while no deference
/// at all lets simultaneous shrinkers orphan a sector.
fn shrink_recheck(
    cov: &[u32],
    live_intents: &[(AgentId, ShrinkIntent)],
    own_id: &AgentId,
    vacate: (u32, u32),
    target_redundancy: u32,
) -> bool {
    let mut eff = cov.to_vec();
    for (id, intent) in live_intents {
        if id < own_id {
            subtract_sectors(&mut eff, intent.vacate.0, intent.vacate.1);
        }
    }
    min_over(&eff, vacate.0, vacate.1) >= target_redundancy
}

/// While the lowest racer waits, its intent must stay live at peers
/// through its next re-check and a grace period after it, as at first
/// announcement (`expires_at = execute_at + wait`); otherwise a peer
/// could stop counting it as gone while it may still leave. Returns the
/// new expiry when a refresh is due, extending by two waits so the
/// re-broadcast happens about once per wait rather than every interval.
fn refreshed_expiry(
    expires_at: Timestamp,
    next_check: Timestamp,
    wait: Duration,
) -> Option<Timestamp> {
    (expires_at < next_check + wait).then(|| next_check + wait + wait)
}

/// Outcome of a patient-lowest re-check.
#[derive(Debug, PartialEq, Eq)]
enum Recheck {
    /// Vacate now.
    Proceed,
    /// Keep the intent up and re-check next interval.
    Wait,
    /// Withdraw the intent.
    StandDown,
}

/// Do half-open sector ranges overlap?
fn overlaps(a: (u32, u32), b: (u32, u32)) -> bool {
    a.0.max(b.0) < a.1.min(b.1)
}

/// Is an intender with a lower id than ours racing us for any sector of
/// `vacate`?
fn lower_racer(
    live_intents: &[(AgentId, ShrinkIntent)],
    own_id: &AgentId,
    vacate: (u32, u32),
) -> bool {
    live_intents
        .iter()
        .any(|(id, i)| id < own_id && overlaps(i.vacate, vacate))
}

/// File a received intent, or apply its withdrawal, in the table it
/// belongs to. Grow intents are kept apart from shrink intents: a grower
/// counts an announced grower as arriving, and an announced vacate as
/// gone.
fn receive_intent(
    intents: &IntentTable,
    grow_intents: &IntentTable,
    peer: Url,
    intent: K2ShardingShrinkIntentMessage,
) {
    let expires_at = Timestamp::from_micros(intent.expires_at_us);
    let table = if intent.grow { grow_intents } else { intents };
    if intent.withdrawn {
        let agent = AgentId::from(intent.agent_id);
        if table.withdraw(agent.clone(), Some(peer), expires_at) {
            if intent.grow {
                tracing::debug!(
                    agent = ?agent,
                    "sharding: applied a grow withdrawal from its sender"
                );
            } else {
                tracing::debug!(
                    agent = ?agent,
                    "sharding: applied a withdrawal from its sender"
                );
            }
        }
        return;
    }
    if expires_at <= Timestamp::now() {
        return;
    }
    // Block halves never wrap the ring; ignore anything else.
    if intent.vacate_end < intent.vacate_start {
        return;
    }
    let vacate = (
        intent.vacate_start / SECTOR_SIZE,
        intent.vacate_end / SECTOR_SIZE + 1,
    );
    // A vacate range is always half of an aligned block, so
    // it can never span more than half the ring; reject
    // implausible claims before they enter the table. (Full
    // containment in the announcer's declared arc is checked
    // at consumption time, where a peer-store snapshot is in
    // hand — see retain_declared.)
    if vacate.1 - vacate.0 > super::blocks::NUM_SECTORS / 2 {
        return;
    }
    table.insert(
        AgentId::from(intent.agent_id),
        ShrinkIntent {
            vacate,
            expires_at,
            from: Some(peer),
        },
    );
}

/// When an announced intent is re-checked, and when it lapses: one wait
/// after the re-check, so peers keep counting it while the outcome
/// propagates.
fn intent_times(now: Timestamp, wait: Duration) -> (Timestamp, Timestamp) {
    let execute_at = now + wait;
    (execute_at, execute_at + wait)
}

/// A range down to one copy or none, which V7 repairs at once instead of
/// announcing and waiting.
fn is_critical(eff: &[u32], range: (u32, u32)) -> bool {
    min_over(eff, range.0, range.1) <= 1
}

/// The V7 grow re-check, the patient-lowest rule read for growth (as
/// `DecideJoin` with `Rule = "patient"` in polite-shrink's StaleBand.tla):
/// every other agent's announced vacate counts as gone and every other
/// announced grower overlapping `range` as already there. Grow if the
/// range is still short; otherwise the lowest of several growers waits
/// while higher ones are racing, and the rest stand down.
fn grow_recheck(
    cov: &[u32],
    live_intents: &[(AgentId, ShrinkIntent)],
    live_grows: &[(AgentId, ShrinkIntent)],
    own_id: &AgentId,
    range: (u32, u32),
    target_redundancy: u32,
) -> Recheck {
    let mut eff = grow_view(cov, live_intents, own_id);
    let (mut lower, mut higher) = (false, false);
    for (id, intent) in live_grows {
        if id == own_id || !overlaps(intent.vacate, range) {
            continue;
        }
        add_sectors(&mut eff, intent.vacate.0, intent.vacate.1);
        lower |= id < own_id;
        higher |= id > own_id;
    }
    if min_over(&eff, range.0, range.1) < target_redundancy {
        Recheck::Proceed
    } else if higher && !lower {
        Recheck::Wait
    } else {
        Recheck::StandDown
    }
}

/// Whether a peer last probed at `last` is due another probe.
fn probe_due(
    last: Option<&Timestamp>,
    now: Timestamp,
    interval: Duration,
) -> bool {
    last.is_none_or(|t| (now - *t).unwrap_or_default() >= interval)
}

/// The coverage a grower judges a range by: what it sees, with every
/// other agent's announced vacate counted as already gone.
fn grow_view(
    cov: &[u32],
    live_intents: &[(AgentId, ShrinkIntent)],
    own_id: &AgentId,
) -> Vec<u32> {
    let mut eff = cov.to_vec();
    for (id, intent) in live_intents {
        if id != own_id {
            subtract_sectors(&mut eff, intent.vacate.0, intent.vacate.1);
        }
    }
    eff
}

/// Add one copy to every sector of `[start, end)`.
fn add_sectors(cov: &mut [u32], start: u32, end: u32) {
    for c in cov
        .iter_mut()
        .skip(start as usize)
        .take((end - start) as usize)
    {
        *c += 1;
    }
}

/// Is there room for us to vacate alongside every lower-id agent racing
/// us for `vacate`? `cov` is the coverage this agent sees without itself;
/// each lower-id racer overlapping the range counts as gone.
fn room_alongside(
    cov: &[u32],
    live_intents: &[(AgentId, ShrinkIntent)],
    own_id: &AgentId,
    vacate: (u32, u32),
    target_redundancy: u32,
) -> bool {
    let mut eff = cov.to_vec();
    for (id, intent) in live_intents {
        if id < own_id && overlaps(intent.vacate, vacate) {
            subtract_sectors(&mut eff, intent.vacate.0, intent.vacate.1);
        }
    }
    min_over(&eff, vacate.0, vacate.1) >= target_redundancy
}

/// The patient-lowest (V6) re-check.
///
/// Every announced intender overlapping our vacate range counts as
/// already gone — including higher ids, which the default rule ignores.
/// No decision then relies on a peer that may itself be leaving, which is
/// what keeps it safe when announcements are staggered against stale
/// views and when a peer dies mid-race; the default rule's reliance on
/// higher-id intenders "counting us and deferring" is exactly what a
/// staggered cascade breaks.
///
/// Counting everyone would let identical intenders cancel each other
/// forever, so the id order decides only who waits: if the check fails,
/// a higher-id intender overlaps and no lower-id one does, we are the
/// lowest racer and wait. The higher racers fail their own checks, stand
/// down, and (by the announce-time yield) stay quiet until we are done.
fn patient_recheck(
    cov: &[u32],
    live_intents: &[(AgentId, ShrinkIntent)],
    own_id: &AgentId,
    vacate: (u32, u32),
    target_redundancy: u32,
) -> Recheck {
    let mut eff = cov.to_vec();
    let mut higher = false;
    for (id, intent) in live_intents {
        if id == own_id || !overlaps(intent.vacate, vacate) {
            continue;
        }
        subtract_sectors(&mut eff, intent.vacate.0, intent.vacate.1);
        higher |= id > own_id;
    }
    if min_over(&eff, vacate.0, vacate.1) >= target_redundancy {
        Recheck::Proceed
    } else if higher && !lower_racer(live_intents, own_id, vacate) {
        Recheck::Wait
    } else {
        Recheck::StandDown
    }
}

/// Add one copy to every sector fully covered by a (sector-aligned)
/// current storage arc.
fn add_cur_arc(cov: &mut [u32], arc: &DhtArc) {
    if matches!(arc, DhtArc::Empty) {
        return;
    }
    for (sector, c) in cov.iter_mut().enumerate() {
        let start = sector as u32 * SECTOR_SIZE;
        let end = start + (SECTOR_SIZE - 1);
        if arc.contains(start) && arc.contains(end) {
            *c += 1;
        }
    }
}

impl TxBaseHandler for K2Sharding {}

impl TxModuleHandler for K2Sharding {
    fn recv_module_msg(
        &self,
        peer: Url,
        space_id: SpaceId,
        module: String,
        data: bytes::Bytes,
    ) -> K2Result<()> {
        if space_id != self.space_id || module != SHARDING_MOD_NAME {
            return Ok(());
        }
        let msg = deserialize_sharding_message(data)?;
        match msg.msg {
            Some(k2_sharding_message::Msg::ShrinkIntent(intent)) => {
                receive_intent(&self.intents, &self.grow_intents, peer, intent);
            }
            Some(k2_sharding_message::Msg::Probe(_)) | None => (),
        }
        Ok(())
    }
}

#[cfg(test)]
mod stale_model;

#[cfg(any(test, feature = "test-utils"))]
pub(crate) mod view_delay {
    //! Test-only artificial staleness, and a test-only announce
    //! schedule for forcing a specific race.
    //!
    //! On the in-memory transport gossip is near instant, so every node
    //! acts on a fresh view and the stale-view shrink race cannot occur.
    //! With a delay set, each controller tick acts on the peer-store and
    //! intent-table snapshot taken `delay` ago (both from the same
    //! instant, so the view is old but self-consistent), with intents
    //! that have since expired dropped. The node's own controller state
    //! and its sibling local agents' arcs stay fresh. Process-wide: every
    //! sharding instance in the process sees the same delay.

    use super::*;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicU64, Ordering};

    static DELAY_MS: AtomicU64 = AtomicU64::new(0);

    /// Set the artificial view delay for every sharding instance in
    /// this process. Zero turns it off.
    pub(crate) fn set(delay: Duration) {
        DELAY_MS.store(delay.as_millis() as u64, Ordering::Relaxed);
    }

    static IGNORE_UNRESPONSIVE: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);

    /// Make every sharding instance in this process ignore unresponsive
    /// marks, as if no death were ever detected.
    pub(crate) fn set_ignore_unresponsive(on: bool) {
        IGNORE_UNRESPONSIVE.store(on, Ordering::Relaxed);
    }

    /// Whether unresponsive marks are being ignored.
    pub(crate) fn ignore_unresponsive() -> bool {
        IGNORE_UNRESPONSIVE.load(Ordering::Relaxed)
    }

    /// The current artificial view delay.
    pub(crate) fn current() -> Duration {
        Duration::from_millis(DELAY_MS.load(Ordering::Relaxed))
    }

    /// Announce schedule: when set, an agent may announce a shrink only
    /// once, and not before its scheduled time; agents not listed never
    /// announce. `None` restores normal behaviour.
    static SCHEDULE: std::sync::Mutex<Option<HashMap<AgentId, Timestamp>>> =
        std::sync::Mutex::new(None);

    /// Set (or with `None`, clear) the announce schedule for every
    /// sharding instance in this process.
    pub(crate) fn set_schedule(schedule: Option<Vec<(AgentId, Timestamp)>>) {
        *SCHEDULE.lock().expect("poisoned") =
            schedule.map(|s| s.into_iter().collect());
    }

    /// Whether `agent` may announce now under the schedule; consumes its
    /// slot when it may.
    pub(crate) fn announce_allowed(agent: &AgentId, now: Timestamp) -> bool {
        let mut lock = SCHEDULE.lock().expect("poisoned");
        let Some(schedule) = lock.as_mut() else {
            return true;
        };
        match schedule.get(agent) {
            Some(at) if *at <= now => {
                schedule.remove(agent);
                true
            }
            _ => false,
        }
    }

    type Intents = Vec<(AgentId, ShrinkIntent)>;
    type Snapshot = (Timestamp, Vec<Arc<AgentInfoSigned>>, Intents, Intents);

    #[derive(Debug, Default)]
    pub(crate) struct ViewDelay {
        history: std::sync::Mutex<VecDeque<Snapshot>>,
    }

    impl ViewDelay {
        pub(crate) fn apply(
            &self,
            now: Timestamp,
            peers: Vec<Arc<AgentInfoSigned>>,
            intents: Intents,
            grows: Intents,
        ) -> (Vec<Arc<AgentInfoSigned>>, Intents, Intents) {
            let delay = Duration::from_millis(DELAY_MS.load(Ordering::Relaxed));
            let mut history = self.history.lock().expect("poisoned");
            if delay.is_zero() {
                history.clear();
                return (peers, intents, grows);
            }
            history.push_back((now, peers, intents, grows));
            let Ok(cutoff) = now - delay else {
                return history
                    .back()
                    .map(|(_, p, i, g)| (p.clone(), i.clone(), g.clone()))
                    .expect("just pushed");
            };
            // Keep the newest snapshot at or before the cutoff at the
            // front; until there is one, the oldest we have is used.
            while history.len() >= 2 && history[1].0 <= cutoff {
                history.pop_front();
            }
            let (_, p, i, g) = &history[0];
            let live = |v: &Intents| -> Intents {
                v.iter()
                    .filter(|(_, intent)| intent.expires_at > now)
                    .cloned()
                    .collect()
            };
            (p.clone(), live(i), live(g))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sharding::blocks::NUM_SECTORS;

    fn agent(b: &'static [u8]) -> AgentId {
        AgentId::from(bytes::Bytes::from_static(b))
    }

    fn intent(vacate: (u32, u32)) -> ShrinkIntent {
        ShrinkIntent {
            vacate,
            expires_at: Timestamp::now() + Duration::from_secs(60),
            from: None,
        }
    }

    /// Two agents concurrently intend to vacate the same range, which
    /// can only spare one of them: each sees coverage-without-self of
    /// exactly R (so the other intender is one of the R). The lower id
    /// must proceed and the higher id must stand down. Symmetric
    /// deference here (both standing down, then re-announcing, forever)
    /// is the livelock the first design of this rule suffered; symmetric
    /// progress would leave the range at R - 1.
    #[test]
    fn contested_range_lowest_id_proceeds_highest_defers() {
        let r = 3;
        let cov = vec![r; NUM_SECTORS as usize];
        let vacate = (0, 8);
        let low = agent(b"aaa");
        let high = agent(b"zzz");
        let intents = vec![
            (low.clone(), intent(vacate)),
            (high.clone(), intent(vacate)),
        ];

        // The lower id ignores the higher intender: R >= R, proceed.
        assert!(shrink_recheck(&cov, &intents, &low, vacate, r));
        // The higher id counts the lower as gone: R - 1 < R, stand down.
        assert!(!shrink_recheck(&cov, &intents, &high, vacate, r));
    }

    /// When the range is rich enough to spare both intenders, both may
    /// proceed: each decider's coverage-without-self already includes
    /// the other intender, so R + 1 without me, minus the lower-id
    /// intender, still clears R. Concurrent shrinks are not serialised,
    /// only made safe.
    #[test]
    fn rich_range_lets_both_intenders_proceed() {
        let r = 3;
        let cov = vec![r + 1; NUM_SECTORS as usize];
        let vacate = (0, 8);
        let low = agent(b"aaa");
        let high = agent(b"zzz");
        let intents = vec![
            (low.clone(), intent(vacate)),
            (high.clone(), intent(vacate)),
        ];

        assert!(shrink_recheck(&cov, &intents, &low, vacate, r));
        assert!(shrink_recheck(&cov, &intents, &high, vacate, r));
    }

    /// An agent's own intent record (it is in the table while pending)
    /// must not count against itself.
    #[test]
    fn own_intent_is_not_double_counted() {
        let r = 3;
        let cov = vec![r; NUM_SECTORS as usize];
        let me = agent(b"mmm");
        let intents = vec![(me.clone(), intent((0, 8)))];
        assert!(shrink_recheck(&cov, &intents, &me, (0, 8), r));
    }

    /// Non-overlapping intents from lower ids in other parts of the ring
    /// must not block a shrink.
    #[test]
    fn disjoint_intents_do_not_interfere() {
        let r = 3;
        let cov = vec![r; NUM_SECTORS as usize];
        let low = agent(b"aaa");
        let me = agent(b"mmm");
        let intents = vec![(low, intent((100, 108)))];
        assert!(shrink_recheck(&cov, &intents, &me, (0, 8), r));
    }

    /// A lower-id intent overlapping only part of the vacate range still
    /// blocks if the overlap would fall below target.
    #[test]
    fn partial_overlap_blocks_when_tight() {
        let r = 3;
        let cov = vec![r; NUM_SECTORS as usize];
        let low = agent(b"aaa");
        let me = agent(b"mmm");
        // Overlaps sectors 4..8 of my 0..8 vacate range.
        let intents = vec![(low, intent((4, 12)))];
        assert!(!shrink_recheck(&cov, &intents, &me, (0, 8), r));
    }

    /// The staggered cascade that breaks the default rule, reduced to one
    /// decision: the lowest racer sees a higher-id intender still holding
    /// and the default rule lets it go counting that intender as staying.
    /// If the higher one then leaves too (it was the lowest in *its* stale
    /// view), the range drops to R - 1. Patient lowest counts the higher
    /// intender as gone and does not go.
    #[test]
    fn patient_counts_higher_intenders_as_gone() {
        let r = 3;
        let cov = vec![r; NUM_SECTORS as usize];
        let vacate = (0, 8);
        let low = agent(b"aaa");
        let high = agent(b"zzz");
        let intents = vec![
            (low.clone(), intent(vacate)),
            (high.clone(), intent(vacate)),
        ];
        assert!(shrink_recheck(&cov, &intents, &low, vacate, r));
        assert_ne!(
            patient_recheck(&cov, &intents, &low, vacate, r),
            Recheck::Proceed
        );
    }

    /// In the contested range the lowest racer waits instead of
    /// cancelling (cancelling everyone is the stall that counting every
    /// intender alone suffers), and the higher racer stands down.
    #[test]
    fn patient_lowest_waits_highest_stands_down() {
        let r = 3;
        let cov = vec![r; NUM_SECTORS as usize];
        let vacate = (0, 8);
        let low = agent(b"aaa");
        let high = agent(b"zzz");
        let intents = vec![
            (low.clone(), intent(vacate)),
            (high.clone(), intent(vacate)),
        ];
        assert_eq!(
            patient_recheck(&cov, &intents, &low, vacate, r),
            Recheck::Wait
        );
        assert_eq!(
            patient_recheck(&cov, &intents, &high, vacate, r),
            Recheck::StandDown
        );
    }

    /// Once the higher racer's intent has cleared, the waiting lowest
    /// racer re-checks with nobody left to discount and proceeds.
    #[test]
    fn patient_lowest_proceeds_once_racers_clear() {
        let r = 3;
        let cov = vec![r; NUM_SECTORS as usize];
        let vacate = (0, 8);
        let low = agent(b"aaa");
        let intents = vec![(low.clone(), intent(vacate))];
        assert_eq!(
            patient_recheck(&cov, &intents, &low, vacate, r),
            Recheck::Proceed
        );
    }

    /// A middle racer — lower ids and higher ids both overlapping —
    /// stands down rather than waiting: only the lowest may wait.
    #[test]
    fn patient_middle_racer_stands_down() {
        let r = 3;
        let cov = vec![r; NUM_SECTORS as usize];
        let vacate = (0, 8);
        let intents = vec![
            (agent(b"aaa"), intent(vacate)),
            (agent(b"mmm"), intent(vacate)),
            (agent(b"zzz"), intent(vacate)),
        ];
        assert_eq!(
            patient_recheck(&cov, &intents, &agent(b"mmm"), vacate, r),
            Recheck::StandDown
        );
    }

    /// With a wide margin every racer clears the count at once: patient
    /// lowest stays concurrent and does not serialise shrinks.
    #[test]
    fn patient_rich_range_lets_all_racers_proceed() {
        let r = 3;
        let cov = vec![r + 2; NUM_SECTORS as usize];
        let vacate = (0, 8);
        let ids = [agent(b"aaa"), agent(b"mmm"), agent(b"zzz")];
        let intents: Vec<_> =
            ids.iter().map(|a| (a.clone(), intent(vacate))).collect();
        for me in &ids {
            assert_eq!(
                patient_recheck(&cov, &intents, me, vacate, r),
                Recheck::Proceed
            );
        }
    }

    /// No racers but too little coverage (say a peer died): stand down,
    /// do not wait.
    #[test]
    fn patient_short_range_without_racers_stands_down() {
        let r = 3;
        let cov = vec![r - 1; NUM_SECTORS as usize];
        let me = agent(b"mmm");
        let intents = vec![(me.clone(), intent((0, 8)))];
        assert_eq!(
            patient_recheck(&cov, &intents, &me, (0, 8), r),
            Recheck::StandDown
        );
    }

    /// Announcing alongside lower ids: allowed only while the range keeps
    /// the target counting us and every lower-id racer as gone. Higher-id
    /// racers and disjoint ones do not count.
    #[test]
    fn room_alongside_counts_lower_racers_only() {
        let r = 3;
        let me = agent(b"mmm");
        // 5 copies besides us: room for us and one lower racer (3 left)...
        let cov = vec![5; NUM_SECTORS as usize];
        let one = vec![(agent(b"aaa"), intent((0, 8)))];
        assert!(room_alongside(&cov, &one, &me, (0, 8), r));
        // ... but not for us and three lower racers (1 left).
        let three = vec![
            (agent(b"aaa"), intent((0, 8))),
            (agent(b"bbb"), intent((0, 8))),
            (agent(b"ccc"), intent((4, 12))),
        ];
        assert!(!room_alongside(&cov, &three, &me, (0, 8), r));
        // Higher-id and disjoint racers are not counted: three higher
        // racers would leave 2 if they were.
        let others = vec![
            (agent(b"zzz"), intent((0, 8))),
            (agent(b"yyy"), intent((0, 8))),
            (agent(b"xxx"), intent((0, 8))),
            (agent(b"aaa"), intent((100, 108))),
        ];
        assert!(room_alongside(&cov, &others, &me, (0, 8), r));
        // With exactly the target besides us, no lower racer fits.
        let thin = vec![r; NUM_SECTORS as usize];
        assert!(!room_alongside(&thin, &one, &me, (0, 8), r));
    }

    /// The grow re-check (patient lowest, for growth): grow while the range
    /// is short counting every other announced grower as there; when too
    /// many are racing, the lowest waits and the others stand down.
    #[test]
    fn grow_recheck_lets_only_the_lowest_grower_take_a_short_range() {
        let r = 3;
        let me = agent(b"mmm");
        let short = vec![r - 1; NUM_SECTORS as usize];
        let check = |cov: &[u32], vac: &[(AgentId, ShrinkIntent)], grows| {
            grow_recheck(cov, vac, grows, &me, (0, 8), r)
        };
        assert_eq!(check(&short, &[], &[]), Recheck::Proceed);
        // One copy short and a lower id growing too: stand down.
        let lower = vec![(agent(b"aaa"), intent((0, 8)))];
        assert_eq!(check(&short, &[], &lower), Recheck::StandDown);
        // One copy short and only higher ids growing: the lowest waits.
        let higher = vec![(agent(b"zzz"), intent((0, 8)))];
        assert_eq!(check(&short, &[], &higher), Recheck::Wait);
        // A middle grower stands down.
        let both = vec![
            (agent(b"aaa"), intent((0, 8))),
            (agent(b"zzz"), intent((0, 8))),
        ];
        assert_eq!(check(&short, &[], &both), Recheck::StandDown);
        let disjoint = vec![(agent(b"aaa"), intent((8, 16)))];
        assert_eq!(check(&short, &[], &disjoint), Recheck::Proceed);
        // Two copies short: two growers both go, a third does not.
        let two_short = vec![r - 2; NUM_SECTORS as usize];
        assert_eq!(check(&two_short, &[], &higher), Recheck::Proceed);
        assert_eq!(check(&two_short, &[], &lower), Recheck::Proceed);
        assert_eq!(check(&two_short, &[], &both), Recheck::StandDown);
        // At the target there is nothing to repair...
        let full = vec![r; NUM_SECTORS as usize];
        assert_eq!(check(&full, &[], &[]), Recheck::StandDown);
        // ... unless another agent has announced it is leaving.
        let leaving = vec![(agent(b"bbb"), intent((0, 8)))];
        assert_eq!(check(&full, &leaving, &[]), Recheck::Proceed);
        // Our own announced vacate is not counted against us.
        let own = vec![(me.clone(), intent((0, 8)))];
        assert_eq!(check(&full, &own, &[]), Recheck::StandDown);
    }

    /// Critical repair: one copy or none left, never two.
    #[test]
    fn critical_means_one_copy_or_none() {
        let mut cov = vec![2; NUM_SECTORS as usize];
        assert!(!is_critical(&cov, (0, 8)));
        cov[5] = 1;
        assert!(is_critical(&cov, (0, 8)));
        assert!(!is_critical(&cov, (8, 16)));
        cov[5] = 0;
        assert!(is_critical(&cov, (0, 8)));
    }

    /// Intents, grow or shrink, are re-checked one wait after announcing
    /// and stay live one more wait after that.
    #[test]
    fn intents_lapse_one_wait_after_the_recheck() {
        let now = Timestamp::now();
        let wait = Duration::from_secs(3);
        let (execute_at, expires_at) = intent_times(now, wait);
        assert_eq!(execute_at, now + wait);
        assert_eq!(expires_at, now + wait + wait);
    }

    /// A peer is probed again only once the interval has passed.
    #[test]
    fn probes_respect_the_interval() {
        let now = Timestamp::now();
        let interval = Duration::from_secs(10);
        assert!(probe_due(None, now, interval));
        assert!(!probe_due(Some(&now), now, interval));
        let recent = (now - Duration::from_secs(5)).unwrap();
        assert!(!probe_due(Some(&recent), now, interval));
        let old = (now - interval).unwrap();
        assert!(probe_due(Some(&old), now, interval));
    }

    /// Received grow intents, and their withdrawals, go to the grow table
    /// and never to the shrink table, and the reverse.
    #[test]
    fn received_intents_are_filed_by_kind() {
        let peer = Url::from_str("ws://127.0.0.1:80").unwrap();
        let now = Timestamp::now();
        let expires = now + Duration::from_secs(60);
        let msg = |id: &'static [u8], grow, withdrawn| {
            K2ShardingShrinkIntentMessage {
                agent_id: bytes::Bytes::from_static(id),
                vacate_start: 0,
                vacate_end: 8 * SECTOR_SIZE - 1,
                expires_at_us: expires.as_micros(),
                withdrawn,
                grow,
            }
        };
        let (shrinks, grows) = (IntentTable::default(), IntentTable::default());
        let ids = |t: &IntentTable| {
            t.live(now).into_iter().map(|(a, _)| a).collect::<Vec<_>>()
        };
        receive_intent(
            &shrinks,
            &grows,
            peer.clone(),
            msg(b"ggg", true, false),
        );
        receive_intent(
            &shrinks,
            &grows,
            peer.clone(),
            msg(b"sss", false, false),
        );
        assert_eq!(ids(&grows), vec![agent(b"ggg")]);
        assert_eq!(ids(&shrinks), vec![agent(b"sss")]);
        assert_eq!(grows.live(now)[0].1.vacate, (0, 8));
        // A grow withdrawal leaves the shrink table alone.
        receive_intent(&shrinks, &grows, peer.clone(), msg(b"sss", true, true));
        assert_eq!(ids(&shrinks), vec![agent(b"sss")]);
        receive_intent(&shrinks, &grows, peer.clone(), msg(b"ggg", true, true));
        assert!(ids(&grows).is_empty());
        receive_intent(&shrinks, &grows, peer, msg(b"sss", false, true));
        assert!(ids(&shrinks).is_empty());
    }

    /// Ranges that only touch neither overlap nor block a re-check.
    #[test]
    fn adjacent_ranges_do_not_overlap() {
        // Half-open ranges that only touch share no sector.
        assert!(!overlaps((0, 8), (8, 16)));
        assert!(!overlaps((8, 16), (0, 8)));
        assert!(overlaps((0, 9), (8, 16)));
        let r = 3;
        let cov = vec![r; NUM_SECTORS as usize];
        let me = agent(b"mmm");
        let intents = vec![(agent(b"aaa"), intent((8, 16)))];
        assert_eq!(
            patient_recheck(&cov, &intents, &me, (0, 8), r),
            Recheck::Proceed
        );
        assert!(!lower_racer(&intents, &me, (0, 8)));
    }

    /// Disjoint intents, from either side of us in id order, neither
    /// count nor make us wait.
    #[test]
    fn patient_ignores_disjoint_intents() {
        let r = 3;
        let cov = vec![r; NUM_SECTORS as usize];
        let me = agent(b"mmm");
        let intents = vec![
            (agent(b"aaa"), intent((100, 108))),
            (agent(b"zzz"), intent((200, 208))),
        ];
        assert_eq!(
            patient_recheck(&cov, &intents, &me, (0, 8), r),
            Recheck::Proceed
        );
        assert!(!lower_racer(&intents, &me, (0, 8)));
    }

    /// However long the lowest racer waits, its intent never lapses at
    /// peers before its next re-check plus a grace period, and it is
    /// re-broadcast about once per wait, not once per check interval.
    #[test]
    fn waiting_intent_stays_live_and_refreshes_sparingly() {
        for (interval_ms, wait_ms) in
            [(100, 1_000), (5_000, 10_000), (900, 1_000), (3_000, 1_000)]
        {
            let interval = Duration::from_millis(interval_ms);
            let wait = Duration::from_millis(wait_ms);
            let start = Timestamp::now();
            let mut now = start + wait;
            let mut expires_at = now + wait;
            let mut refreshes = 0u64;
            let steps = 1_000u64;
            for _ in 0..steps {
                let next_check = now + interval;
                if let Some(e) = refreshed_expiry(expires_at, next_check, wait)
                {
                    expires_at = e;
                    refreshes += 1;
                }
                assert!(
                    expires_at >= next_check + wait,
                    "intent would lapse (interval {interval_ms} ms,                      wait {wait_ms} ms)"
                );
                now = next_check;
            }
            // At most one refresh per wait of waiting; when the interval
            // is longer than the wait, one per re-check.
            let waited_ms = steps * interval_ms;
            let bound = if interval_ms <= wait_ms {
                waited_ms / wait_ms + 1
            } else {
                steps
            };
            assert!(
                refreshes <= bound,
                "{refreshes} refreshes over {waited_ms} ms of waiting"
            );
        }
    }

    /// The announce-time yield: a lower-id intent overlapping the half we
    /// would vacate holds us back; a higher-id one, or our own, does not.
    #[test]
    fn yield_only_to_overlapping_lower_ids() {
        let me = agent(b"mmm");
        let vacate = (0, 8);
        assert!(lower_racer(
            &[(agent(b"aaa"), intent((4, 12)))],
            &me,
            vacate
        ));
        assert!(!lower_racer(
            &[(agent(b"zzz"), intent(vacate))],
            &me,
            vacate
        ));
        assert!(!lower_racer(&[(me.clone(), intent(vacate))], &me, vacate));
    }
}
