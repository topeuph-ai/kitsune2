//! The sharding controller: decides target arcs and coordinates shrinks.

use super::blocks::{
    MAX_LEVEL, block_arc, home_sector, sibling_half, vacate_half,
};
use super::coverage::{min_over, others_coverage, subtract_sectors};
use super::intents::{IntentTable, ShrinkIntent, retain_declared};
use super::protocol::{
    K2ShardingMessage, K2ShardingShrinkIntentMessage,
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
}

#[derive(Debug)]
struct PendingShrink {
    /// When to re-check and execute.
    execute_at: Timestamp,
    /// Half-open sector range that would be vacated.
    vacate: (u32, u32),
}

/// The sharding module implementation.
///
/// Runs one controller tick per configured interval over every local
/// agent, and receives shrink-intent announcements from peers as a
/// transport module handler. See the [module docs](super) for the
/// decision rules.
#[derive(Debug)]
pub struct K2Sharding {
    config: Arc<K2ShardingConfig>,
    space_id: SpaceId,
    peer_store: DynPeerStore,
    local_agent_store: DynLocalAgentStore,
    peer_meta_store: Arc<K2PeerMetaStore>,
    transport: WeakDynTransport,
    intents: IntentTable,
    states: Mutex<HashMap<AgentId, AgentControl>>,
    /// Peers seen as gone (unresponsive or tombstoned) last tick, to
    /// detect fresh departures.
    last_gone: std::sync::Mutex<HashSet<AgentId>>,
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
            states: Mutex::new(HashMap::new()),
            last_gone: std::sync::Mutex::new(HashSet::new()),
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

        let mut live_intents = self.intents.live(now);
        // Range-validation: only count an intent over sectors its
        // announcer actually declares (see retain_declared). Uses the
        // same peer snapshot as this tick's coverage, so intent and arc
        // are judged on one consistent view.
        let declared_arcs: HashMap<AgentId, DhtArc> = peers
            .iter()
            .filter(|p| !p.is_tombstone)
            .map(|p| (p.agent.clone(), p.storage_arc))
            .collect();
        retain_declared(&mut live_intents, &declared_arcs, &own_ids);
        let lag = self.lag_estimate(&peers, &gone, now).await;

        let mut states = self.states.lock().await;

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
                    if ctl.pending.take().is_some() {
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
        for agent in &local_agents {
            // The declared redundancy this agent sees around itself:
            // remote peers plus sibling local agents, excluding itself.
            let mut cov = base_cov.clone();
            for other in &local_agents {
                if other.agent() != agent.agent() {
                    add_cur_arc(&mut cov, &other.get_cur_storage_arc());
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
                    }
                });

            self.tick_agent(
                agent,
                ctl,
                cov,
                &live_intents,
                visible_peers,
                lag,
                now,
                elapsed,
            )
            .await;
        }

        // Forget agents that have left the space.
        states.retain(|id, _| own_ids.contains(id));

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
        cov: Vec<u32>,
        live_intents: &[(AgentId, ShrinkIntent)],
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
                self.execute_intent(agent, ctl, &cov, live_intents);
            }
            return;
        }

        // The declared arc must have caught up with the model before any
        // decision: a growing (or freshly joined, still empty) arc means
        // the verified sync is in flight, and deciding from an arc we do
        // not actually hold could shrink us onto data we never synced.
        if agent.get_cur_storage_arc() != block_arc(ctl.home, ctl.target_level)
        {
            return;
        }
        ctl.declared_level = ctl.target_level;

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
        if ctl.declared_level < MAX_LEVEL {
            let mut eff = cov.clone();
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
        }

        // Shrink check: does the half we would drop stay strictly above
        // the target without us?
        let mut shrink_cond = false;
        if ctl.declared_level > 0 {
            let (vs, ve) = vacate_half(ctl.home, ctl.declared_level);
            shrink_cond = min_over(&cov, vs, ve) > cfg.target_redundancy;
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
            self.start_grow(agent, ctl);
        } else if shrink_cond && ctl.shrink_acc >= shrink_need {
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
        self.intents.remove(agent.agent());
        agent.set_tgt_storage_arc_hint(block_arc(ctl.home, ctl.target_level));
        tracing::debug!(
            agent = ?agent.agent(),
            level = ctl.target_level,
            "sharding: growing target arc"
        );
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
        let wait = std::cmp::max(
            lag.mul_f64(cfg.intent_wait),
            Duration::from_millis(cfg.intent_min_wait_ms as u64),
        );
        let execute_at = now + wait;
        // Keep the intent visible for a grace period past execution so
        // growers stay conservative while the shrink propagates.
        let expires_at = execute_at + wait;

        let vacate = vacate_half(ctl.home, ctl.declared_level);
        let vacate_arc = super::blocks::sectors_to_arc(vacate.0, vacate.1);
        let DhtArc::Arc(vacate_start, vacate_end) = vacate_arc else {
            return;
        };

        ctl.pending = Some(PendingShrink { execute_at, vacate });

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

        // Local sibling agents must see this intent too.
        self.intents
            .insert(agent.agent().clone(), ShrinkIntent { vacate, expires_at });

        tracing::debug!(
            agent = ?agent.agent(),
            ?vacate,
            ?wait,
            "sharding: announcing shrink intent"
        );

        let msg = serialize_sharding_message(K2ShardingMessage {
            msg: Some(k2_sharding_message::Msg::ShrinkIntent(
                K2ShardingShrinkIntentMessage {
                    agent_id: agent.agent().0.0.clone(),
                    vacate_start,
                    vacate_end,
                    expires_at_us: expires_at.as_micros(),
                },
            )),
        });
        self.broadcast(msg).await;
    }

    /// Phase two: re-check with fresher information, counting announced
    /// intenders that outrank us (lower agent id) as already gone. If it
    /// still holds, drop the vacated half; otherwise stand down.
    fn execute_intent(
        &self,
        agent: &DynLocalAgent,
        ctl: &mut AgentControl,
        cov: &[u32],
        live_intents: &[(AgentId, ShrinkIntent)],
    ) {
        let Some(pending) = ctl.pending.take() else {
            return;
        };
        self.intents.remove(agent.agent());

        if shrink_recheck(
            cov,
            live_intents,
            agent.agent(),
            pending.vacate,
            self.config.target_redundancy,
        ) {
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
        _peer: Url,
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
                let expires_at = Timestamp::from_micros(intent.expires_at_us);
                if expires_at <= Timestamp::now() {
                    return Ok(());
                }
                // Block halves never wrap the ring; ignore anything else.
                if intent.vacate_end < intent.vacate_start {
                    return Ok(());
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
                    return Ok(());
                }
                self.intents.insert(
                    AgentId::from(intent.agent_id),
                    ShrinkIntent { vacate, expires_at },
                );
            }
            None => (),
        }
        Ok(())
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
}
