//! Configuration parameters for the sharding module.

/// Configuration parameters for K2Sharding.
///
/// This will be set as a default by the
/// [K2GossipFactory](crate::K2GossipFactory) when the `sharding` feature
/// is enabled.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct K2ShardingConfig {
    /// The desired number of copies of every DHT sector.
    ///
    /// The controller grows an agent's target arc while any sector it
    /// could take on has fewer declared holders than this, and only
    /// considers shrinking when every sector it would drop has strictly
    /// more than this many holders besides itself.
    ///
    /// Default: 5
    #[cfg_attr(feature = "schema", schemars(default))]
    pub target_redundancy: u32,

    /// Below this many visible (non-tombstoned) peers, agents keep a full
    /// arc and never shrink.
    ///
    /// Small networks gain nothing from sharding and are the most fragile
    /// under it, so the controller only engages once the network is large
    /// enough to be worth it.
    ///
    /// Default: 25
    #[cfg_attr(feature = "schema", schemars(default))]
    pub clamp_min_peers: u32,

    /// The interval in milliseconds between controller decision checks.
    ///
    /// This is the decision-epoch period: hysteresis accumulates in units
    /// of this interval. It should be comfortably smaller than the typical
    /// gossip staleness, or the hysteresis quantisation gets coarse.
    ///
    /// Default: 5,000 (5s)
    #[cfg_attr(feature = "schema", schemars(default))]
    pub check_interval_ms: u32,

    /// Grow persistence factor: the grow condition must hold continuously
    /// for this multiple of the measured gossip staleness before the
    /// target arc is widened.
    ///
    /// Default: 1.0
    #[cfg_attr(feature = "schema", schemars(default))]
    pub grow_persistence: f64,

    /// Shrink persistence factor: the shrink condition must hold
    /// continuously for this multiple of the measured gossip staleness
    /// before a shrink intent is announced.
    ///
    /// Shrinking must be much more reluctant than growing: an unnecessary
    /// grow wastes bandwidth, an unnecessary shrink risks data loss.
    ///
    /// Default: 4.0
    #[cfg_attr(feature = "schema", schemars(default))]
    pub shrink_persistence: f64,

    /// After announcing a shrink intent, wait this multiple of the
    /// measured gossip staleness before re-checking and executing.
    ///
    /// This must be long enough that every peer whose decision could
    /// interact with ours has seen the announcement.
    ///
    /// Default: 2.5
    #[cfg_attr(feature = "schema", schemars(default))]
    pub intent_wait: f64,

    /// A lower bound in milliseconds on the shrink-intent wait, applied
    /// regardless of how small the measured staleness is.
    ///
    /// Default: 10,000 (10s)
    #[cfg_attr(feature = "schema", schemars(default))]
    pub intent_min_wait_ms: u32,

    /// Lower clamp in milliseconds on the measured gossip staleness used
    /// to scale the hysteresis.
    ///
    /// Default: 1,000 (1s)
    #[cfg_attr(feature = "schema", schemars(default))]
    pub lag_floor_ms: u32,

    /// Upper clamp in milliseconds on the measured gossip staleness used
    /// to scale the hysteresis. Also used as the assumed staleness while
    /// no gossip round has completed yet.
    ///
    /// Default: 300,000 (5m)
    #[cfg_attr(feature = "schema", schemars(default))]
    pub lag_ceiling_ms: u32,

    /// Carry the vacate intent on the gossiped `AgentInfo` arc claim
    /// instead of a dedicated `ShrinkIntent` message.
    ///
    /// When true, announcing a shrink means publishing the *reduced* arc
    /// immediately and re-checking against what peers still declare; no
    /// sharding message is sent or consumed. Peers cannot then distinguish
    /// "announced but still holding" from "already gone", so the lower-id
    /// tie-break has no input and is given up: the re-check simply
    /// requires the target redundancy among current claimants. That is
    /// safe (model-checked) but strictly more conservative — it stands
    /// down more often and settles at a wider arc.
    ///
    /// Note that in kitsune2 the declared arc is also what a peer is asked
    /// to serve, so during the wait this makes the vacated half
    /// unreachable even though the ops are still held locally.
    ///
    /// Default: false
    #[cfg_attr(feature = "schema", schemars(default))]
    pub agentinfo_encoding: bool,

    /// Use the "patient lowest" re-check (V6) rather than the lower-id
    /// tie-break (V3).
    ///
    /// At re-check, *every* overlapping announced intender counts as
    /// already gone, so no decision ever relies on a peer that may itself
    /// be leaving; the tie-break alone can be broken by a staggered
    /// cascade of announcements under stale views. Counting everyone
    /// would stall identical intenders, so ordering by id decides only
    /// *when* to act, never what to count: if the re-check fails, some
    /// higher-id intender overlaps and no lower-id one does, this agent
    /// keeps its intent up and re-checks next interval instead of standing
    /// down; and no agent announces while a lower-id intent overlaps the
    /// half it would vacate.
    ///
    /// Needs the explicit intent channel, so it is ignored when
    /// `agentinfo_encoding` is set: a narrowed arc claim cannot tell an
    /// intender from a peer that has already left.
    ///
    /// This is the recommended rule. Set to false for the original
    /// lower-id tie-break (V3), kept for comparison.
    ///
    /// Default: true
    #[cfg_attr(feature = "schema", schemars(default))]
    pub patient_lowest: bool,

    /// When an agent stands down at re-check, broadcast a withdrawal of
    /// its intent so peers drop it at once instead of when it expires.
    ///
    /// Without it, a withdrawn intent keeps counting as an announced
    /// vacate at peers until it expires, and growers treat announced
    /// vacates as already gone: they grow to cover departures that will
    /// never happen. The withdrawal names the exact intent (agent and
    /// expiry) and is only honoured from the agent's own peer URL, so a
    /// late or forged one cannot remove a newer or genuine intent.
    ///
    /// Safe because an agent that stood down really does keep holding
    /// the range, so peers counting it as present again are right. This
    /// also brings the code in line with the model checks, where a
    /// stood-down intent is gone from the next view.
    ///
    /// Default: true
    #[cfg_attr(feature = "schema", schemars(default))]
    pub withdraw_on_stand_down: bool,

    /// For shrink decisions, count a remote holder only if this node has
    /// heard from it within the current wait.
    ///
    /// "Heard from" means a gossip message arrived from the peer (the
    /// per-peer completed-rounds counter went up), in either direction.
    /// kitsune2's last-gossip timestamp cannot be used for this: it is set
    /// when a round is *attempted*, even if the peer never answers.
    ///
    /// This turns two of the rule's timing assumptions from trusted into
    /// enforced, for holders: the information a shrink relies on about a
    /// holder is never older than the wait, and a holder that has gone
    /// silent (a death that has not been detected yet) stops counting
    /// within one wait of its last message, however long the transport
    /// takes to mark it unresponsive. When either assumption would be
    /// violated the agent simply does not shrink: a cost in progress, not
    /// in safety. Grow decisions are unaffected and still count every
    /// holder not marked unresponsive.
    ///
    /// Default: true
    #[cfg_attr(feature = "schema", schemars(default))]
    pub require_recent_contact: bool,

    /// Let an agent announce a shrink alongside lower-id agents racing it
    /// for the same range, when there is room for all of them.
    ///
    /// Without it, an agent never announces while a lower id is announcing
    /// for an overlapping range, so agents sharing a range shrink one at a
    /// time. From full arcs every agent's first shrink drops one of the
    /// same two halves, and sharding down can then be several times slower
    /// than it needs to be (polite-shrink `reports/REPORT_v7.md`). With it,
    /// an agent holds off only if, counting itself and every lower-id
    /// racer as gone, its half would keep fewer than `target_redundancy`
    /// copies. The re-check is unchanged, so this cannot affect safety.
    ///
    /// Default: true
    #[cfg_attr(feature = "schema", schemars(default))]
    pub announce_alongside: bool,

    /// Coordinate growing as well as shrinking (V7).
    ///
    /// Without it, every agent that sees a short sector grows on its own,
    /// so one gap can draw several growers. With it, an agent announces
    /// the range it means to take, waits as for a shrink, and re-checks:
    /// it grows only if the range is still short counting every lower-id
    /// agent that has announced growing into it as already there;
    /// otherwise it stands down. Growing below the small-network clamp,
    /// and a critical repair (see `critical_repair`), happen at once.
    ///
    /// Default: false (V7 is under test)
    #[cfg_attr(feature = "schema", schemars(default))]
    pub coordinated_growth: bool,

    /// With `coordinated_growth`, grow at once, without announcing, when a
    /// sector of the range is down to one copy or none.
    ///
    /// Default: true
    #[cfg_attr(feature = "schema", schemars(default))]
    pub critical_repair: bool,

    /// Keep the full arc and never shrink.
    ///
    /// For a machine that should hold everything: a server, or a site's
    /// main machine that must keep the whole network available to that
    /// site when it is cut off from the rest. Other agents count it like
    /// any holder.
    ///
    /// Default: false
    #[cfg_attr(feature = "schema", schemars(default))]
    pub pinned: bool,
}

impl Default for K2ShardingConfig {
    fn default() -> Self {
        Self {
            target_redundancy: 5,
            clamp_min_peers: 25,
            check_interval_ms: 5_000,
            grow_persistence: 1.0,
            shrink_persistence: 4.0,
            intent_wait: 2.5,
            intent_min_wait_ms: 10_000,
            lag_floor_ms: 1_000,
            lag_ceiling_ms: 300_000,
            agentinfo_encoding: false,
            patient_lowest: true,
            withdraw_on_stand_down: true,
            require_recent_contact: true,
            announce_alongside: true,
            coordinated_growth: false,
            critical_repair: true,
            pinned: false,
        }
    }
}

impl K2ShardingConfig {
    /// The interval between controller decision checks.
    pub fn check_interval(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.check_interval_ms as u64)
    }
}

/// Module-level configuration for K2Sharding.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct K2ShardingModConfig {
    /// K2Sharding configuration.
    pub k2_sharding: K2ShardingConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped default is the recommended rule: patient lowest (V6)
    /// with withdrawal, and the explicit intent channel it needs.
    #[test]
    fn default_is_patient_lowest_with_withdrawal() {
        let c = K2ShardingConfig::default();
        assert!(c.patient_lowest);
        assert!(c.withdraw_on_stand_down);
        assert!(c.require_recent_contact);
        assert!(c.announce_alongside);
        assert!(!c.coordinated_growth);
        assert!(c.critical_repair);
        assert!(!c.pinned);
        assert!(!c.agentinfo_encoding);
    }
}
