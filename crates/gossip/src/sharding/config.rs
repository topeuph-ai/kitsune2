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
