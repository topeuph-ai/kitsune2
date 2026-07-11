//! Dynamic storage-arc sizing ("sharding module", issue #160).
//!
//! Decides a target storage arc for each local agent from the observed
//! per-sector redundancy, and coordinates *shrinking* with a two-phase
//! polite handoff so that concurrent shrinks cannot orphan a sector.
//!
//! Arcs are quantised: an agent claims an aligned power-of-two block of
//! sectors containing its home sector (the sector its agent id hashes
//! into). Growing doubles the block, shrinking halves it. Growing is
//! applied via the target-arc hint and completed by the existing
//! verified-sync machinery in [`crate::storage_arc`]; shrinking drops
//! authority directly, but only after the polite handoff:
//!
//! 1. Announce a [`ShrinkIntent`](protocol::K2ShardingShrinkIntentMessage)
//!    on the `k2sharding` module channel.
//! 2. Wait long enough for the announcement to propagate (a multiple of
//!    the measured gossip staleness).
//! 3. Re-check that the vacated range keeps the redundancy target with
//!    this agent gone, counting any *lower agent id* announced intenders
//!    as already gone. The lowest conflicting intender proceeds; the
//!    others observe the conflict and stand down.
//!
//! Growers treat announced vacates as already gone, so a grower moves to
//! fill a hole before it exists rather than after.
//!
//! Controller dynamics (hysteresis scaled to measured gossip staleness,
//! grow readily / shrink reluctantly, the two-phase handoff, and the
//! lowest-id tie-break) were selected by simulation across disruption
//! scenarios and an adversarial search; the two-phase variant showed no
//! sector loss in 1248 multi-seed runs where naive and damped-only
//! controllers lost data. The controller here ports those decision rules
//! one-to-one; sim ticks map to wall-clock via the measured staleness.

mod blocks;
mod config;
mod controller;
mod coverage;
mod intents;
pub(crate) mod protocol;

pub use config::*;
pub use controller::K2Sharding;

/// The module name for use when registering as a module handler.
pub const SHARDING_MOD_NAME: &str = "k2sharding";
