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
//!    this agent gone. Which announced intenders count as gone depends
//!    on the rule (below).
//!
//! **Re-check rule.** The default is the *patient-lowest* rule (V6,
//! [`K2ShardingConfig::patient_lowest`]), the recommended rule as of
//! October 2026: every overlapping announced intender counts as already
//! gone; if that fails only because of higher-id intenders, the lowest
//! racer keeps its intent up and re-checks next interval instead of
//! standing down; and nobody announces while a lower-id intent overlaps
//! the half they would vacate. Setting `patient_lowest: false` restores
//! the original lower-id tie-break (V3), which counts only lower-id
//! intenders as gone. V3 is kept for comparison: under stale views a
//! staggered cascade of announcements can break it (see below).
//!
//! Growers treat announced vacates as already gone, so a grower moves to
//! fill a hole before it exists rather than after.
//!
//! Controller dynamics (hysteresis scaled to measured gossip staleness,
//! grow readily / shrink reluctantly, the two-phase handoff) were
//! selected by simulation across disruption scenarios and an adversarial
//! search. The controller here ports those decision rules one-to-one;
//! sim ticks map to wall-clock via the measured staleness.
//!
//! **What is and is not established.** The research lives at
//! <https://github.com/ValiChord/polite-shrink>. Its TLA+ model gave the
//! re-check an atomic, fresh view, which never exercises the tie-break.
//! Exhaustive timed model checking with *stale* views
//! (`spec/timed_check.py`, `REPORT_stale_views.md`, small networks,
//! N <= 5) finds that V3 can drop a sector below target, and that V6 is
//! safe with a wait of at least one gossip lag, provided deaths are
//! noticed before anyone acts on a view still counting the dead peer,
//! while still making progress and shrinking concurrently. No local rule
//! can see a death that has not been detected yet. These are simulation
//! and small-model results, not a proof for real networks.

mod blocks;
mod config;
mod controller;
mod coverage;
mod intents;
pub(crate) mod protocol;

pub use config::*;
pub use controller::K2Sharding;

#[cfg(any(test, feature = "test-utils"))]
pub(crate) use controller::view_delay::{
    set as view_delay_set, set_schedule as announce_schedule_set,
};

/// The module name for use when registering as a module handler.
pub const SHARDING_MOD_NAME: &str = "k2sharding";
