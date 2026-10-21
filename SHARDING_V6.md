# Testing dynamic sharding: polite shrink, V6 "patient lowest"

This branch (`feat/sharding-v6`) adds dynamic storage-arc sizing to
kitsune2's gossip crate, behind the existing `sharding` feature flag. It is
research for [holochain/kitsune2#160](https://github.com/holochain/kitsune2/issues/160),
not an upstream pull request. Design, simulations and model checks live at
[ValiChord/polite-shrink](https://github.com/ValiChord/polite-shrink).

**What to test:** the rule that decides when a node may drop part of its
storage arc without leaving any part of the DHT below the redundancy target.
The recommended rule is **V6, "patient lowest"**, and it is the default here.
The original rule, V3 (lowest-ID tie-break), is kept for comparison.

## Setup

You need git and [rustup](https://rustup.rs). The pinned toolchain (Rust
1.99.0, see `rust-toolchain.toml`) installs itself on first build.

```
git clone -b feat/sharding-v6 https://github.com/topeuph-ai/kitsune2.git
cd kitsune2
```

## Run the tests

All run on one machine; all but `sharding_iroh` use the in-memory transport. The first build takes a
few minutes.

| command | what it checks | a pass means |
|---|---|---|
| `cargo test -p kitsune2_gossip --features sharding --lib sharding` | unit tests, plus an exhaustive model check that drives the real re-check code through every announce timing, view lag and death (about 3 s) | V6 is safe in every schedule; V3 fails the schedules the Python checker says it fails, with matching counts |
| `cargo test -p kitsune2_gossip --features sharding --test sharding_cascade` | the staggered cascade, forced: four nodes, 2 s old views, descending-ID announcements (about 2 min) | **V3 loses every copy of a half** (the test expects that), V6 keeps one, and two controls behave as predicted |
| `cargo test -p kitsune2_gossip --features sharding --test sharding_stale` | every node acting on 2 s old views (about 1.5 min) | V6 never takes any part of the DHT below target |
| `cargo test -p kitsune2_gossip --features sharding --test sharding_race` | a simultaneous race with a thin margin (about 1 min) | V6's lowest racer waits, then leaves; nothing goes below target; nobody is left able to shrink |
| `cargo test -p kitsune2_gossip --features sharding --test sharding_iroh` | the storm over the real iroh transport (QUIC, local relay), 8 nodes (about 1.5 min) | the network shards down, a quarter of it dies at once, and no part of the DHT is ever left with no holder |
| `cargo test -p kitsune2_gossip --features sharding --test sharding_storm` | 8 nodes shard down, 3 are killed at once, the rest recover (about 20 s) | no part of the DHT is ever left with no holder, under V6 and V3 |

Slower comparisons, which print CSV rather than assert, are marked
`#[ignore]`. Run them with `-- --ignored --nocapture`. Each test file's header
says what its comparison measures. The 14-million-schedule model check is one
of them:
`cargo test --release -p kitsune2_gossip --features sharding --lib stale_model -- --ignored`.

## Configuration

`K2ShardingConfig` (`crates/gossip/src/sharding/config.rs`):

- `patient_lowest` (default `true`): V6. Set to `false` for V3.
- `withdraw_on_stand_down` (default `true`): a node that backs off withdraws
  its intent at once instead of letting it age out. Without it, V3 re-grows
  repeatedly.
- `agentinfo_encoding` (default `false`): an older experiment that carries the
  intent on the AgentInfo arc claim. V6 needs the explicit intent channel, so
  it is ignored when this is set.

Two test-only hooks, compiled only with `cfg(test)` or the `test-utils`
feature, are in `crates/gossip/src/harness.rs`:
`set_sharding_view_delay` (make nodes act on old views) and
`set_sharding_announce_schedule` (dictate who announces when).

## Results and limits

What these tests found, with raw data:
[results/rust_fork_v6.md](https://github.com/ValiChord/polite-shrink/blob/main/results/rust_fork_v6.md).

One machine, the in-memory transport, a handful of nodes. No Holochain
conductor, no wide-area network. Real-transport (iroh) measurements of the
older V3 code are in polite-shrink's `wind_tunnel/`. They have not been
repeated for V6.
