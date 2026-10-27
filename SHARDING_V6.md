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

**What V6 changes.** When several nodes want to drop the same part of the
DHT at once, V3 lets the lowest-ID one go ahead on the assumption that the
others stay. V6 counts every node that has announced an intent to shrink as
already gone. If that leaves too few holders only because of higher-ID
announcers, the lowest one keeps its intent and waits instead of giving up.

**What the tests check.** Three properties, named separately below:

- *at target*: every sector keeps at least the redundancy target of holders.
- *never orphaned*: every sector keeps at least one holder. This is weaker. It
  is used where a test deliberately kills nodes to push sectors below target.
- *settled*: no live V6 node is left holding a shrink it could still make.

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
| `cargo test -p kitsune2_gossip --features sharding --lib sharding` | unit tests, plus an exhaustive model check that drives the real re-check code through every announce timing, view lag and death (about 3 s) | V6 stays at target in every schedule within the check's bounds (small networks; see results §1, and the large run below: 4 nodes, 2 copies, wait 2.5× the view lag), assuming deaths are noticed within the wait; V3 fails the schedules the Python checker says it fails, with matching counts |
| `cargo test -p kitsune2_gossip --features sharding --test sharding_cascade` | the staggered cascade, forced: four nodes, 2 s old views, descending-ID announcements (about 2 min) | **V3 loses every copy of a half** (the test expects that), V6 keeps one, and two controls behave as predicted |
| `cargo test -p kitsune2_gossip --features sharding --test sharding_stale` | every node acting on 2 s old views (about 1.5 min) | V6 never takes any part of the DHT below target |
| `cargo test -p kitsune2_gossip --features sharding --test sharding_race` | a simultaneous race with a thin margin (about 1 min) | V6's lowest racer waits, then leaves; nothing goes below target; nobody is left able to shrink |
| `cargo test -p kitsune2_gossip --features sharding --test sharding_iroh` | the storm over the real iroh transport (QUIC, local relay), 8 nodes (about 1.5 min) | the network shards down, a quarter of it dies at once, and no part of the DHT is ever left with no holder |
| `cargo test -p kitsune2_gossip --features sharding --test sharding_upgrade` | a real rollout: V6 nodes alongside nodes that never shrink (as nodes without dynamic sharding behave); six such nodes replaced by V6 nodes one at a time; V6 nodes with two different sets of timing settings (about 2.5 min) | no sector ever drops below target with no deaths, the fixed nodes never change, no V6 agent is left able to shrink, and the rollout never leaves part of the DHT with no holder and ends back at target |
| `CHAOS_SECS=3600 cargo test -p kitsune2_gossip --features sharding --test sharding_chaos -- --ignored --nocapture` | an hour of random kills, joins and staleness changes (0–1.5 s, wait 2 s), 7–14 nodes; `CHAOS_SEED` picks the sequence; the control is `SHARDING_RECENT_CONTACT=0 CHAOS_WAIT_MS=300` | no shrink ever takes a sector below target and no sector is left without a holder (outside 5 s after a kill, which is the window before a death can be noticed), and at the end everything returns to target. The control is expected to **fail**. (With recent contact on, a 300 ms wait cannot fail: every peer looks silent and nobody shrinks) |
| `cargo test -p kitsune2_gossip --features sharding --test sharding_alongside` | 16 nodes shard down from full arcs, with and without `announce_alongside` (about 3.5 min) | with it, more nodes announce at once; nothing goes below target with no deaths; nobody is left able to shrink |
| `cargo test -p kitsune2_gossip --features sharding --test sharding_v7` | 3 of 12 sharded nodes killed, regrowth watched with coordinated growth off and on; and a pinned node (about 5 min) | every short range some node could grow into is repaired; every announced grow is executed or stood down; the pinned node keeps everything |
| `cargo test -p kitsune2_gossip --features sharding --test sharding_storm` | 8 nodes shard down, 3 are killed at once, the rest recover (about 20 s) | no part of the DHT is ever left with no holder, under V6 and V3 |

Set `SHARDING_COORDINATED_GROWTH=1` to run any of these with V7's coordinated
growth on, and `SHARDING_PROBE=0` to turn the probes off.

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
- `require_recent_contact` (default `true`): for shrink decisions, count a
  holder only if this node has heard from it (a gossip message arrived)
  within the current wait. A holder that has gone silent, including one
  that died without being detected, stops counting within one wait. Set
  `SHARDING_RECENT_CONTACT=0` to run the tests without it.
- `announce_alongside` (default `true`): a node may announce alongside
  lower-id nodes racing it for the same half when, counting itself and all of
  them as gone, the half still keeps the target. Fixes V6's slow start from
  full arcs, where nodes otherwise queue and shrink one at a time. The re-check
  is unchanged.
- `probe_sibling_holders` (default `true`, every `probe_interval_ms`, 10 s):
  send an empty message to each holder of the half this node would grow into.
  kitsune2 notices a death only when a send fails, and sharded nodes rarely
  message that half, so without probes a death there could go unnoticed until
  the dead node's agent info expired, up to about 20 minutes.
- `coordinated_growth` (default `false`): V7. Announce a grow, wait, and grow
  only if the range is still short counting lower-id announced growers as
  already there, so one gap does not draw a herd. `critical_repair` (default
  `true`) makes a range down to one copy or none grow at once. Off by default
  until a larger test shows its storage saving on the real code; see
  [REPORT_v7](https://github.com/ValiChord/polite-shrink/blob/main/reports/REPORT_v7.md).
  Run it with a target one copy above the minimum.
- `pinned` (default `false`): this node keeps everything and never shrinks.
  One in a group that may be cut off gives that group all the data.
- `agentinfo_encoding` (default `false`): an older experiment that carries the
  intent on the AgentInfo arc claim. ⚠️ **Setting this turns V6 off**:
  `patient_lowest` is ignored and the node falls back to V3, because a
  narrowed arc claim cannot tell an intender from a peer that has already
  left. Leave it `false` to test V6.

Three test-only hooks, compiled only with `cfg(test)` or the `test-utils`
feature, are in `crates/gossip/src/harness.rs`:
`set_sharding_view_delay` (make nodes act on old views),
`set_sharding_announce_schedule` (dictate who announces when) and
`set_sharding_ignore_unresponsive` (simulate deaths that are never detected;
`CHAOS_UNDETECTED=1` in the chaos soak).

## Results and limits

What these tests found, with raw data:
[results/rust_fork_v6.md](https://github.com/ValiChord/polite-shrink/blob/main/results/rust_fork_v6.md);
for the V7 settings,
[REPORT_v7](https://github.com/ValiChord/polite-shrink/blob/main/reports/REPORT_v7.md#v7-on-the-kitsune2-code).

One machine, the in-memory transport, a handful of nodes. No Holochain
conductor, no wide-area network. Real-transport (iroh) measurements of the
older V3 code are in polite-shrink's `wind_tunnel/`. They have not been
repeated for V6.

Covered in the results file but not in this test suite:

- V6 over iroh at 8, 16, 24 and 32 nodes (§8.4).
- Nodes that lie about what they store (§8.3, in the Python simulation).
- Waits shorter than the real staleness (§8.2).

Network partitions, where two isolated groups each judge it safe to shrink,
are tested in TLA+ and in the Python simulation (REPORT_v7), not in this suite.
