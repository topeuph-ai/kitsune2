//! Table of announced shrink intents.
//!
//! Holds both remote intents received on the `k2sharding` channel and
//! this node's own local agents' pending intents (local agents must see
//! each other's announcements the same way remote peers do). Intents
//! expire at the timestamp their announcer attached; an executed shrink
//! shows up as the shrunk arc in the peer store well before that, and an
//! abandoned one simply ages out.

use super::blocks::NUM_SECTORS;
use kitsune2_api::{AgentId, DhtArc, Timestamp};
use kitsune2_dht::SECTOR_SIZE;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

/// One announced shrink intent.
#[derive(Debug, Clone)]
pub(crate) struct ShrinkIntent {
    /// Half-open sector range being vacated.
    pub vacate: (u32, u32),
    /// When this intent is void.
    pub expires_at: Timestamp,
}

/// Shared, expiring table of announced shrink intents, keyed by intender.
///
/// A later announcement from the same agent replaces the earlier one; an
/// agent can only have one shrink in flight.
#[derive(Debug, Default)]
pub(crate) struct IntentTable {
    inner: Mutex<HashMap<AgentId, ShrinkIntent>>,
}

impl IntentTable {
    pub fn insert(&self, agent: AgentId, intent: ShrinkIntent) {
        self.inner.lock().expect("poisoned").insert(agent, intent);
    }

    pub fn remove(&self, agent: &AgentId) {
        self.inner.lock().expect("poisoned").remove(agent);
    }

    /// Drop expired intents and return the live ones.
    pub fn live(&self, now: Timestamp) -> Vec<(AgentId, ShrinkIntent)> {
        let mut lock = self.inner.lock().expect("poisoned");
        lock.retain(|_, i| i.expires_at > now);
        lock.iter().map(|(a, i)| (a.clone(), i.clone())).collect()
    }
}

/// Receiver-side range-validation: keep only intents whose vacate range
/// lies inside the announcer's currently *declared* arc, as this node
/// sees it in its peer store. Local agents bypass the check — their
/// intents are made here, not received, and their declared arc may
/// briefly lag their own executed shrink.
///
/// Why: intents are consumed in the coverage-lowering direction
/// (growers subtract announced vacates; executing shrinkers count
/// higher-priority intenders as gone), so a forged intent can talk
/// declared coverage down over sectors its claimed sender never held.
/// That cannot orphan data, but it is a cheap resource lever — the
/// polite-shrink defense study measured a whole-ring forgery pinning
/// arcs wide at 2.3x sync cost, and this filter reducing the residual
/// to ~4%. The filter also drops stale intents whose announcer already
/// executed the shrink (the vacated range has left their declared arc),
/// which would otherwise be double-counted against coverage.
///
/// An intent from an agent absent from the peer store cannot be
/// validated and is dropped.
pub(crate) fn retain_declared(
    live: &mut Vec<(AgentId, ShrinkIntent)>,
    declared: &HashMap<AgentId, DhtArc>,
    own_ids: &HashSet<AgentId>,
) {
    live.retain(|(agent, intent)| {
        own_ids.contains(agent)
            || declared
                .get(agent)
                .is_some_and(|arc| arc_contains_sectors(arc, intent.vacate))
    });
}

/// True if every sector in the half-open range `[start, end)` lies fully
/// within `arc`. Handles wrapping arcs by checking per sector, the same
/// containment rule the coverage counters use.
fn arc_contains_sectors(arc: &DhtArc, (start, end): (u32, u32)) -> bool {
    if start >= end || end > NUM_SECTORS {
        return false;
    }
    (start..end).all(|sector| {
        let lo = sector * SECTOR_SIZE;
        arc.contains(lo) && arc.contains(lo + (SECTOR_SIZE - 1))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn agent(b: &'static [u8]) -> AgentId {
        AgentId::from(bytes::Bytes::from_static(b))
    }

    #[test]
    fn expired_intents_age_out() {
        let table = IntentTable::default();
        let now = Timestamp::now();
        table.insert(
            agent(b"a"),
            ShrinkIntent {
                vacate: (0, 8),
                expires_at: now + Duration::from_secs(10),
            },
        );
        table.insert(
            agent(b"b"),
            ShrinkIntent {
                vacate: (8, 16),
                expires_at: (now - Duration::from_secs(1)).unwrap(),
            },
        );

        let live = table.live(now);
        assert_eq!(1, live.len());
        assert_eq!(agent(b"a"), live[0].0);
    }

    fn intent(vacate: (u32, u32)) -> ShrinkIntent {
        ShrinkIntent {
            vacate,
            expires_at: Timestamp::now() + Duration::from_secs(60),
        }
    }

    #[test]
    fn range_validation_drops_forged_stale_and_unknown_intents() {
        let sector = |s: u32| s * SECTOR_SIZE;
        // "small" declares sectors [0, 16); "ghost" is not in the store.
        let declared: HashMap<AgentId, DhtArc> =
            [(agent(b"small"), DhtArc::Arc(0, sector(16) - 1))].into();
        let own: HashSet<AgentId> = [agent(b"me")].into();

        let mut live = vec![
            // inside the declared arc: kept
            (agent(b"small"), intent((8, 16))),
            // whole-ring forgery in small's name: dropped
            (agent(b"small"), intent((0, NUM_SECTORS))),
            // outside the declared arc (forged, or stale after an
            // executed shrink): dropped
            (agent(b"small"), intent((16, 24))),
            // announcer unknown to the peer store: dropped
            (agent(b"ghost"), intent((0, 8))),
            // own agents bypass validation
            (agent(b"me"), intent((0, 8))),
        ];
        retain_declared(&mut live, &declared, &own);

        let kept: Vec<_> =
            live.iter().map(|(a, i)| (a.clone(), i.vacate)).collect();
        assert_eq!(
            vec![(agent(b"small"), (8, 16)), (agent(b"me"), (0, 8))],
            kept
        );
    }

    #[test]
    fn range_validation_handles_wrapping_arcs() {
        let sector = |s: u32| s * SECTOR_SIZE;
        // wraps: sectors [500, 512) and [0, 4)
        let declared: HashMap<AgentId, DhtArc> =
            [(agent(b"w"), DhtArc::Arc(sector(500), sector(4) - 1))].into();
        let own = HashSet::new();

        let mut live = vec![
            (agent(b"w"), intent((500, 512))), // fully inside: kept
            (agent(b"w"), intent((2, 6))),     // straddles the gap: dropped
        ];
        retain_declared(&mut live, &declared, &own);
        assert_eq!(1, live.len());
        assert_eq!((500, 512), live[0].1.vacate);
    }

    #[test]
    fn reannouncement_replaces() {
        let table = IntentTable::default();
        let now = Timestamp::now();
        for vacate in [(0, 8), (8, 16)] {
            table.insert(
                agent(b"a"),
                ShrinkIntent {
                    vacate,
                    expires_at: now + Duration::from_secs(10),
                },
            );
        }
        let live = table.live(now);
        assert_eq!(1, live.len());
        assert_eq!((8, 16), live[0].1.vacate);
    }
}
