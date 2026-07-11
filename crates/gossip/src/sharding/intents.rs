//! Table of announced shrink intents.
//!
//! Holds both remote intents received on the `k2sharding` channel and
//! this node's own local agents' pending intents (local agents must see
//! each other's announcements the same way remote peers do). Intents
//! expire at the timestamp their announcer attached; an executed shrink
//! shows up as the shrunk arc in the peer store well before that, and an
//! abandoned one simply ages out.

use kitsune2_api::{AgentId, Timestamp};
use std::collections::HashMap;
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
