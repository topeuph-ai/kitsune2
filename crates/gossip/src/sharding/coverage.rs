//! Per-sector redundancy counting from declared storage arcs.

use super::blocks::NUM_SECTORS;
use kitsune2_api::{AgentId, AgentInfoSigned, DhtArc};
use kitsune2_dht::SECTOR_SIZE;
use std::collections::HashSet;
use std::sync::Arc;

/// Count, for every sector, how many of the given peers declare a storage
/// arc covering it. Tombstoned peers and the agents in `exclude` (this
/// node's own local agents) are not counted, so the result is the
/// redundancy provided by *others*.
///
/// A sector is only counted as covered when the whole sector is inside
/// the declared arc. Kitsune2 arcs are sector-aligned by construction, so
/// checking both sector bounds is exact rather than conservative.
pub(crate) fn others_coverage(
    peers: &[Arc<AgentInfoSigned>],
    exclude: &HashSet<AgentId>,
) -> Vec<u32> {
    let mut cov = vec![0u32; NUM_SECTORS as usize];
    for peer in peers {
        if peer.is_tombstone || exclude.contains(&peer.agent) {
            continue;
        }
        add_arc(&mut cov, &peer.storage_arc);
    }
    cov
}

/// Add one copy to every sector fully covered by `arc`.
fn add_arc(cov: &mut [u32], arc: &DhtArc) {
    match arc {
        DhtArc::Empty => {}
        DhtArc::Arc(..) => {
            for (sector, c) in cov.iter_mut().enumerate() {
                let start = sector as u32 * SECTOR_SIZE;
                let end = start + (SECTOR_SIZE - 1);
                if arc.contains(start) && arc.contains(end) {
                    *c += 1;
                }
            }
        }
    }
}

/// Saturating-subtract one copy from every sector in the half-open sector
/// range `[start, end)`. Used to discount announced vacates.
pub(crate) fn subtract_sectors(cov: &mut [u32], start: u32, end: u32) {
    for c in cov
        .iter_mut()
        .skip(start as usize)
        .take((end - start) as usize)
    {
        *c = c.saturating_sub(1);
    }
}

/// The minimum coverage over the half-open sector range `[start, end)`.
pub(crate) fn min_over(cov: &[u32], start: u32, end: u32) -> u32 {
    cov[start as usize..end as usize]
        .iter()
        .copied()
        .min()
        .unwrap_or(0)
}
