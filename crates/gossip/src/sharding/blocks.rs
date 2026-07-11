//! Quantised storage-arc block math.
//!
//! An agent's arc is always an aligned power-of-two block of sectors
//! containing its home sector. The only controller state is the block
//! *level*: level `L` is a block of `2^L` sectors, and level
//! [`MAX_LEVEL`] is the full ring. Quantisation keeps the decision space
//! tiny and makes overlap between agents' blocks either total or none at
//! each level, which is what the shrink tie-break relies on.
//!
//! All ranges here are half-open `[start, end)` in sector indices.
//! Conversion to the inclusive-bound [`DhtArc`] happens only at the edge.

use kitsune2_api::DhtArc;
use kitsune2_dht::SECTOR_SIZE;

/// The number of sectors in the full ring.
pub(crate) const NUM_SECTORS: u32 = u32::MAX / SECTOR_SIZE + 1;

/// The block level that covers the full ring: `2^MAX_LEVEL == NUM_SECTORS`.
pub(crate) const MAX_LEVEL: u8 = NUM_SECTORS.trailing_zeros() as u8;

/// The sector containing the given DHT location.
pub(crate) fn home_sector(loc: u32) -> u32 {
    loc / SECTOR_SIZE
}

/// The aligned block `[start, end)` of `2^level` sectors containing `home`.
pub(crate) fn block(home: u32, level: u8) -> (u32, u32) {
    if level >= MAX_LEVEL {
        return (0, NUM_SECTORS);
    }
    let size = 1u32 << level;
    let start = (home >> level) << level;
    (start, start + size)
}

/// The sectors gained by growing `level -> level + 1`: the sibling half of
/// the parent block. Contiguous by construction.
pub(crate) fn sibling_half(home: u32, level: u8) -> (u32, u32) {
    let (parent_start, parent_end) = block(home, level + 1);
    let (cur_start, cur_end) = block(home, level);
    if parent_start == cur_start {
        (cur_end, parent_end)
    } else {
        (parent_start, cur_start)
    }
}

/// The sectors dropped by shrinking `level -> level - 1`: the half of the
/// current block not containing `home`.
pub(crate) fn vacate_half(home: u32, level: u8) -> (u32, u32) {
    let (cur_start, cur_end) = block(home, level);
    let (keep_start, keep_end) = block(home, level - 1);
    if cur_start == keep_start {
        (keep_end, cur_end)
    } else {
        (cur_start, keep_start)
    }
}

/// Convert a half-open sector range to a [`DhtArc`] with inclusive
/// location bounds.
pub(crate) fn sectors_to_arc(start: u32, end: u32) -> DhtArc {
    debug_assert!(start < end && end <= NUM_SECTORS);
    if start == 0 && end == NUM_SECTORS {
        return DhtArc::FULL;
    }
    DhtArc::Arc(
        start * SECTOR_SIZE,
        (end - 1) * SECTOR_SIZE + (SECTOR_SIZE - 1),
    )
}

/// The [`DhtArc`] for an agent's block at the given level.
pub(crate) fn block_arc(home: u32, level: u8) -> DhtArc {
    let (start, end) = block(home, level);
    sectors_to_arc(start, end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_constants() {
        // The DHT model quantises the ring into 512 sectors; the block
        // math relies on that being a power of two.
        assert_eq!(512, NUM_SECTORS);
        assert_eq!(9, MAX_LEVEL);
        assert!(NUM_SECTORS.is_power_of_two());
    }

    #[test]
    fn block_fixtures() {
        // level 0 is just the home sector
        assert_eq!((37, 38), block(37, 0));
        // blocks are aligned, not centred on home
        assert_eq!((36, 38), block(37, 1));
        assert_eq!((32, 40), block(37, 3));
        // max level is the full ring regardless of home
        assert_eq!((0, NUM_SECTORS), block(37, MAX_LEVEL));
        assert_eq!((0, NUM_SECTORS), block(511, MAX_LEVEL + 3));
    }

    #[test]
    fn sibling_is_other_half_of_parent() {
        for home in [0u32, 1, 37, 255, 256, 511] {
            for level in 0..MAX_LEVEL {
                let (cs, ce) = block(home, level);
                let (ss, se) = sibling_half(home, level);
                let (ps, pe) = block(home, level + 1);
                // sibling and current tile the parent exactly
                assert_eq!((pe - ps), (ce - cs) + (se - ss));
                assert!(ss >= ps && se <= pe);
                // and do not overlap
                assert!(se <= cs || ss >= ce);
            }
        }
    }

    #[test]
    fn vacate_is_half_not_containing_home() {
        for home in [0u32, 1, 37, 255, 256, 511] {
            for level in 1..=MAX_LEVEL {
                let (vs, ve) = vacate_half(home, level);
                let (ks, ke) = block(home, level - 1);
                // home stays in the kept half
                assert!(ks <= home && home < ke);
                // vacated half is outside the kept half
                assert!(ve <= ks || vs >= ke);
                // together they tile the current block
                let (cs, ce) = block(home, level);
                assert_eq!(ce - cs, (ke - ks) + (ve - vs));
            }
        }
    }

    #[test]
    fn sectors_to_arc_bounds() {
        assert_eq!(DhtArc::FULL, sectors_to_arc(0, NUM_SECTORS));
        assert_eq!(DhtArc::Arc(0, SECTOR_SIZE - 1), sectors_to_arc(0, 1));
        // the last sector's inclusive end is u32::MAX
        assert_eq!(
            DhtArc::Arc((NUM_SECTORS - 1) * SECTOR_SIZE, u32::MAX),
            sectors_to_arc(NUM_SECTORS - 1, NUM_SECTORS)
        );
    }
}
