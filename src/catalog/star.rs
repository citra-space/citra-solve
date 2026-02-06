//! Packed star representation for memory-efficient storage.

use bytemuck::{Pod, Zeroable};
use crate::core::types::{RaDec, CatalogStar};
use std::f64::consts::PI;

/// Packed star entry in the index file.
///
/// Uses fixed-point representation for compact storage:
/// - RA: 32-bit signed, scaled to [-π, π) -> [-2^31, 2^31)
/// - Dec: 32-bit signed, scaled to [-π/2, π/2] -> [-2^31, 2^31)
/// - Magnitude: 16-bit signed, scaled by 100 (e.g., 5.5 mag = 550)
/// - ID: 16-bit catalog index
///
/// Total: 12 bytes per star (vs 28 bytes for f64 representation)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PackedStar {
    /// Right ascension in fixed-point radians.
    pub ra_fixed: i32,
    /// Declination in fixed-point radians.
    pub dec_fixed: i32,
    /// Magnitude * 100 (e.g., 5.5 mag = 550).
    pub mag_x100: i16,
    /// Catalog index (0-65535).
    pub catalog_idx: u16,
}

// Scale factor for RA: maps [0, 2π) to [0, 2^32)
const RA_SCALE: f64 = (1u64 << 32) as f64 / (2.0 * PI);
// Scale factor for Dec: maps [-π/2, π/2] to [-2^31, 2^31)
const DEC_SCALE: f64 = (1u64 << 31) as f64 / (PI / 2.0);

impl PackedStar {
    /// Create a packed star from floating-point values.
    pub fn new(ra: f64, dec: f64, magnitude: f32, catalog_idx: u16) -> Self {
        // Normalize RA to [0, 2π)
        let ra_norm = ra.rem_euclid(2.0 * PI);
        // Convert to fixed-point (unsigned, then cast to signed for storage)
        let ra_fixed = (ra_norm * RA_SCALE) as u32 as i32;
        // Dec is already in [-π/2, π/2]
        let dec_fixed = (dec * DEC_SCALE) as i32;
        let mag_x100 = (magnitude * 100.0) as i16;

        Self {
            ra_fixed,
            dec_fixed,
            mag_x100,
            catalog_idx,
        }
    }

    /// Get right ascension in radians.
    #[inline]
    pub fn ra(&self) -> f64 {
        (self.ra_fixed as u32 as f64) / RA_SCALE
    }

    /// Get declination in radians.
    #[inline]
    pub fn dec(&self) -> f64 {
        (self.dec_fixed as f64) / DEC_SCALE
    }

    /// Get magnitude.
    #[inline]
    pub fn magnitude(&self) -> f32 {
        self.mag_x100 as f32 / 100.0
    }

    /// Get catalog index.
    #[inline]
    pub fn catalog_index(&self) -> u16 {
        self.catalog_idx
    }

    /// Convert to RaDec.
    #[inline]
    pub fn to_radec(&self) -> RaDec {
        RaDec::new(self.ra(), self.dec())
    }

    /// Convert to CatalogStar.
    pub fn to_catalog_star(&self, full_id: u32) -> CatalogStar {
        CatalogStar {
            id: full_id,
            position: self.to_radec(),
            magnitude: self.magnitude(),
        }
    }
}

/// Packed pattern entry in the index file.
///
/// Each pattern references 4 stars and stores 5 edge ratios.
/// - Star indices: 4 x 16-bit = 8 bytes
/// - Edge ratios: 5 x 16-bit (0-65535 maps to 0.0-1.0) = 10 bytes
///
/// Total: 18 bytes per pattern
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PackedPattern {
    /// Indices of the 4 stars forming this pattern.
    pub star_indices: [u16; 4],
    /// Normalized edge ratios (sorted, excluding longest = 1.0).
    pub edge_ratios: [u16; 5],
}

const RATIO_SCALE: f64 = 65535.0;

impl PackedPattern {
    /// Create a packed pattern from star indices and ratios.
    pub fn new(star_indices: [u16; 4], ratios: [f64; 5]) -> Self {
        let edge_ratios = [
            (ratios[0] * RATIO_SCALE) as u16,
            (ratios[1] * RATIO_SCALE) as u16,
            (ratios[2] * RATIO_SCALE) as u16,
            (ratios[3] * RATIO_SCALE) as u16,
            (ratios[4] * RATIO_SCALE) as u16,
        ];
        Self {
            star_indices,
            edge_ratios,
        }
    }

    /// Get edge ratios as f64 values in [0, 1].
    #[inline]
    pub fn ratios(&self) -> [f64; 5] {
        [
            self.edge_ratios[0] as f64 / RATIO_SCALE,
            self.edge_ratios[1] as f64 / RATIO_SCALE,
            self.edge_ratios[2] as f64 / RATIO_SCALE,
            self.edge_ratios[3] as f64 / RATIO_SCALE,
            self.edge_ratios[4] as f64 / RATIO_SCALE,
        ]
    }

    /// Get star indices.
    #[inline]
    pub fn stars(&self) -> [u16; 4] {
        self.star_indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn test_packed_star_roundtrip() {
        let ra = 1.234;
        let dec = 0.567;
        let mag = 4.5f32;
        let idx = 12345u16;

        let packed = PackedStar::new(ra, dec, mag, idx);

        // Check roundtrip (with some precision loss expected)
        assert!((packed.ra() - ra).abs() < 1e-6);
        assert!((packed.dec() - dec).abs() < 1e-6);
        assert!((packed.magnitude() - mag).abs() < 0.01);
        assert_eq!(packed.catalog_index(), idx);
    }

    #[test]
    fn test_packed_star_edge_cases() {
        // RA near 0 and 2π
        let packed = PackedStar::new(0.0, 0.0, 0.0, 0);
        assert!(packed.ra() < 0.001);

        // Dec at poles
        let packed = PackedStar::new(0.0, FRAC_PI_2, 0.0, 0);
        assert!((packed.dec() - FRAC_PI_2).abs() < 1e-6);

        let packed = PackedStar::new(0.0, -FRAC_PI_2, 0.0, 0);
        assert!((packed.dec() + FRAC_PI_2).abs() < 1e-6);
    }

    #[test]
    fn test_packed_pattern_roundtrip() {
        let stars = [1u16, 2, 3, 4];
        let ratios = [0.1, 0.25, 0.5, 0.75, 0.9];

        let packed = PackedPattern::new(stars, ratios);
        let recovered = packed.ratios();

        for i in 0..5 {
            assert!((recovered[i] - ratios[i]).abs() < 0.0001);
        }
        assert_eq!(packed.stars(), stars);
    }

    #[test]
    fn test_packed_sizes() {
        assert_eq!(std::mem::size_of::<PackedStar>(), 12);
        assert_eq!(std::mem::size_of::<PackedPattern>(), 18);
    }
}
