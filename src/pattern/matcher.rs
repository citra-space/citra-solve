//! Pattern matching between detected quads and catalog patterns.

use crate::catalog::Index;
use crate::catalog::star::PackedPattern;
use super::quad::Quad;
use super::hash::{query_hash_bins, ratios_match};

/// A match between a detected quad and a catalog pattern.
#[derive(Debug, Clone)]
pub struct PatternMatch {
    /// Index of the detected quad.
    pub detected_quad_idx: usize,
    /// The catalog pattern that matched.
    pub catalog_pattern: PackedPattern,
    /// Distance between ratio vectors (lower is better).
    pub ratio_distance: f64,
}

/// Pattern matcher for finding catalog matches.
pub struct PatternMatcher<'a> {
    index: &'a Index,
    /// Tolerance for hash bin queries (expands search space).
    pub bin_tolerance: f64,
    /// Tolerance for ratio matching (final verification).
    pub ratio_tolerance: f64,
}

impl<'a> PatternMatcher<'a> {
    /// Create a new pattern matcher.
    pub fn new(index: &'a Index) -> Self {
        Self {
            index,
            bin_tolerance: 0.02,
            ratio_tolerance: 0.01,
        }
    }

    /// Set the hash bin tolerance.
    pub fn with_bin_tolerance(mut self, tolerance: f64) -> Self {
        self.bin_tolerance = tolerance;
        self
    }

    /// Set the ratio matching tolerance.
    pub fn with_ratio_tolerance(mut self, tolerance: f64) -> Self {
        self.ratio_tolerance = tolerance;
        self
    }

    /// Find all catalog patterns matching a single detected quad.
    pub fn find_matches(&self, quad: &Quad) -> Vec<PatternMatch> {
        self.find_matches_for_quad(0, quad)
    }

    /// Find matches for a quad with its index.
    fn find_matches_for_quad(&self, quad_idx: usize, quad: &Quad) -> Vec<PatternMatch> {
        let mut matches = Vec::new();

        // Get all bins to query
        let bins = query_hash_bins(
            quad.ratios(),
            self.index.num_bins(),
            self.bin_tolerance,
        );

        // Query each bin
        for bin in bins {
            if let Ok(patterns) = self.index.get_patterns_in_bin(bin) {
                for pattern in patterns {
                    let cat_ratios = pattern.ratios();
                    if ratios_match(quad.ratios(), &cat_ratios, self.ratio_tolerance) {
                        let dist = super::hash::ratio_distance(quad.ratios(), &cat_ratios);
                        matches.push(PatternMatch {
                            detected_quad_idx: quad_idx,
                            catalog_pattern: *pattern,
                            ratio_distance: dist,
                        });
                    }
                }
            }
        }

        // Sort by ratio distance (best matches first)
        matches.sort_by(|a, b| {
            a.ratio_distance
                .partial_cmp(&b.ratio_distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        matches
    }

    /// Find matches for multiple quads, returning best matches.
    pub fn find_matches_batch(
        &self,
        quads: &[Quad],
        max_matches_per_quad: usize,
    ) -> Vec<PatternMatch> {
        let mut all_matches = Vec::new();

        for (idx, quad) in quads.iter().enumerate() {
            let mut matches = self.find_matches_for_quad(idx, quad);
            matches.truncate(max_matches_per_quad);
            all_matches.extend(matches);
        }

        // Sort all matches by ratio distance
        all_matches.sort_by(|a, b| {
            a.ratio_distance
                .partial_cmp(&b.ratio_distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        all_matches
    }

    /// Count total matches across all quads (useful for diagnostics).
    pub fn count_matches(&self, quads: &[Quad]) -> usize {
        quads
            .iter()
            .map(|q| self.find_matches(q).len())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Integration tests would require an actual index file
}
