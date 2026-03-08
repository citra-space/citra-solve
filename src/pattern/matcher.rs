//! Pattern matching between detected quads and catalog patterns.

use super::hash::{query_hash_bins, ratios_match};
use super::quad::Quad;
use crate::catalog::star::PackedPattern;
use crate::catalog::Index;

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
    /// Hard cap on hash bins queried per quad to avoid combinatorial blowups.
    pub max_hash_bins: usize,
    /// Hard cap on patterns scanned per hash bin.
    pub max_patterns_per_bin: usize,
}

impl<'a> PatternMatcher<'a> {
    /// Create a new pattern matcher.
    pub fn new(index: &'a Index) -> Self {
        Self {
            index,
            bin_tolerance: 0.02,
            ratio_tolerance: 0.01,
            max_hash_bins: 20_000,
            max_patterns_per_bin: 4_096,
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

    /// Set max hash bins queried per quad.
    pub fn with_max_hash_bins(mut self, max_hash_bins: usize) -> Self {
        self.max_hash_bins = max_hash_bins.max(1);
        self
    }

    /// Set max patterns scanned per queried hash bin.
    pub fn with_max_patterns_per_bin(mut self, max_patterns_per_bin: usize) -> Self {
        self.max_patterns_per_bin = max_patterns_per_bin.max(1);
        self
    }

    /// Find all catalog patterns matching a single detected quad.
    pub fn find_matches(&self, quad: &Quad) -> Vec<PatternMatch> {
        self.find_matches_for_quad(0, quad, usize::MAX)
    }

    /// Find matches for a quad with its index.
    fn find_matches_for_quad(
        &self,
        quad_idx: usize,
        quad: &Quad,
        max_matches: usize,
    ) -> Vec<PatternMatch> {
        if max_matches == 0 {
            return Vec::new();
        }

        let mut matches = Vec::new();

        // Get all bins to query
        let mut bins = query_hash_bins(quad.ratios(), self.index.num_bins(), self.bin_tolerance);
        if bins.len() > self.max_hash_bins {
            bins.truncate(self.max_hash_bins);
        }

        // Query each bin
        for bin in bins {
            if let Ok(patterns) = self.index.get_patterns_in_bin(bin) {
                for pattern in patterns.iter().take(self.max_patterns_per_bin) {
                    let cat_ratios = pattern.ratios();
                    if ratios_match(quad.ratios(), &cat_ratios, self.ratio_tolerance) {
                        let dist = super::hash::ratio_distance(quad.ratios(), &cat_ratios);
                        let candidate = PatternMatch {
                            detected_quad_idx: quad_idx,
                            catalog_pattern: *pattern,
                            ratio_distance: dist,
                        };

                        if max_matches == usize::MAX {
                            matches.push(candidate);
                        } else if matches.len() < max_matches {
                            matches.push(candidate);
                        } else {
                            // Keep only the best N matches by replacing the current
                            // worst candidate when a better one is found.
                            let mut worst_idx = 0usize;
                            let mut worst_dist = matches[0].ratio_distance;
                            for (idx, m) in matches.iter().enumerate().skip(1) {
                                if m.ratio_distance > worst_dist {
                                    worst_dist = m.ratio_distance;
                                    worst_idx = idx;
                                }
                            }
                            if candidate.ratio_distance < worst_dist {
                                matches[worst_idx] = candidate;
                            }
                        }
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
            let matches = self.find_matches_for_quad(idx, quad, max_matches_per_quad);
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
        quads.iter().map(|q| self.find_matches(q).len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Integration tests would require an actual index file
}
