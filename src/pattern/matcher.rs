//! Pattern matching between detected quads and catalog patterns.

use super::hash::{query_hash_bins_ranked, ratio_distance, ratios_match_per_dim};
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
    /// Integer-grid distance of the hash probe from hash center.
    pub hash_l1_distance: u16,
    /// Distance between predicted and matched scale bands.
    pub scale_band_distance: u8,
    /// Composite ranking score.
    pub match_score: f64,
}

/// Pattern matcher for finding catalog matches.
pub struct PatternMatcher<'a> {
    index: &'a Index,
    /// Tolerance for hash bin queries (expands search space).
    pub bin_tolerance: f64,
    /// Tolerance for ratio matching (final verification).
    pub ratio_tolerance: f64,
    /// Tolerance for tetra signature matching (per dimension).
    pub tetra_tolerance: f64,
    /// Hard cap on hash bins queried per quad to avoid combinatorial blowups.
    pub max_hash_bins: usize,
    /// Hard cap on patterns scanned per hash bin.
    pub max_patterns_per_bin: usize,
    /// Optional geometric context for scale-band prediction.
    image_diagonal_pixels: Option<f64>,
    fov_hint_deg: Option<(f64, f64)>,
    /// Maximum allowed mismatch between predicted and stored scale band.
    pub max_scale_band_delta: u8,
}

impl<'a> PatternMatcher<'a> {
    /// Create a new pattern matcher.
    pub fn new(index: &'a Index) -> Self {
        Self {
            index,
            bin_tolerance: 0.02,
            ratio_tolerance: 0.01,
            tetra_tolerance: 0.030,
            max_hash_bins: 20_000,
            max_patterns_per_bin: 4_096,
            image_diagonal_pixels: None,
            fov_hint_deg: None,
            max_scale_band_delta: 1,
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

    /// Set tetra signature matching tolerance.
    pub fn with_tetra_tolerance(mut self, tolerance: f64) -> Self {
        self.tetra_tolerance = tolerance.max(0.001);
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

    /// Set scale hint context to prioritize relevant scale bands.
    pub fn with_scale_hint(
        mut self,
        image_diagonal_pixels: f64,
        fov_min_deg: f64,
        fov_max_deg: f64,
    ) -> Self {
        let min = fov_min_deg.max(0.1);
        let max = fov_max_deg.max(min + 0.1);
        self.image_diagonal_pixels = Some(image_diagonal_pixels.max(1.0));
        self.fov_hint_deg = Some((min, max));
        self
    }

    /// Set allowed distance between predicted and stored scale bands.
    pub fn with_max_scale_band_delta(mut self, max_scale_band_delta: u8) -> Self {
        self.max_scale_band_delta = max_scale_band_delta;
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
        let band_distance = self.scale_band_distance_lookup(quad);
        let probes = query_hash_bins_ranked(
            quad.ratios(),
            quad.tetra(),
            self.index.num_bins(),
            self.ratio_tolerance,
            self.tetra_tolerance,
            self.max_hash_bins,
        );

        for probe in probes {
            let Ok(patterns) = self.index.get_patterns_in_bin(probe.bin) else {
                continue;
            };

            for pattern in patterns.iter().take(self.max_patterns_per_bin) {
                let band_delta = band_distance
                    .get(pattern.scale_band() as usize)
                    .copied()
                    .unwrap_or(u8::MAX);
                if band_delta > self.max_scale_band_delta {
                    continue;
                }

                let mut scale_penalty = 0.0;
                if let (Some(image_diag), Some((fov_min, fov_max))) =
                    (self.image_diagonal_pixels, self.fov_hint_deg)
                {
                    let edge_deg = pattern.max_edge_deg();
                    if edge_deg <= 0.02 {
                        continue;
                    }
                    let implied_fov = edge_deg * image_diag / quad.max_edge_pixels.max(1.0);
                    let min_ok = fov_min * 0.86;
                    let max_ok = fov_max * 1.16;
                    if implied_fov < min_ok || implied_fov > max_ok {
                        continue;
                    }
                    let center = (fov_min * fov_max).sqrt();
                    scale_penalty = (implied_fov / center).ln().abs().min(2.5);
                }

                let cat_ratios = pattern.ratios();
                if !ratios_match_per_dim(quad.ratios(), &cat_ratios, self.ratio_tolerance) {
                    continue;
                }
                let cat_tetra = pattern.tetra_signature();
                if !tetra_match(quad.tetra(), &cat_tetra, self.tetra_tolerance) {
                    continue;
                }

                let ratio_dist = ratio_distance(quad.ratios(), &cat_ratios);
                let tetra_dist = tetra_distance(quad.tetra(), &cat_tetra);
                let score = ratio_dist
                    + 0.55 * tetra_dist
                    + 0.0018 * probe.l1_distance as f64
                    + 0.010 * band_delta as f64
                    + 0.080 * scale_penalty;

                let candidate = PatternMatch {
                    detected_quad_idx: quad_idx,
                    catalog_pattern: *pattern,
                    ratio_distance: ratio_dist,
                    hash_l1_distance: probe.l1_distance,
                    scale_band_distance: band_delta,
                    match_score: score,
                };

                if max_matches == usize::MAX {
                    matches.push(candidate);
                } else if matches.len() < max_matches {
                    matches.push(candidate);
                } else {
                    let mut worst_idx = 0usize;
                    let mut worst_score = matches[0].match_score;
                    for (idx, m) in matches.iter().enumerate().skip(1) {
                        if m.match_score > worst_score {
                            worst_score = m.match_score;
                            worst_idx = idx;
                        }
                    }
                    if candidate.match_score < worst_score {
                        matches[worst_idx] = candidate;
                    }
                }
            }
        }

        matches.sort_by(|a, b| {
            a.match_score
                .partial_cmp(&b.match_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    a.ratio_distance
                        .partial_cmp(&b.ratio_distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        matches
    }

    fn estimate_scale_band_candidates(&self, quad: &Quad) -> Vec<(u8, u8)> {
        let num_bands = self.index.num_scale_bands().max(1);
        if num_bands == 1 {
            return vec![(0, 0)];
        }

        let mut out = Vec::new();
        if let (Some(image_diag), Some((fov_min, fov_max))) =
            (self.image_diagonal_pixels, self.fov_hint_deg)
        {
            let frac = (quad.max_edge_pixels / image_diag).clamp(1e-4, 1.0);
            let min_angle = (frac * fov_min * 0.78).max(0.02);
            let max_angle = (frac * fov_max * 1.28).max(min_angle + 0.01);
            let center_angle = (min_angle * max_angle).sqrt();

            let mut min_band = self.index.scale_band_for_angle_deg(min_angle) as i32;
            let mut max_band = self.index.scale_band_for_angle_deg(max_angle) as i32;
            let center_band = self.index.scale_band_for_angle_deg(center_angle) as i32;

            if min_band > max_band {
                std::mem::swap(&mut min_band, &mut max_band);
            }

            let expand = self.max_scale_band_delta as i32;
            min_band = (min_band - expand).max(0);
            max_band = (max_band + expand).min(num_bands as i32 - 1);

            for band in min_band..=max_band {
                out.push((band as u8, (band - center_band).abs() as u8));
            }

            out.sort_by_key(|(_, dist)| *dist);
            out.dedup_by(|a, b| a.0 == b.0);
        }

        if out.is_empty() {
            for band in 0..num_bands {
                out.push((band, 0));
            }
        }

        out
    }

    fn scale_band_distance_lookup(&self, quad: &Quad) -> Vec<u8> {
        let num_bands = self.index.num_scale_bands().max(1) as usize;
        let mut lookup = vec![u8::MAX; num_bands];
        let candidates = self.estimate_scale_band_candidates(quad);
        if candidates.is_empty() {
            lookup.fill(0);
            return lookup;
        }

        for (band, dist) in candidates {
            let idx = band as usize;
            if idx < lookup.len() {
                lookup[idx] = lookup[idx].min(dist);
            }
        }
        lookup
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

        // Sort all matches by combined score.
        all_matches.sort_by(|a, b| {
            a.match_score
                .partial_cmp(&b.match_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        all_matches
    }

    /// Count total matches across all quads (useful for diagnostics).
    pub fn count_matches(&self, quads: &[Quad]) -> usize {
        quads.iter().map(|q| self.find_matches(q).len()).sum()
    }
}

fn tetra_match(a: &[f64; 4], b: &[f64; 4], tol: f64) -> bool {
    for i in 0..4 {
        if (a[i] - b[i]).abs() > tol {
            return false;
        }
    }
    true
}

fn tetra_distance(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let mut sum = 0.0;
    for i in 0..4 {
        sum += (a[i] - b[i]).abs();
    }
    sum / 4.0
}

#[cfg(test)]
mod tests {
    use super::*;

    // Integration tests would require an actual index file.
}
