//! Bayesian verification of match hypotheses.

use std::cmp::Ordering;
use std::collections::HashSet;

use super::hypothesis::Hypothesis;
use crate::catalog::Index;
use crate::core::math;
use crate::core::types::{CatalogStar, DetectedStar, RaDec};
use crate::wcs::Wcs;

/// Spatial index for detected stars to accelerate nearest-neighbor lookups.
#[derive(Debug, Clone)]
pub struct DetectedStarIndex {
    cell_size: f64,
    grid_width: usize,
    grid_height: usize,
    cells: Vec<Vec<usize>>,
}

impl DetectedStarIndex {
    /// Build a grid index over detected stars.
    pub fn new(
        detected_stars: &[DetectedStar],
        image_width: u32,
        image_height: u32,
        max_match_distance: f64,
    ) -> Self {
        let cell_size = max_match_distance.max(1.0);
        let grid_width = ((image_width as f64 / cell_size).ceil() as usize).max(1);
        let grid_height = ((image_height as f64 / cell_size).ceil() as usize).max(1);
        let mut cells = vec![Vec::new(); grid_width * grid_height];

        for (idx, star) in detected_stars.iter().enumerate() {
            let gx = (star.x / cell_size).floor() as isize;
            let gy = (star.y / cell_size).floor() as isize;
            if gx < 0 || gy < 0 {
                continue;
            }
            let gx = gx as usize;
            let gy = gy as usize;
            if gx < grid_width && gy < grid_height {
                cells[gy * grid_width + gx].push(idx);
            }
        }

        Self {
            cell_size,
            grid_width,
            grid_height,
            cells,
        }
    }

    /// Gather nearby candidate detected-star indices around a pixel location.
    fn nearby_indices(&self, x: f64, y: f64, out: &mut Vec<usize>) {
        out.clear();

        let gx = (x / self.cell_size).floor() as isize;
        let gy = (y / self.cell_size).floor() as isize;

        for ny in (gy - 1)..=(gy + 1) {
            if ny < 0 || ny >= self.grid_height as isize {
                continue;
            }
            for nx in (gx - 1)..=(gx + 1) {
                if nx < 0 || nx >= self.grid_width as isize {
                    continue;
                }
                let idx = ny as usize * self.grid_width + nx as usize;
                out.extend(self.cells[idx].iter().copied());
            }
        }
    }
}

/// Spatial index for catalog stars using declination bands.
///
/// This avoids full catalog scans for every hypothesis verification call.
#[derive(Debug, Clone)]
pub struct CatalogStarIndex {
    dec_bin_size_rad: f64,
    dec_bins: Vec<Vec<CatalogStar>>,
}

impl CatalogStarIndex {
    /// Build a catalog-star index from the loaded pattern index.
    pub fn from_index(index: &Index, dec_bin_size_deg: f64) -> Self {
        let dec_bin_size_rad = dec_bin_size_deg
            .to_radians()
            .clamp(1f64.to_radians(), 30f64.to_radians());
        let num_bins = (std::f64::consts::PI / dec_bin_size_rad).ceil().max(1.0) as usize;
        let mut dec_bins = vec![Vec::new(); num_bins];

        for (idx, star) in index.stars() {
            let catalog_star = star.to_catalog_star(idx);
            let bin = Self::dec_to_bin(catalog_star.position.dec, dec_bin_size_rad, num_bins);
            dec_bins[bin].push(catalog_star);
        }

        Self {
            dec_bin_size_rad,
            dec_bins,
        }
    }

    /// Query stars within an angular cone around `center`.
    pub fn query_cone(&self, center: &RaDec, radius_rad: f64) -> Vec<CatalogStar> {
        let mut visible = Vec::new();
        let num_bins = self.dec_bins.len();
        if num_bins == 0 {
            return visible;
        }

        let search_radius = radius_rad.max(0.0);
        let dec_min = (center.dec - search_radius).max(-std::f64::consts::FRAC_PI_2);
        let dec_max = (center.dec + search_radius).min(std::f64::consts::FRAC_PI_2);
        let start_bin = Self::dec_to_bin(dec_min, self.dec_bin_size_rad, num_bins);
        let end_bin = Self::dec_to_bin(dec_max, self.dec_bin_size_rad, num_bins);

        for bin in start_bin..=end_bin {
            for cat in &self.dec_bins[bin] {
                if (cat.position.dec - center.dec).abs() > search_radius {
                    continue;
                }
                let sep = math::angular_separation(center, &cat.position);
                if sep < search_radius {
                    visible.push(*cat);
                }
            }
        }

        visible
    }

    fn dec_to_bin(dec_rad: f64, bin_size_rad: f64, num_bins: usize) -> usize {
        let bin = ((dec_rad + std::f64::consts::FRAC_PI_2) / bin_size_rad).floor() as isize;
        bin.clamp(0, num_bins as isize - 1) as usize
    }
}

/// Configuration for verification.
#[derive(Debug, Clone)]
pub struct VerifyConfig {
    /// Expected star position error in pixels (sigma).
    pub position_sigma_pixels: f64,
    /// Maximum distance in pixels to consider a match.
    pub max_match_distance_pixels: f64,
    /// Prior probability that a detected star is a false positive.
    pub false_positive_rate: f64,
    /// Minimum log-odds to accept a solution.
    pub log_odds_threshold: f64,
}

impl Default for VerifyConfig {
    fn default() -> Self {
        Self {
            position_sigma_pixels: 2.0,
            max_match_distance_pixels: 10.0,
            false_positive_rate: 0.1,
            log_odds_threshold: 20.0,
        }
    }
}

/// Result of hypothesis verification.
#[derive(Debug, Clone)]
pub struct VerifyResult {
    /// Updated hypothesis with log-odds.
    pub hypothesis: Hypothesis,
    /// Number of detected stars matched to catalog.
    pub num_matched: usize,
    /// Total number of catalog stars expected in field.
    pub num_expected: usize,
    /// RMS residual in pixels.
    pub rms_residual_pixels: f64,
    /// Median residual in pixels.
    pub median_residual_pixels: f64,
    /// Fraction of detected stars that were matched.
    pub detected_coverage: f64,
    /// Fraction of expected catalog stars that were matched.
    pub expected_coverage: f64,
    /// Brightness-rank consistency between detected and catalog matches [0, 1].
    pub brightness_consistency: f64,
    /// Individual residuals for each matched star.
    pub residuals: Vec<f64>,
}

/// Verify a hypothesis against detected stars.
pub fn verify_hypothesis(
    hypothesis: &Hypothesis,
    detected_stars: &[DetectedStar],
    index: &Index,
    image_width: u32,
    image_height: u32,
    config: &VerifyConfig,
) -> VerifyResult {
    verify_hypothesis_with_index(
        hypothesis,
        detected_stars,
        index,
        image_width,
        image_height,
        config,
        None,
        None,
    )
}

/// Verify a hypothesis against detected stars, optionally using a detected-star index.
pub fn verify_hypothesis_with_index(
    hypothesis: &Hypothesis,
    detected_stars: &[DetectedStar],
    index: &Index,
    image_width: u32,
    image_height: u32,
    config: &VerifyConfig,
    detected_index: Option<&DetectedStarIndex>,
    catalog_index: Option<&CatalogStarIndex>,
) -> VerifyResult {
    let wcs = &hypothesis.wcs;

    // Find expected catalog stars and cap to the brightest subset so wide-FOV
    // fields are not dominated by faint expected stars that are unlikely to be
    // detected in practice.
    let mut expected_stars = rank_and_cap_expected_stars(
        find_stars_in_fov(index, wcs, image_width, image_height, catalog_index),
        detected_stars.len(),
    );
    // Ensure the seed stars that produced the hypothesis are retained in the
    // expected set even when brightness capping truncates the cone query.
    for (_, cat_star) in &hypothesis.star_matches {
        if !expected_stars.iter().any(|s| s.id == cat_star.id) {
            expected_stars.push(*cat_star);
        }
    }

    // Match detected stars to expected catalog stars
    let mut matched: Vec<(usize, CatalogStar)> = Vec::new();
    let mut residuals = Vec::new();
    let mut used_detected = vec![false; detected_stars.len()];
    let mut candidate_indices = Vec::new();
    let mut seeded_catalog_ids: HashSet<u32> = HashSet::new();

    // Seed verification with the 4-star hypothesis correspondences. This
    // bootstraps robust hypotheses while still requiring additional support
    // in later acceptance logic.
    let seed_max_dist = config.max_match_distance_pixels * 2.0;
    for (det_idx, cat_star) in &hypothesis.star_matches {
        if *det_idx >= detected_stars.len() || used_detected[*det_idx] {
            continue;
        }
        let det_star = &detected_stars[*det_idx];
        let (pred_x, pred_y) = wcs.sky_to_pixel(&cat_star.position);
        let dx = det_star.x - pred_x;
        let dy = det_star.y - pred_y;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist <= seed_max_dist {
            matched.push((*det_idx, *cat_star));
            residuals.push(dist);
            used_detected[*det_idx] = true;
            seeded_catalog_ids.insert(cat_star.id);
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct MatchCandidate {
        det_idx: usize,
        cat_idx: usize,
        distance: f64,
    }

    // Build candidate pair list and solve one-to-one assignment greedily by
    // smallest residual. This is more stable than catalog-first greedy in
    // crowded regions and increases inlier retention for noisy detections.
    let mut pair_candidates: Vec<MatchCandidate> = Vec::new();
    for (cat_idx, cat_star) in expected_stars.iter().enumerate() {
        if seeded_catalog_ids.contains(&cat_star.id) {
            continue;
        }

        let (pred_x, pred_y) = wcs.sky_to_pixel(&cat_star.position);
        if pred_x < 0.0
            || pred_x >= image_width as f64
            || pred_y < 0.0
            || pred_y >= image_height as f64
        {
            continue;
        }

        if let Some(spatial_index) = detected_index {
            spatial_index.nearby_indices(pred_x, pred_y, &mut candidate_indices);
            for det_idx in &candidate_indices {
                if used_detected[*det_idx] {
                    continue;
                }
                let det_star = &detected_stars[*det_idx];
                let dx = det_star.x - pred_x;
                let dy = det_star.y - pred_y;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist <= config.max_match_distance_pixels {
                    pair_candidates.push(MatchCandidate {
                        det_idx: *det_idx,
                        cat_idx,
                        distance: dist,
                    });
                }
            }
        } else {
            for (det_idx, det_star) in detected_stars.iter().enumerate() {
                if used_detected[det_idx] {
                    continue;
                }

                let dx = det_star.x - pred_x;
                let dy = det_star.y - pred_y;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist <= config.max_match_distance_pixels {
                    pair_candidates.push(MatchCandidate {
                        det_idx,
                        cat_idx,
                        distance: dist,
                    });
                }
            }
        }
    }

    pair_candidates.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(Ordering::Equal)
    });

    let mut used_catalog = vec![false; expected_stars.len()];
    for (idx, cat_star) in expected_stars.iter().enumerate() {
        if seeded_catalog_ids.contains(&cat_star.id) {
            used_catalog[idx] = true;
        }
    }

    for cand in pair_candidates {
        if used_detected[cand.det_idx] || used_catalog[cand.cat_idx] {
            continue;
        }
        used_detected[cand.det_idx] = true;
        used_catalog[cand.cat_idx] = true;
        matched.push((cand.det_idx, expected_stars[cand.cat_idx]));
        residuals.push(cand.distance);
    }

    let num_matched = matched.len();
    let num_expected = expected_stars.len();
    let (effective_detected, effective_expected) =
        effective_population_sizes(num_matched, detected_stars.len(), num_expected);
    let expected_coverage = num_matched as f64 / effective_expected as f64;
    let detected_coverage = num_matched as f64 / effective_detected as f64;
    let brightness_consistency = compute_brightness_consistency(detected_stars, &matched);

    // Compute RMS residual
    let rms_residual = if residuals.is_empty() {
        f64::MAX
    } else {
        let sum_sq: f64 = residuals.iter().map(|r| r * r).sum();
        (sum_sq / residuals.len() as f64).sqrt()
    };
    let median_residual = median(&residuals).unwrap_or(f64::MAX);

    // Compute log-odds using Bayesian framework
    let log_odds = compute_log_odds(
        num_matched,
        detected_stars.len(),
        num_expected,
        &residuals,
        config,
        image_width,
        image_height,
    );

    let mut updated_hypothesis = hypothesis.clone();
    updated_hypothesis.log_odds = log_odds;

    // Update star matches to include all verified matches
    updated_hypothesis.star_matches = matched.clone();

    VerifyResult {
        hypothesis: updated_hypothesis,
        num_matched,
        num_expected,
        rms_residual_pixels: rms_residual,
        median_residual_pixels: median_residual,
        detected_coverage,
        expected_coverage,
        brightness_consistency,
        residuals,
    }
}

fn compute_brightness_consistency(
    detected_stars: &[DetectedStar],
    matched: &[(usize, CatalogStar)],
) -> f64 {
    let n = matched.len();
    if n < 3 {
        return 0.5;
    }

    // Rank by detected brightness (flux descending).
    let mut by_detected: Vec<(usize, f64)> = matched
        .iter()
        .enumerate()
        .map(|(mi, (det_idx, _))| (mi, detected_stars[*det_idx].flux))
        .collect();
    by_detected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let mut rank_detected = vec![0usize; n];
    for (rank, (mi, _)) in by_detected.iter().enumerate() {
        rank_detected[*mi] = rank;
    }

    // Rank by catalog brightness (magnitude ascending).
    let mut by_catalog: Vec<(usize, f32)> = matched
        .iter()
        .enumerate()
        .map(|(mi, (_, cat))| (mi, cat.magnitude))
        .collect();
    by_catalog.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    let mut rank_catalog = vec![0usize; n];
    for (rank, (mi, _)) in by_catalog.iter().enumerate() {
        rank_catalog[*mi] = rank;
    }

    let mut sum_d2 = 0.0f64;
    for i in 0..n {
        let d = rank_detected[i] as f64 - rank_catalog[i] as f64;
        sum_d2 += d * d;
    }

    let n_f = n as f64;
    let denom = n_f * (n_f * n_f - 1.0);
    if denom <= 1e-9 {
        return 0.5;
    }
    let rho = 1.0 - (6.0 * sum_d2) / denom;
    ((rho + 1.0) * 0.5).clamp(0.0, 1.0)
}

fn rank_and_cap_expected_stars(
    mut stars: Vec<CatalogStar>,
    num_detected: usize,
) -> Vec<CatalogStar> {
    if stars.is_empty() {
        return stars;
    }

    stars.sort_by(|a, b| {
        a.magnitude
            .partial_cmp(&b.magnitude)
            .unwrap_or(Ordering::Equal)
    });

    // Keep a moderate over-complete set of bright stars. Very large expected
    // sets (wide fields) destabilize verification score calibration and slow
    // matching while not increasing solve robustness.
    let cap = (num_detected.saturating_mul(3)).clamp(36, 360);
    if stars.len() > cap {
        stars.truncate(cap);
    }

    stars
}

/// Find catalog stars expected to be visible in the field of view.
fn find_stars_in_fov(
    index: &Index,
    wcs: &Wcs,
    image_width: u32,
    image_height: u32,
    catalog_index: Option<&CatalogStarIndex>,
) -> Vec<CatalogStar> {
    let mut visible = Vec::new();

    // Get approximate FOV center and radius
    let center = wcs.crval();
    let corners = [
        wcs.pixel_to_sky(0.0, 0.0),
        wcs.pixel_to_sky(image_width as f64, 0.0),
        wcs.pixel_to_sky(image_width as f64, image_height as f64),
        wcs.pixel_to_sky(0.0, image_height as f64),
    ];

    // Find max angular distance from center to any corner
    let fov_radius = corners
        .iter()
        .map(|c| math::angular_separation(&center, c))
        .fold(0.0f64, f64::max);

    let search_radius = fov_radius * 1.2;

    if let Some(spatial_index) = catalog_index {
        return spatial_index.query_cone(&center, search_radius);
    }

    // Fallback path: full scan.
    for (idx, star) in index.stars() {
        let pos = star.to_radec();
        let sep = math::angular_separation(&center, &pos);

        if sep < search_radius {
            visible.push(star.to_catalog_star(idx));
        }
    }

    visible
}

/// Compute log-odds score for the match hypothesis.
///
/// Combines three factors:
/// - Residual quality: penalizes when median residual is large relative to sigma
/// - Match count: logarithmic bonus for number of matches
/// - Match fraction: bonus for matching a large fraction of detected stars
fn compute_log_odds(
    num_matched: usize,
    num_detected: usize,
    num_expected: usize,
    residuals: &[f64],
    config: &VerifyConfig,
    image_width: u32,
    image_height: u32,
) -> f64 {
    if num_matched < 4 {
        return f64::NEG_INFINITY;
    }

    let sigma = config.position_sigma_pixels.max(0.5);
    let fp_rate = config.false_positive_rate.clamp(0.01, 0.85);
    let max_match = config.max_match_distance_pixels.max(1.0);
    let image_area = (image_width.max(1) as f64) * (image_height.max(1) as f64);

    let (effective_detected, effective_expected) =
        effective_population_sizes(num_matched, num_detected, num_expected);

    // Robust residual quality term (median over squared normalized residuals).
    let mut sq_normalized: Vec<f64> = residuals.iter().map(|r| (r / sigma).powi(2)).collect();
    sq_normalized.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let median_sq = sq_normalized[sq_normalized.len() / 2];
    let residual_score = if median_sq < 1.5 {
        -2.1 * median_sq
    } else {
        -3.15 - 5.8 * (median_sq - 1.5)
    };

    // Match support and coverage terms.
    let count_score = (num_matched as f64).ln() * 18.0;
    let match_fraction_detected = num_matched as f64 / effective_detected.max(1) as f64;
    let match_fraction_expected = num_matched as f64 / effective_expected.max(1) as f64;
    let coverage_score = 22.0 * match_fraction_detected + 30.0 * match_fraction_expected;

    // Mild penalties for weak support and very loose residual concentration.
    let miss_penalty = -7.5 * (1.0 - match_fraction_expected).powf(1.15);
    let false_penalty = -4.5 * fp_rate * (1.0 - match_fraction_detected).powf(1.0);
    let concentration_penalty = if let Some(med) = median(residuals) {
        -1.3 * (med / max_match).powi(2)
    } else {
        -20.0
    };

    // Tetra3-style resilience: reward inlier concentration but avoid letting
    // this dominate when only a handful of stars are matched.
    let inlier_ratio =
        residuals.iter().filter(|&&r| r <= sigma * 2.0).count() as f64 / num_matched as f64;
    let inlier_score = 6.0 * (inlier_ratio - 0.50);

    // Random-alignment term: approximate chance of a spurious match inside the
    // spatial tolerance window. This keeps low-support hypotheses in check.
    let p_rand_single =
        ((std::f64::consts::PI * max_match * max_match) / image_area).clamp(1e-9, 0.30);
    let random_alignment_score = ((num_matched as f64) * p_rand_single.ln().abs()) * 0.20;

    residual_score
        + count_score
        + coverage_score
        + miss_penalty
        + false_penalty
        + concentration_penalty
        + inlier_score
        + random_alignment_score
}

fn effective_population_sizes(
    num_matched: usize,
    num_detected: usize,
    num_expected: usize,
) -> (usize, usize) {
    // Evaluate against a bounded subset of detections and expected stars.
    // This keeps log-odds stable when wide-field catalog queries return many
    // stars that are below the practical detection limit.
    let effective_detected = num_detected
        .min(num_matched.saturating_mul(5).saturating_add(12))
        .max(num_matched)
        .max(1);
    let effective_expected = num_expected
        .min(num_detected.saturating_add(num_matched).saturating_add(6))
        .max(num_matched)
        .max(1);
    (effective_detected, effective_expected)
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut tmp = values.to_vec();
    tmp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    Some(tmp[tmp.len() / 2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_config_default() {
        let config = VerifyConfig::default();
        assert!(config.position_sigma_pixels > 0.0);
        assert!(config.log_odds_threshold > 0.0);
    }
}
