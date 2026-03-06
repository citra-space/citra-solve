//! Bayesian verification of match hypotheses.

use crate::core::types::{DetectedStar, CatalogStar};
use crate::core::math;
use crate::catalog::Index;
use crate::wcs::Wcs;
use super::hypothesis::Hypothesis;

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
) -> VerifyResult {
    let wcs = &hypothesis.wcs;

    // Find all catalog stars that should be in the field of view
    let expected_stars = find_stars_in_fov(index, wcs, image_width, image_height);

    // Match detected stars to expected catalog stars
    let mut matched = Vec::new();
    let mut residuals = Vec::new();
    let mut used_detected = vec![false; detected_stars.len()];
    let mut candidate_indices = Vec::new();

    // Greedy matching: for each expected star, find closest unmatched detected star
    for (cat_idx, cat_star) in expected_stars.iter().enumerate() {
        // Project catalog star to pixel coordinates
        let (pred_x, pred_y) = wcs.sky_to_pixel(&cat_star.position);

        // Skip if outside image
        if pred_x < 0.0
            || pred_x >= image_width as f64
            || pred_y < 0.0
            || pred_y >= image_height as f64
        {
            continue;
        }

        // Find closest unmatched detected star
        let mut best_dist = f64::MAX;
        let mut best_idx = None;

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
                if dist < best_dist && dist < config.max_match_distance_pixels {
                    best_dist = dist;
                    best_idx = Some(*det_idx);
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

                if dist < best_dist && dist < config.max_match_distance_pixels {
                    best_dist = dist;
                    best_idx = Some(det_idx);
                }
            }
        }

        if let Some(det_idx) = best_idx {
            matched.push((det_idx, cat_idx));
            residuals.push(best_dist);
            used_detected[det_idx] = true;
        }
    }

    let num_matched = matched.len();
    let num_expected = expected_stars.len();

    // Compute RMS residual
    let rms_residual = if residuals.is_empty() {
        f64::MAX
    } else {
        let sum_sq: f64 = residuals.iter().map(|r| r * r).sum();
        (sum_sq / residuals.len() as f64).sqrt()
    };

    // Compute log-odds using Bayesian framework
    let log_odds = compute_log_odds(
        num_matched,
        detected_stars.len(),
        num_expected,
        &residuals,
        config,
    );

    let mut updated_hypothesis = hypothesis.clone();
    updated_hypothesis.log_odds = log_odds;

    // Update star matches to include all verified matches
    updated_hypothesis.star_matches = matched
        .iter()
        .map(|&(det_idx, cat_idx)| (det_idx, expected_stars[cat_idx]))
        .collect();

    VerifyResult {
        hypothesis: updated_hypothesis,
        num_matched,
        num_expected,
        rms_residual_pixels: rms_residual,
        residuals,
    }
}

/// Find catalog stars expected to be visible in the field of view.
fn find_stars_in_fov(
    index: &Index,
    wcs: &Wcs,
    image_width: u32,
    image_height: u32,
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

    // Scan all stars (for now - could use spatial index)
    for (idx, star) in index.stars() {
        let pos = star.to_radec();
        let sep = math::angular_separation(&center, &pos);

        if sep < fov_radius * 1.2 {
            // 20% margin
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
) -> f64 {
    if num_matched < 4 {
        return f64::NEG_INFINITY;
    }

    let sigma = config.position_sigma_pixels;

    // Use median squared normalized residual (robust to outliers)
    let mut sq_normalized: Vec<f64> = residuals
        .iter()
        .map(|r| (r / sigma).powi(2))
        .collect();
    sq_normalized.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_sq = sq_normalized[sq_normalized.len() / 2];

    // Residual quality score: tolerant near 1-sigma, then quickly penalize.
    let residual_score = if median_sq < 1.5 {
        -3.0 * median_sq
    } else {
        -4.5 - 8.0 * (median_sq - 1.5)
    };

    // Match count bonus (log scale to avoid dominating)
    let count_score = (num_matched as f64).ln() * 15.0;

    // Match coverage bonuses.
    let match_fraction_detected = num_matched as f64 / num_detected.max(1) as f64;
    let match_fraction_expected = num_matched as f64 / num_expected.max(1) as f64;
    let coverage_score = 22.0 * match_fraction_detected + 34.0 * match_fraction_expected;

    // Bounded penalty for weak expected-star coverage.
    let miss_penalty = -12.0 * (1.0 - match_fraction_expected).powf(1.3);

    residual_score + count_score + coverage_score + miss_penalty
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
