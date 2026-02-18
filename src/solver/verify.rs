//! Bayesian verification of match hypotheses.

use crate::core::types::{DetectedStar, CatalogStar};
use crate::core::math;
use crate::catalog::Index;
use crate::wcs::Wcs;
use super::hypothesis::Hypothesis;

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
    let wcs = &hypothesis.wcs;

    // Find all catalog stars that should be in the field of view
    let expected_stars = find_stars_in_fov(index, wcs, image_width, image_height);

    // Match detected stars to expected catalog stars
    let mut matched = Vec::new();
    let mut residuals = Vec::new();
    let mut used_detected = vec![false; detected_stars.len()];
    let mut used_catalog = vec![false; expected_stars.len()];

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

        if let Some(det_idx) = best_idx {
            matched.push((det_idx, cat_idx));
            residuals.push(best_dist);
            used_detected[det_idx] = true;
            used_catalog[cat_idx] = true;
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
    _num_expected: usize,
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

    // Residual quality score: gentle penalty when median is near expected,
    // steep penalty when median is much larger than expected.
    let residual_score = if median_sq < 2.0 {
        -2.0 * median_sq
    } else {
        -4.0 - 5.0 * (median_sq - 2.0)
    };

    // Match count bonus (log scale to avoid dominating)
    let count_score = (num_matched as f64).ln() * 15.0;

    // Match fraction bonus
    let match_fraction = num_matched as f64 / num_detected.max(1) as f64;
    let fraction_score = 30.0 * match_fraction;

    residual_score + count_score + fraction_score
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
