//! Solution refinement using iterative least squares.

use crate::core::types::{DetectedStar, CatalogStar, RaDec, Vec3};
use crate::core::math;
use crate::wcs::Wcs;

/// Configuration for refinement.
#[derive(Debug, Clone)]
pub struct RefineConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence threshold (RMS change in arcsec).
    pub convergence_threshold: f64,
    /// Sigma clipping threshold for outlier rejection.
    pub sigma_clip: f64,
}

impl Default for RefineConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            convergence_threshold: 0.1, // 0.1 arcsec
            sigma_clip: 3.0,
        }
    }
}

/// Result of refinement.
#[derive(Debug, Clone)]
pub struct RefineResult {
    /// Refined WCS solution.
    pub wcs: Wcs,
    /// Final RMS residual in arcseconds.
    pub rms_arcsec: f64,
    /// Number of stars used (after outlier rejection).
    pub num_stars_used: usize,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether convergence was achieved.
    pub converged: bool,
}

/// Refine a WCS solution using iterative least squares.
pub fn refine_solution(
    detected_stars: &[DetectedStar],
    star_matches: &[(usize, CatalogStar)],
    initial_wcs: &Wcs,
    config: &RefineConfig,
) -> RefineResult {
    if star_matches.len() < 3 {
        return RefineResult {
            wcs: initial_wcs.clone(),
            rms_arcsec: f64::MAX,
            num_stars_used: star_matches.len(),
            iterations: 0,
            converged: false,
        };
    }

    let mut wcs = initial_wcs.clone();
    let mut active_matches: Vec<bool> = vec![true; star_matches.len()];
    let mut prev_rms = f64::MAX;

    for iteration in 0..config.max_iterations {
        // Collect active matches
        let active: Vec<&(usize, CatalogStar)> = star_matches
            .iter()
            .zip(active_matches.iter())
            .filter_map(|(m, &active)| if active { Some(m) } else { None })
            .collect();

        if active.len() < 3 {
            return RefineResult {
                wcs,
                rms_arcsec: prev_rms,
                num_stars_used: active.len(),
                iterations: iteration,
                converged: false,
            };
        }

        // Compute residuals
        let mut residuals = Vec::with_capacity(active.len());
        for &(det_idx, ref cat_star) in &active {
            let det = &detected_stars[*det_idx];
            let predicted = wcs.sky_to_pixel(&cat_star.position);
            let dx = det.x - predicted.0;
            let dy = det.y - predicted.1;
            residuals.push((dx, dy));
        }

        // Compute RMS in pixels, then convert to arcsec
        let rms_pixels = compute_rms(&residuals);
        let pixel_scale = wcs.pixel_scale_arcsec();
        let rms_arcsec = rms_pixels * pixel_scale;

        // Check convergence
        if (prev_rms - rms_arcsec).abs() < config.convergence_threshold {
            return RefineResult {
                wcs,
                rms_arcsec,
                num_stars_used: active.len(),
                iterations: iteration + 1,
                converged: true,
            };
        }
        prev_rms = rms_arcsec;

        // Sigma clipping
        let (mean_dx, std_dx) = mean_std_1d(&residuals.iter().map(|r| r.0).collect::<Vec<_>>());
        let (mean_dy, std_dy) = mean_std_1d(&residuals.iter().map(|r| r.1).collect::<Vec<_>>());

        let mut active_iter = active_matches.iter_mut();
        for &(det_idx, _) in star_matches.iter() {
            if let Some(is_active) = active_iter.next() {
                if *is_active {
                    let det = &detected_stars[det_idx];
                    let predicted = wcs.sky_to_pixel(&star_matches.iter().find(|(i, _)| *i == det_idx).unwrap().1.position);
                    let dx = det.x - predicted.0;
                    let dy = det.y - predicted.1;

                    if (dx - mean_dx).abs() > config.sigma_clip * std_dx
                        || (dy - mean_dy).abs() > config.sigma_clip * std_dy
                    {
                        *is_active = false;
                    }
                }
            }
        }

        // Update WCS parameters
        // Simple approach: adjust CRVAL based on mean residual
        let mean_residual_ra = mean_dx * pixel_scale / 3600.0 * std::f64::consts::PI / 180.0;
        let mean_residual_dec = mean_dy * pixel_scale / 3600.0 * std::f64::consts::PI / 180.0;

        let new_crval = RaDec::new(
            wcs.crval().ra - mean_residual_ra / wcs.crval().dec.cos(),
            wcs.crval().dec - mean_residual_dec,
        );
        wcs = wcs.with_crval(new_crval);
    }

    RefineResult {
        wcs,
        rms_arcsec: prev_rms,
        num_stars_used: active_matches.iter().filter(|&&a| a).count(),
        iterations: config.max_iterations,
        converged: false,
    }
}

fn compute_rms(residuals: &[(f64, f64)]) -> f64 {
    if residuals.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = residuals.iter().map(|(dx, dy)| dx * dx + dy * dy).sum();
    (sum_sq / residuals.len() as f64).sqrt()
}

fn mean_std_1d(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 1.0);
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    (mean, variance.sqrt().max(0.001)) // Avoid zero std
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refine_config_default() {
        let config = RefineConfig::default();
        assert!(config.max_iterations > 0);
        assert!(config.sigma_clip > 0.0);
    }

    #[test]
    fn test_compute_rms() {
        let residuals = vec![(3.0, 4.0)];
        assert!((compute_rms(&residuals) - 5.0).abs() < 1e-10);
    }
}
