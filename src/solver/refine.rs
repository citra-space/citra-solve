//! Solution refinement using iterative least squares and SIP distortion.

use crate::core::types::{DetectedStar, CatalogStar, RaDec};
use crate::wcs::{Wcs, WcsWithDistortion, SipDistortion, fit_sip_distortion};

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

/// Result of linear WCS refinement.
#[derive(Debug, Clone)]
pub struct LinearRefineResult {
    /// Refined linear WCS.
    pub wcs: Wcs,
    /// RMS residual before refinement (pixels).
    pub rms_before_pixels: f64,
    /// RMS residual after refinement (pixels).
    pub rms_after_pixels: f64,
    /// Number of stars used.
    pub num_stars_used: usize,
}

/// Result of SIP distortion fitting.
#[derive(Debug, Clone)]
pub struct SipRefineResult {
    /// WCS with SIP distortion.
    pub wcs: WcsWithDistortion,
    /// RMS residual before distortion correction (pixels).
    pub rms_before_pixels: f64,
    /// RMS residual after distortion correction (pixels).
    pub rms_after_pixels: f64,
    /// Number of stars used in fit.
    pub num_stars_used: usize,
    /// SIP polynomial order used.
    pub sip_order: usize,
}

/// Fit SIP distortion coefficients to improve WCS accuracy.
///
/// This takes a linear WCS and matched star pairs, and fits polynomial
/// distortion coefficients to reduce residuals.
///
/// # Arguments
/// * `detected_stars` - All detected stars
/// * `star_matches` - Matched pairs (detected index, catalog star)
/// * `wcs` - Initial linear WCS
/// * `sip_order` - Polynomial order (2-5, typically 2 or 3)
///
/// # Returns
/// SipRefineResult with the improved WCS and statistics
pub fn fit_sip(
    detected_stars: &[DetectedStar],
    star_matches: &[(usize, CatalogStar)],
    wcs: &Wcs,
    sip_order: usize,
) -> SipRefineResult {
    let sip_order = sip_order.clamp(2, 5);

    // Extract pixel and sky coordinates from matches
    let mut pixels: Vec<(f64, f64)> = Vec::with_capacity(star_matches.len());
    let mut sky: Vec<RaDec> = Vec::with_capacity(star_matches.len());

    for (det_idx, cat_star) in star_matches {
        let det = &detected_stars[*det_idx];
        pixels.push((det.x, det.y));
        sky.push(cat_star.position);
    }

    // Compute RMS before distortion
    let rms_before = compute_pixel_rms(&pixels, &sky, wcs);

    // Need enough points for the polynomial fit
    // Order 2: 3 terms, Order 3: 7 terms, Order 4: 12 terms
    let min_points = match sip_order {
        2 => 6,
        3 => 10,
        4 => 15,
        _ => 20,
    };

    if star_matches.len() < min_points {
        return SipRefineResult {
            wcs: WcsWithDistortion::new(wcs.clone()),
            rms_before_pixels: rms_before,
            rms_after_pixels: rms_before,
            num_stars_used: star_matches.len(),
            sip_order: 0,
        };
    }

    // Fit SIP distortion
    let sip = fit_sip_distortion(&pixels, &sky, wcs, sip_order);

    // Create WCS with distortion
    let wcs_with_dist = WcsWithDistortion::with_distortion(wcs.clone(), sip);

    // Compute RMS after distortion
    let rms_after = compute_pixel_rms_with_dist(&pixels, &sky, &wcs_with_dist);

    SipRefineResult {
        wcs: wcs_with_dist,
        rms_before_pixels: rms_before,
        rms_after_pixels: rms_after,
        num_stars_used: star_matches.len(),
        sip_order,
    }
}

/// Compute RMS residual in pixels (linear WCS).
fn compute_pixel_rms(
    detected: &[(f64, f64)],
    catalog: &[RaDec],
    wcs: &Wcs,
) -> f64 {
    if detected.is_empty() {
        return 0.0;
    }
    let mut sum_sq = 0.0;
    for (det, cat) in detected.iter().zip(catalog.iter()) {
        let (pred_x, pred_y) = wcs.sky_to_pixel(cat);
        let dx = det.0 - pred_x;
        let dy = det.1 - pred_y;
        sum_sq += dx * dx + dy * dy;
    }
    (sum_sq / detected.len() as f64).sqrt()
}

/// Compute RMS residual in pixels (WCS with distortion).
fn compute_pixel_rms_with_dist(
    detected: &[(f64, f64)],
    catalog: &[RaDec],
    wcs: &WcsWithDistortion,
) -> f64 {
    if detected.is_empty() {
        return 0.0;
    }
    let mut sum_sq = 0.0;
    for (det, cat) in detected.iter().zip(catalog.iter()) {
        let (pred_x, pred_y) = wcs.sky_to_pixel(cat);
        let dx = det.0 - pred_x;
        let dy = det.1 - pred_y;
        sum_sq += dx * dx + dy * dy;
    }
    (sum_sq / detected.len() as f64).sqrt()
}

/// Re-estimate the linear WCS (CRPIX, CRVAL, CD) using all matched stars.
///
/// This improves upon the initial 4-star estimate by using the full set of
/// verified matches to compute a better least-squares fit for the CD matrix.
///
/// # Arguments
/// * `detected_stars` - All detected stars
/// * `star_matches` - Verified matched pairs (detected index, catalog star)
/// * `initial_wcs` - Initial WCS estimate (used for CRVAL at image center)
/// * `image_width` - Image width in pixels
/// * `image_height` - Image height in pixels
///
/// # Returns
/// LinearRefineResult with the improved WCS
pub fn refine_linear_wcs(
    detected_stars: &[DetectedStar],
    star_matches: &[(usize, CatalogStar)],
    initial_wcs: &Wcs,
    image_width: u32,
    image_height: u32,
) -> Option<LinearRefineResult> {
    if star_matches.len() < 6 {
        return None;
    }

    // Extract pixel and sky coordinates
    let mut pixels: Vec<(f64, f64)> = Vec::with_capacity(star_matches.len());
    let mut sky: Vec<RaDec> = Vec::with_capacity(star_matches.len());

    for (det_idx, cat_star) in star_matches {
        let det = &detected_stars[*det_idx];
        pixels.push((det.x, det.y));
        sky.push(cat_star.position);
    }

    // Use image center as reference point for better numerical stability
    let crpix = (image_width as f64 / 2.0, image_height as f64 / 2.0);

    // Use initial WCS to get sky position at image center as CRVAL
    // This is more accurate than using centroid of matched stars
    let crval = initial_wcs.pixel_to_sky(crpix.0, crpix.1);

    // Precompute trig values for crval
    let sin_crval_dec = crval.dec.sin();
    let cos_crval_dec = crval.dec.cos();

    // Build matrices for least squares: pixel_offset = CD * tangent_plane
    // We solve: dx = CD11*xi + CD12*eta, dy = CD21*xi + CD22*eta
    let mut a_mat: Vec<[f64; 2]> = Vec::new();
    let mut b_x: Vec<f64> = Vec::new();
    let mut b_y: Vec<f64> = Vec::new();

    for ((px, py), sky_pos) in pixels.iter().zip(sky.iter()) {
        let dx = px - crpix.0;
        let dy = py - crpix.1;

        // Compute tangent plane coordinates
        let delta_ra = sky_pos.ra - crval.ra;
        let sin_dec = sky_pos.dec.sin();
        let cos_dec = sky_pos.dec.cos();
        let sin_delta_ra = delta_ra.sin();
        let cos_delta_ra = delta_ra.cos();

        let denom = sin_crval_dec * sin_dec + cos_crval_dec * cos_dec * cos_delta_ra;
        if denom < 0.1 {
            continue;
        }

        let xi = (cos_dec * sin_delta_ra) / denom;
        let eta = (cos_crval_dec * sin_dec - sin_crval_dec * cos_dec * cos_delta_ra) / denom;

        a_mat.push([xi, eta]);
        b_x.push(dx);
        b_y.push(dy);
    }

    if a_mat.len() < 4 {
        return None;
    }

    // Solve least squares for CD matrix (pixel/radian)
    let (cd11, cd12) = solve_2x2_lstsq(&a_mat, &b_x)?;
    let (cd21, cd22) = solve_2x2_lstsq(&a_mat, &b_y)?;

    // Convert from pixel/radian to degrees/pixel (WCS convention)
    let cd_det = cd11 * cd22 - cd12 * cd21;
    if cd_det.abs() < 1e-20 {
        return None;
    }

    let rad_to_deg = 180.0 / std::f64::consts::PI;
    let cd_inv_scale = rad_to_deg / cd_det;

    let cd = [
        [cd22 * cd_inv_scale, -cd12 * cd_inv_scale],
        [-cd21 * cd_inv_scale, cd11 * cd_inv_scale],
    ];

    let new_wcs = Wcs::new(crpix, crval, cd);

    // Compute RMS before (using a simple scale estimate)
    let initial_scale = estimate_scale(&pixels, &sky);
    let rms_before = compute_pixel_rms_simple(&pixels, &sky, crpix, crval, initial_scale);

    // Compute RMS after
    let rms_after = compute_pixel_rms(&pixels.iter().cloned().collect::<Vec<_>>(), &sky, &new_wcs);

    Some(LinearRefineResult {
        wcs: new_wcs,
        rms_before_pixels: rms_before,
        rms_after_pixels: rms_after,
        num_stars_used: a_mat.len(),
    })
}

/// Compute centroid of sky positions.
fn compute_sky_centroid(sky: &[RaDec]) -> Option<RaDec> {
    if sky.is_empty() {
        return None;
    }

    // Convert to 3D vectors and average
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_z = 0.0;

    for pos in sky {
        let cos_dec = pos.dec.cos();
        sum_x += cos_dec * pos.ra.cos();
        sum_y += cos_dec * pos.ra.sin();
        sum_z += pos.dec.sin();
    }

    let n = sky.len() as f64;
    sum_x /= n;
    sum_y /= n;
    sum_z /= n;

    let r = (sum_x * sum_x + sum_y * sum_y + sum_z * sum_z).sqrt();
    if r < 1e-10 {
        return None;
    }

    let dec = (sum_z / r).asin();
    let ra = sum_y.atan2(sum_x);

    Some(RaDec::new(ra, dec).normalize())
}

/// Solve 2x2 least squares: A * [x1, x2]^T = b
fn solve_2x2_lstsq(a_mat: &[[f64; 2]], b: &[f64]) -> Option<(f64, f64)> {
    let n = a_mat.len();
    if n < 2 || b.len() != n {
        return None;
    }

    // Compute A^T * A
    let mut ata = [[0.0; 2]; 2];
    for row in a_mat {
        ata[0][0] += row[0] * row[0];
        ata[0][1] += row[0] * row[1];
        ata[1][0] += row[1] * row[0];
        ata[1][1] += row[1] * row[1];
    }

    // Compute A^T * b
    let mut atb = [0.0; 2];
    for (i, row) in a_mat.iter().enumerate() {
        atb[0] += row[0] * b[i];
        atb[1] += row[1] * b[i];
    }

    // Solve 2x2 system
    let det = ata[0][0] * ata[1][1] - ata[0][1] * ata[1][0];
    if det.abs() < 1e-20 {
        return None;
    }

    let inv_det = 1.0 / det;
    let x1 = (ata[1][1] * atb[0] - ata[0][1] * atb[1]) * inv_det;
    let x2 = (-ata[1][0] * atb[0] + ata[0][0] * atb[1]) * inv_det;

    Some((x1, x2))
}

/// Estimate scale from pixel and sky positions.
fn estimate_scale(pixels: &[(f64, f64)], sky: &[RaDec]) -> f64 {
    // Compare distances between first few pairs
    if pixels.len() < 2 {
        return 60.0 / 3600.0; // Default 1 arcmin/pixel
    }

    let px_dist = ((pixels[0].0 - pixels[1].0).powi(2) + (pixels[0].1 - pixels[1].1).powi(2)).sqrt();

    // Angular distance in radians
    let v0 = sky[0].to_vec3();
    let v1 = sky[1].to_vec3();
    let sky_dist = v0.angle_to(&v1);

    if px_dist > 1.0 {
        sky_dist / px_dist
    } else {
        60.0 / 3600.0 * std::f64::consts::PI / 180.0
    }
}

/// Simple RMS computation for initial estimate.
fn compute_pixel_rms_simple(
    pixels: &[(f64, f64)],
    sky: &[RaDec],
    crpix: (f64, f64),
    crval: RaDec,
    scale: f64,
) -> f64 {
    // Very rough approximation - not used for actual accuracy
    // Just for comparison purposes
    let cos_dec = crval.dec.cos();
    let mut sum_sq = 0.0;

    for ((px, py), sky_pos) in pixels.iter().zip(sky.iter()) {
        let dx = px - crpix.0;
        let dy = py - crpix.1;

        // Approximate predicted pixel from sky
        let dra = sky_pos.ra - crval.ra;
        let ddec = sky_pos.dec - crval.dec;

        let pred_dx = dra * cos_dec / scale;
        let pred_dy = ddec / scale;

        sum_sq += (dx - pred_dx).powi(2) + (dy - pred_dy).powi(2);
    }

    (sum_sq / pixels.len() as f64).sqrt()
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
