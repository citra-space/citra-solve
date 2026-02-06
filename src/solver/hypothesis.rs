//! Match hypothesis generation and initial WCS estimation.

use crate::core::types::{DetectedStar, CatalogStar};
use crate::catalog::Index;
use crate::pattern::{Quad, PatternMatch};
use crate::wcs::Wcs;

/// A match hypothesis: a proposed correspondence between detected and catalog stars.
#[derive(Debug, Clone)]
pub struct Hypothesis {
    /// Mapping from detected star index to catalog star.
    pub star_matches: Vec<(usize, CatalogStar)>,
    /// Initial WCS solution based on the quad match.
    pub wcs: Wcs,
    /// Ratio distance of the original pattern match.
    pub pattern_distance: f64,
    /// Log-odds from verification (set during verification).
    pub log_odds: f64,
}

impl Hypothesis {
    /// Create a new unverified hypothesis.
    pub fn new(star_matches: Vec<(usize, CatalogStar)>, wcs: Wcs, pattern_distance: f64) -> Self {
        Self {
            star_matches,
            wcs,
            pattern_distance,
            log_odds: f64::NEG_INFINITY,
        }
    }
}

/// All 24 permutations of 4 elements.
const PERMUTATIONS_4: [[usize; 4]; 24] = [
    [0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1], [0, 3, 1, 2], [0, 3, 2, 1],
    [1, 0, 2, 3], [1, 0, 3, 2], [1, 2, 0, 3], [1, 2, 3, 0], [1, 3, 0, 2], [1, 3, 2, 0],
    [2, 0, 1, 3], [2, 0, 3, 1], [2, 1, 0, 3], [2, 1, 3, 0], [2, 3, 0, 1], [2, 3, 1, 0],
    [3, 0, 1, 2], [3, 0, 2, 1], [3, 1, 0, 2], [3, 1, 2, 0], [3, 2, 0, 1], [3, 2, 1, 0],
];

/// Generate hypotheses from pattern matches.
///
/// Since both detected and catalog quads have sorted ratios (losing vertex correspondence),
/// we try all 24 permutations and keep the one with the best WCS fit.
pub fn generate_hypotheses(
    detected_stars: &[DetectedStar],
    quads: &[Quad],
    matches: &[PatternMatch],
    index: &Index,
    image_width: u32,
    image_height: u32,
) -> Vec<Hypothesis> {
    let mut hypotheses = Vec::new();
    let image_diagonal = ((image_width as f64).powi(2) + (image_height as f64).powi(2)).sqrt();

    for pattern_match in matches {
        let quad = &quads[pattern_match.detected_quad_idx];
        let cat_pattern = &pattern_match.catalog_pattern;

        // Get catalog stars for this pattern
        let cat_star_indices = cat_pattern.stars();
        let mut cat_stars = Vec::with_capacity(4);
        for &idx in &cat_star_indices {
            if let Ok(star) = index.get_catalog_star(idx as u32) {
                cat_stars.push(star);
            }
        }

        if cat_stars.len() != 4 {
            continue;
        }

        // Try all 24 permutations and find the best one
        let mut best_wcs: Option<Wcs> = None;
        let mut best_residual = f64::MAX;
        let mut best_matches: Vec<(usize, CatalogStar)> = Vec::new();

        for perm in &PERMUTATIONS_4 {
            // Create star matches with this permutation
            let star_matches: Vec<(usize, CatalogStar)> = quad
                .star_indices
                .iter()
                .enumerate()
                .map(|(i, &det_idx)| (det_idx, cat_stars[perm[i]]))
                .collect();

            // Estimate WCS
            if let Some(wcs) = estimate_wcs_from_matches(
                detected_stars,
                &star_matches,
                image_width,
                image_height,
            ) {
                // Check if pixel scale is reasonable
                // For the index FOV range, expect reasonable scales
                let scale_arcsec = wcs.pixel_scale_arcsec();
                let fov_deg = scale_arcsec * image_diagonal / 3600.0;

                // Skip if FOV is unreasonable (< 1° or > 60°)
                if fov_deg < 1.0 || fov_deg > 60.0 {
                    continue;
                }

                // Compute residual: how well do the matched stars fit the WCS?
                let residual = compute_match_residual(detected_stars, &star_matches, &wcs);

                if residual < best_residual {
                    best_residual = residual;
                    best_wcs = Some(wcs);
                    best_matches = star_matches;
                }
            }
        }

        // Add the best hypothesis if we found one
        if let Some(wcs) = best_wcs {
            hypotheses.push(Hypothesis::new(
                best_matches,
                wcs,
                pattern_match.ratio_distance,
            ));
        }
    }

    hypotheses
}

/// Compute RMS residual for star matches given a WCS.
fn compute_match_residual(
    detected_stars: &[DetectedStar],
    matches: &[(usize, CatalogStar)],
    wcs: &Wcs,
) -> f64 {
    let mut sum_sq = 0.0;
    for (det_idx, cat_star) in matches {
        let det = &detected_stars[*det_idx];
        let (pred_x, pred_y) = wcs.sky_to_pixel(&cat_star.position);
        let dx = det.x - pred_x;
        let dy = det.y - pred_y;
        sum_sq += dx * dx + dy * dy;
    }
    (sum_sq / matches.len() as f64).sqrt()
}

/// Estimate WCS from matched star pairs using linear least-squares.
///
/// This uses a reference star approach:
/// 1. Pick first matched star as reference (crpix = pixel pos, crval = sky pos)
/// 2. Compute tangent plane coordinates (xi, eta) for other stars relative to crval
/// 3. Solve for CD matrix using least squares fit
/// 4. Shift the WCS so crpix is at image center for better overall fit
fn estimate_wcs_from_matches(
    detected_stars: &[DetectedStar],
    matches: &[(usize, CatalogStar)],
    image_width: u32,
    image_height: u32,
) -> Option<Wcs> {
    if matches.len() < 3 {
        return None;
    }

    // Use first matched star as reference
    let (ref_det_idx, ref_cat) = &matches[0];
    let ref_det = &detected_stars[*ref_det_idx];
    let crpix = (ref_det.x, ref_det.y);
    let crval = ref_cat.position;

    // Precompute trig values for crval (ra and dec are already in radians)
    let crval_ra = crval.ra;
    let crval_dec = crval.dec;
    let sin_crval_dec = crval_dec.sin();
    let cos_crval_dec = crval_dec.cos();

    // Build matrices for least squares: solve for CD matrix
    // For each matched star (except reference):
    //   dx = x - crpix1 = CD11*xi + CD12*eta
    //   dy = y - crpix2 = CD21*xi + CD22*eta
    //
    // We solve two separate least squares problems:
    //   [xi_1, eta_1]   [CD11]   [dx_1]
    //   [xi_2, eta_2] * [CD12] = [dx_2]
    //   ...                      ...
    //
    //   [xi_1, eta_1]   [CD21]   [dy_1]
    //   [xi_2, eta_2] * [CD22] = [dy_2]
    //   ...                      ...

    let mut a_mat: Vec<[f64; 2]> = Vec::new(); // [xi, eta] rows
    let mut b_x: Vec<f64> = Vec::new();  // dx values
    let mut b_y: Vec<f64> = Vec::new();  // dy values

    for (det_idx, cat_star) in matches.iter().skip(1) {
        let det = &detected_stars[*det_idx];

        // Pixel offset from reference
        let dx = det.x - crpix.0;
        let dy = det.y - crpix.1;

        // Compute tangent plane coordinates (xi, eta) for this star
        let ra = cat_star.position.ra;
        let dec = cat_star.position.dec;

        let sin_dec = dec.sin();
        let cos_dec = dec.cos();
        let delta_ra = ra - crval_ra;
        let sin_delta_ra = delta_ra.sin();
        let cos_delta_ra = delta_ra.cos();

        let denom = sin_crval_dec * sin_dec + cos_crval_dec * cos_dec * cos_delta_ra;

        // Skip if star is near or behind tangent plane
        if denom < 0.1 {
            continue;
        }

        // Standard coordinates (radians)
        let xi = (cos_dec * sin_delta_ra) / denom;
        let eta = (cos_crval_dec * sin_dec - sin_crval_dec * cos_dec * cos_delta_ra) / denom;

        a_mat.push([xi, eta]);
        b_x.push(dx);
        b_y.push(dy);
    }

    if a_mat.len() < 2 {
        return None;
    }

    // Solve least squares using normal equations: A^T * A * x = A^T * b
    // For a 2xN system, this is small enough to solve directly

    // Compute A^T * A (2x2 matrix)
    let mut ata = [[0.0; 2]; 2];
    for row in &a_mat {
        ata[0][0] += row[0] * row[0];
        ata[0][1] += row[0] * row[1];
        ata[1][0] += row[1] * row[0];
        ata[1][1] += row[1] * row[1];
    }

    // Compute A^T * b_x and A^T * b_y
    let mut atb_x = [0.0; 2];
    let mut atb_y = [0.0; 2];
    for (i, row) in a_mat.iter().enumerate() {
        atb_x[0] += row[0] * b_x[i];
        atb_x[1] += row[1] * b_x[i];
        atb_y[0] += row[0] * b_y[i];
        atb_y[1] += row[1] * b_y[i];
    }

    // Solve 2x2 system using Cramer's rule
    let det = ata[0][0] * ata[1][1] - ata[0][1] * ata[1][0];
    if det.abs() < 1e-20 {
        return None;
    }

    // Inverse of 2x2 matrix
    let inv_det = 1.0 / det;
    let ata_inv = [
        [ata[1][1] * inv_det, -ata[0][1] * inv_det],
        [-ata[1][0] * inv_det, ata[0][0] * inv_det],
    ];

    // CD matrix first row: maps tangent plane to pixel x
    let cd11 = ata_inv[0][0] * atb_x[0] + ata_inv[0][1] * atb_x[1];
    let cd12 = ata_inv[1][0] * atb_x[0] + ata_inv[1][1] * atb_x[1];

    // CD matrix second row: maps tangent plane to pixel y
    let cd21 = ata_inv[0][0] * atb_y[0] + ata_inv[0][1] * atb_y[1];
    let cd22 = ata_inv[1][0] * atb_y[0] + ata_inv[1][1] * atb_y[1];

    // CD matrix is in pixels per radian, but WCS uses degrees
    // The xi, eta values we computed are in radians
    // CD should convert radians to pixels, but WCS convention is degrees to pixels
    // So we need: CD_deg = CD_rad / (pi/180) = CD_rad * 180/pi
    // Actually wait - WCS CD matrix converts from degrees to pixels
    // Our current CD converts from radians to pixels
    // So we need to multiply by (pi/180) to go from rad/px to deg/px? No...
    //
    // Standard WCS: (x - crpix) = CD * (xi_deg, eta_deg)  where xi, eta in degrees
    // Our computation: (x - crpix) = CD_computed * (xi_rad, eta_rad)
    // So CD_wcs = CD_computed * (180/pi)

    // Actually, let me reconsider. In WCS:
    // x - crpix1 = CD11 * xi + CD12 * eta
    // where CD is in units of pixels/degree and xi, eta in degrees
    //
    // We computed: dx = CD11_computed * xi_rad + CD12_computed * eta_rad
    // where xi_rad, eta_rad are in radians
    //
    // To convert: xi_deg = xi_rad * (180/pi)
    // So: dx = CD11_computed * xi_rad = CD11_computed * xi_deg / (180/pi)
    //        = (CD11_computed / (180/pi)) * xi_deg
    //        = CD11_wcs * xi_deg
    // Therefore: CD_wcs = CD_computed / (180/pi) = CD_computed * pi/180
    //
    // Wait no - WCS CD is in degrees/pixel, not pixels/degree!
    // The WCS transform is: (xi, eta)_degrees = CD * (x - crpix, y - crpix)
    // So CD has units of degrees/pixel
    //
    // But our solve gives us pixel = CD_computed * tangent_radians
    // So CD_computed has units of pixels/radian
    //
    // To get WCS CD (degrees/pixel), we need to invert and convert:
    // tangent_radians = CD_computed^{-1} * pixel
    // tangent_degrees = tangent_radians * (180/pi)
    // tangent_degrees = CD_computed^{-1} * (180/pi) * pixel
    // So CD_wcs = CD_computed^{-1} * (180/pi)
    //
    // Let's compute the inverse of our 2x2 CD matrix
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

    Some(Wcs::new(crpix, crval, cd))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests will be added when needed
}
