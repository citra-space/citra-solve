//! SIP (Simple Imaging Polynomial) distortion model.
//!
//! Implements the FITS SIP convention for polynomial distortion correction.

/// Maximum SIP polynomial order supported.
pub const MAX_SIP_ORDER: usize = 5;

/// SIP distortion model.
///
/// The SIP convention models distortion as polynomial corrections:
/// ```text
/// u = x - CRPIX1
/// v = y - CRPIX2
///
/// f(u,v) = Σ A_i_j * u^i * v^j
/// g(u,v) = Σ B_i_j * u^i * v^j
///
/// u' = u + f(u,v)
/// v' = v + g(u,v)
/// ```
///
/// Where (u', v') are the corrected intermediate pixel coordinates.
#[derive(Debug, Clone)]
pub struct SipDistortion {
    /// Polynomial order for A coefficients.
    pub a_order: usize,
    /// A coefficients: A[i][j] is the coefficient for u^i * v^j.
    pub a: [[f64; MAX_SIP_ORDER + 1]; MAX_SIP_ORDER + 1],

    /// Polynomial order for B coefficients.
    pub b_order: usize,
    /// B coefficients: B[i][j] is the coefficient for u^i * v^j.
    pub b: [[f64; MAX_SIP_ORDER + 1]; MAX_SIP_ORDER + 1],

    /// Polynomial order for AP (inverse) coefficients.
    pub ap_order: usize,
    /// AP coefficients (inverse distortion).
    pub ap: [[f64; MAX_SIP_ORDER + 1]; MAX_SIP_ORDER + 1],

    /// Polynomial order for BP (inverse) coefficients.
    pub bp_order: usize,
    /// BP coefficients (inverse distortion).
    pub bp: [[f64; MAX_SIP_ORDER + 1]; MAX_SIP_ORDER + 1],
}

impl Default for SipDistortion {
    fn default() -> Self {
        Self {
            a_order: 0,
            a: [[0.0; MAX_SIP_ORDER + 1]; MAX_SIP_ORDER + 1],
            b_order: 0,
            b: [[0.0; MAX_SIP_ORDER + 1]; MAX_SIP_ORDER + 1],
            ap_order: 0,
            ap: [[0.0; MAX_SIP_ORDER + 1]; MAX_SIP_ORDER + 1],
            bp_order: 0,
            bp: [[0.0; MAX_SIP_ORDER + 1]; MAX_SIP_ORDER + 1],
        }
    }
}

impl SipDistortion {
    /// Create a new SIP distortion model with the given order.
    pub fn new(order: usize) -> Self {
        let order = order.min(MAX_SIP_ORDER);
        Self {
            a_order: order,
            b_order: order,
            ap_order: order,
            bp_order: order,
            ..Default::default()
        }
    }

    /// Check if distortion is effectively zero.
    pub fn is_zero(&self) -> bool {
        self.a_order == 0 && self.b_order == 0
    }

    /// Apply forward distortion: pixel -> intermediate.
    ///
    /// Given pixel offsets (u, v) from CRPIX, returns corrected (u', v').
    pub fn apply_forward(&self, u: f64, v: f64) -> (f64, f64) {
        let du = self.evaluate_polynomial(&self.a, self.a_order, u, v);
        let dv = self.evaluate_polynomial(&self.b, self.b_order, u, v);
        (u + du, v + dv)
    }

    /// Apply inverse distortion: intermediate -> pixel.
    ///
    /// Given intermediate coordinates (u', v'), returns pixel offsets (u, v).
    pub fn apply_inverse(&self, u_prime: f64, v_prime: f64) -> (f64, f64) {
        // If we have AP/BP coefficients, use them directly
        if self.ap_order > 0 || self.bp_order > 0 {
            let du = self.evaluate_polynomial(&self.ap, self.ap_order, u_prime, v_prime);
            let dv = self.evaluate_polynomial(&self.bp, self.bp_order, u_prime, v_prime);
            return (u_prime + du, v_prime + dv);
        }

        // Otherwise, iterate to find inverse
        self.iterate_inverse(u_prime, v_prime)
    }

    /// Evaluate a polynomial at (u, v).
    fn evaluate_polynomial(
        &self,
        coeffs: &[[f64; MAX_SIP_ORDER + 1]; MAX_SIP_ORDER + 1],
        order: usize,
        u: f64,
        v: f64,
    ) -> f64 {
        let mut result = 0.0;
        let mut u_pow = 1.0;

        for i in 0..=order {
            let mut v_pow = 1.0;
            for j in 0..=(order - i) {
                result += coeffs[i][j] * u_pow * v_pow;
                v_pow *= v;
            }
            u_pow *= u;
        }

        result
    }

    /// Iteratively solve for inverse distortion.
    fn iterate_inverse(&self, u_prime: f64, v_prime: f64) -> (f64, f64) {
        const MAX_ITER: usize = 10;
        const TOLERANCE: f64 = 1e-10;

        let mut u = u_prime;
        let mut v = v_prime;

        for _ in 0..MAX_ITER {
            let (u_test, v_test) = self.apply_forward(u, v);
            let du = u_prime - u_test;
            let dv = v_prime - v_test;

            u += du;
            v += dv;

            if du.abs() < TOLERANCE && dv.abs() < TOLERANCE {
                break;
            }
        }

        (u, v)
    }

    /// Set a coefficient in the A polynomial.
    pub fn set_a(&mut self, i: usize, j: usize, value: f64) {
        if i <= MAX_SIP_ORDER && j <= MAX_SIP_ORDER {
            self.a[i][j] = value;
            self.a_order = self.a_order.max(i + j);
        }
    }

    /// Set a coefficient in the B polynomial.
    pub fn set_b(&mut self, i: usize, j: usize, value: f64) {
        if i <= MAX_SIP_ORDER && j <= MAX_SIP_ORDER {
            self.b[i][j] = value;
            self.b_order = self.b_order.max(i + j);
        }
    }

    /// Set a coefficient in the AP polynomial.
    pub fn set_ap(&mut self, i: usize, j: usize, value: f64) {
        if i <= MAX_SIP_ORDER && j <= MAX_SIP_ORDER {
            self.ap[i][j] = value;
            self.ap_order = self.ap_order.max(i + j);
        }
    }

    /// Set a coefficient in the BP polynomial.
    pub fn set_bp(&mut self, i: usize, j: usize, value: f64) {
        if i <= MAX_SIP_ORDER && j <= MAX_SIP_ORDER {
            self.bp[i][j] = value;
            self.bp_order = self.bp_order.max(i + j);
        }
    }

    /// Generate FITS header keywords for the SIP distortion.
    pub fn to_fits_header(&self) -> String {
        let mut header = String::new();

        if self.a_order > 0 {
            header.push_str(&format!(
                "A_ORDER = {:20} / SIP polynomial order\n",
                self.a_order
            ));
            for i in 0..=self.a_order {
                for j in 0..=(self.a_order - i) {
                    if self.a[i][j].abs() > 1e-20 {
                        header.push_str(&format!("A_{}_{}   = {:20.15E}\n", i, j, self.a[i][j]));
                    }
                }
            }
        }

        if self.b_order > 0 {
            header.push_str(&format!(
                "B_ORDER = {:20} / SIP polynomial order\n",
                self.b_order
            ));
            for i in 0..=self.b_order {
                for j in 0..=(self.b_order - i) {
                    if self.b[i][j].abs() > 1e-20 {
                        header.push_str(&format!("B_{}_{}   = {:20.15E}\n", i, j, self.b[i][j]));
                    }
                }
            }
        }

        if self.ap_order > 0 {
            header.push_str(&format!(
                "AP_ORDER= {:20} / SIP inverse polynomial order\n",
                self.ap_order
            ));
            for i in 0..=self.ap_order {
                for j in 0..=(self.ap_order - i) {
                    if self.ap[i][j].abs() > 1e-20 {
                        header.push_str(&format!("AP_{}_{}  = {:20.15E}\n", i, j, self.ap[i][j]));
                    }
                }
            }
        }

        if self.bp_order > 0 {
            header.push_str(&format!(
                "BP_ORDER= {:20} / SIP inverse polynomial order\n",
                self.bp_order
            ));
            for i in 0..=self.bp_order {
                for j in 0..=(self.bp_order - i) {
                    if self.bp[i][j].abs() > 1e-20 {
                        header.push_str(&format!("BP_{}_{}  = {:20.15E}\n", i, j, self.bp[i][j]));
                    }
                }
            }
        }

        header
    }
}

/// Fit SIP distortion coefficients from matched star pairs.
///
/// Given detected pixel positions, catalog sky positions, and an initial WCS,
/// fit polynomial distortion to minimize residuals.
///
/// # Arguments
/// * `detected_pixels` - Detected star positions (x, y)
/// * `catalog_sky` - Catalog star positions (RaDec)
/// * `wcs` - Initial WCS (without distortion)
/// * `order` - Polynomial order (typically 2-4)
///
/// # Returns
/// SIP distortion coefficients that minimize residuals
pub fn fit_sip_distortion(
    detected_pixels: &[(f64, f64)],
    catalog_sky: &[crate::core::types::RaDec],
    wcs: &super::projection::Wcs,
    order: usize,
) -> SipDistortion {
    if detected_pixels.len() != catalog_sky.len() || detected_pixels.len() < 6 {
        return SipDistortion::default();
    }

    let order = order.min(MAX_SIP_ORDER).max(2);
    let crpix = wcs.crpix();

    // Compute residuals: (predicted - detected) in pixel space
    // We want to find corrections that, when added to detected positions,
    // give us positions that project correctly to sky coordinates
    let mut data: Vec<(f64, f64, f64, f64)> = Vec::new(); // (u, v, residual_x, residual_y)

    for (det, sky) in detected_pixels.iter().zip(catalog_sky.iter()) {
        let (pred_x, pred_y) = wcs.sky_to_pixel(sky);
        let u = det.0 - crpix.0;
        let v = det.1 - crpix.1;
        // Residual: how much we need to shift the detected position
        let res_x = pred_x - det.0;
        let res_y = pred_y - det.1;
        data.push((u, v, res_x, res_y));
    }

    // Build design matrix for polynomial terms (starting at order 2)
    // Terms: u², uv, v², u³, u²v, uv², v³, ...
    // Number of terms for orders 2..=order: Σ(i+1) for i in 2..=order = (order-1)*order/2 + (order-1)
    let num_terms = count_sip_terms(order);
    let n = data.len();

    if n < num_terms {
        return SipDistortion::default();
    }

    // Normalize coordinates to improve conditioning of higher-order terms.
    let coord_scale = data
        .iter()
        .fold(0.0f64, |acc, (u, v, _, _)| acc.max(u.abs()).max(v.abs()))
        .max(64.0);

    // Build design matrix A and target vectors b_x, b_y
    let mut a_mat = vec![vec![0.0; num_terms]; n];
    let mut b_x = vec![0.0; n];
    let mut b_y = vec![0.0; n];

    for (row, (u, v, res_x, res_y)) in data.iter().enumerate() {
        b_x[row] = *res_x;
        b_y[row] = *res_y;

        // Fill polynomial terms (normalized coordinates)
        let u_n = *u / coord_scale;
        let v_n = *v / coord_scale;
        let mut col = 0;
        for total_order in 2..=order {
            for i in 0..=total_order {
                let j = total_order - i;
                let term = u_n.powi(i as i32) * v_n.powi(j as i32);
                a_mat[row][col] = term;
                col += 1;
            }
        }
    }

    // Solve least squares: A^T A x = A^T b using normal equations
    let coeffs_x = solve_normal_equations(&a_mat, &b_x, num_terms, 1e-8);
    let coeffs_y = solve_normal_equations(&a_mat, &b_y, num_terms, 1e-8);

    // Build SIP distortion from coefficients
    let mut sip = SipDistortion::new(order);

    let mut col = 0;
    for total_order in 2..=order {
        for i in 0..=total_order {
            let j = total_order - i;
            let den = coord_scale.powi((i + j) as i32);
            if let Some(c) = coeffs_x.as_ref() {
                sip.set_a(i, j, c[col] / den);
            }
            if let Some(c) = coeffs_y.as_ref() {
                sip.set_b(i, j, c[col] / den);
            }
            col += 1;
        }
    }

    // Compute inverse coefficients (AP, BP) by fitting reverse residuals
    fit_inverse_coefficients(&mut sip, detected_pixels, catalog_sky, wcs);

    // Keep SIP only when it materially improves matched-point residuals.
    let rms_before = sip_fit_rms(detected_pixels, catalog_sky, wcs, None);
    let rms_after = sip_fit_rms(detected_pixels, catalog_sky, wcs, Some(&sip));
    if !rms_after.is_finite() || rms_after >= rms_before * 0.995 {
        return SipDistortion::default();
    }

    sip
}

/// Count number of SIP polynomial terms for orders 2 through max_order.
fn count_sip_terms(max_order: usize) -> usize {
    let mut count = 0;
    for order in 2..=max_order {
        count += order + 1;
    }
    count
}

/// Solve normal equations A^T A x = A^T b using Cholesky-like approach.
fn solve_normal_equations(
    a_mat: &[Vec<f64>],
    b: &[f64],
    num_vars: usize,
    ridge_rel: f64,
) -> Option<Vec<f64>> {
    let n = a_mat.len();
    if n == 0 || num_vars == 0 {
        return None;
    }

    // Compute A^T A
    let mut ata = vec![vec![0.0; num_vars]; num_vars];
    for i in 0..num_vars {
        for j in 0..num_vars {
            let mut sum = 0.0;
            for row in 0..n {
                sum += a_mat[row][i] * a_mat[row][j];
            }
            ata[i][j] = sum;
        }
    }

    // Diagonal regularization for numerical stability.
    let mean_diag = ata
        .iter()
        .enumerate()
        .map(|(i, row)| row[i].abs())
        .sum::<f64>()
        / num_vars as f64;
    let ridge = (mean_diag * ridge_rel).max(1e-14);
    for (i, row) in ata.iter_mut().enumerate().take(num_vars) {
        row[i] += ridge;
    }

    // Compute A^T b
    let mut atb = vec![0.0; num_vars];
    for i in 0..num_vars {
        let mut sum = 0.0;
        for row in 0..n {
            sum += a_mat[row][i] * b[row];
        }
        atb[i] = sum;
    }

    // Solve using Gaussian elimination with partial pivoting
    gaussian_solve(&mut ata, &mut atb)
}

/// Gaussian elimination with partial pivoting.
fn gaussian_solve(a: &mut [Vec<f64>], b: &mut [f64]) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 || b.len() != n {
        return None;
    }

    // Forward elimination
    for col in 0..n {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..n {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }

        if max_val < 1e-15 {
            return None; // Singular matrix
        }

        if max_row != col {
            a.swap(col, max_row);
            b.swap(col, max_row);
        }

        // Eliminate below
        for row in (col + 1)..n {
            let factor = a[row][col] / a[col][col];
            a[row][col] = 0.0;
            for j in (col + 1)..n {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }

    Some(x)
}

/// Fit inverse SIP coefficients (AP, BP) for sky->pixel direction.
fn fit_inverse_coefficients(
    sip: &mut SipDistortion,
    detected_pixels: &[(f64, f64)],
    catalog_sky: &[crate::core::types::RaDec],
    wcs: &super::projection::Wcs,
) {
    let order = sip.a_order.max(sip.b_order);
    if order < 2 {
        return;
    }

    let crpix = wcs.crpix();
    let num_terms = count_sip_terms(order);
    let n = detected_pixels.len();

    if n < num_terms {
        return;
    }

    // For inverse: given intermediate coordinates (u', v'), find correction to get (u, v)
    // u' = u + A(u,v), so u = u' - A(u,v) ≈ u' + AP(u', v')
    let mut data: Vec<(f64, f64, f64, f64)> = Vec::new();

    for (det, _sky) in detected_pixels.iter().zip(catalog_sky.iter()) {
        let u = det.0 - crpix.0;
        let v = det.1 - crpix.1;

        // Apply forward distortion to get u', v'
        let (u_prime, v_prime) = sip.apply_forward(u, v);

        // The inverse correction: what we need to add to u' to get u
        let corr_x = u - u_prime;
        let corr_y = v - v_prime;

        data.push((u_prime, v_prime, corr_x, corr_y));
    }

    // Build design matrix
    let mut a_mat = vec![vec![0.0; num_terms]; n];
    let mut b_x = vec![0.0; n];
    let mut b_y = vec![0.0; n];

    for (row, (u, v, res_x, res_y)) in data.iter().enumerate() {
        b_x[row] = *res_x;
        b_y[row] = *res_y;

        let mut col = 0;
        for total_order in 2..=order {
            for i in 0..=total_order {
                let j = total_order - i;
                let term = u.powi(i as i32) * v.powi(j as i32);
                a_mat[row][col] = term;
                col += 1;
            }
        }
    }

    let coord_scale = data
        .iter()
        .fold(0.0f64, |acc, (u, v, _, _)| acc.max(u.abs()).max(v.abs()))
        .max(64.0);

    for (row, (u, v, _, _)) in data.iter().enumerate() {
        let u_n = *u / coord_scale;
        let v_n = *v / coord_scale;
        let mut col = 0;
        for total_order in 2..=order {
            for i in 0..=total_order {
                let j = total_order - i;
                a_mat[row][col] = u_n.powi(i as i32) * v_n.powi(j as i32);
                col += 1;
            }
        }
    }

    let coeffs_x = solve_normal_equations(&a_mat, &b_x, num_terms, 1e-8);
    let coeffs_y = solve_normal_equations(&a_mat, &b_y, num_terms, 1e-8);

    let mut col = 0;
    for total_order in 2..=order {
        for i in 0..=total_order {
            let j = total_order - i;
            let den = coord_scale.powi((i + j) as i32);
            if let Some(c) = coeffs_x.as_ref() {
                sip.set_ap(i, j, c[col] / den);
            }
            if let Some(c) = coeffs_y.as_ref() {
                sip.set_bp(i, j, c[col] / den);
            }
            col += 1;
        }
    }
}

fn sip_fit_rms(
    detected_pixels: &[(f64, f64)],
    catalog_sky: &[crate::core::types::RaDec],
    wcs: &super::projection::Wcs,
    sip: Option<&SipDistortion>,
) -> f64 {
    if detected_pixels.is_empty() || detected_pixels.len() != catalog_sky.len() {
        return f64::INFINITY;
    }

    let crpix = wcs.crpix();
    let mut sum_sq = 0.0;
    for (det, sky) in detected_pixels.iter().zip(catalog_sky.iter()) {
        let (pred_x, pred_y) = wcs.sky_to_pixel(sky);
        let (corr_x, corr_y) = if let Some(s) = sip {
            let u = det.0 - crpix.0;
            let v = det.1 - crpix.1;
            let (u_corr, v_corr) = s.apply_forward(u, v);
            (crpix.0 + u_corr, crpix.1 + v_corr)
        } else {
            *det
        };
        let dx = pred_x - corr_x;
        let dy = pred_y - corr_y;
        sum_sq += dx * dx + dy * dy;
    }
    (sum_sq / detected_pixels.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sip_zero() {
        let sip = SipDistortion::default();
        assert!(sip.is_zero());

        let (u, v) = sip.apply_forward(100.0, 200.0);
        assert!((u - 100.0).abs() < 1e-10);
        assert!((v - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_sip_simple_distortion() {
        let mut sip = SipDistortion::new(2);

        // Add some simple distortion
        sip.set_a(2, 0, 1e-6); // Quadratic in u
        sip.set_b(0, 2, 1e-6); // Quadratic in v

        let (u_prime, v_prime) = sip.apply_forward(100.0, 100.0);

        // Should have some distortion
        assert!((u_prime - 100.0).abs() > 1e-10);
        assert!((v_prime - 100.0).abs() > 1e-10);

        // Inverse should recover original (approximately)
        // Note: iterative inversion may have small residuals
        let (u_back, v_back) = sip.apply_inverse(u_prime, v_prime);
        assert!(
            (u_back - 100.0).abs() < 0.1,
            "u_back={}, expected 100.0",
            u_back
        );
        assert!(
            (v_back - 100.0).abs() < 0.1,
            "v_back={}, expected 100.0",
            v_back
        );
    }

    #[test]
    fn test_sip_fits_header() {
        let mut sip = SipDistortion::new(2);
        sip.set_a(2, 0, 1e-6);
        sip.set_b(0, 2, 1e-6);

        let header = sip.to_fits_header();
        assert!(header.contains("A_ORDER"));
        assert!(header.contains("A_2_0"));
        assert!(header.contains("B_ORDER"));
        assert!(header.contains("B_0_2"));
    }

    #[test]
    fn test_count_sip_terms() {
        // Order 2: u², uv, v² = 3 terms
        assert_eq!(count_sip_terms(2), 3);
        // Order 3: adds u³, u²v, uv², v³ = 4 more terms = 7 total
        assert_eq!(count_sip_terms(3), 7);
        // Order 4: adds u⁴, u³v, u²v², uv³, v⁴ = 5 more terms = 12 total
        assert_eq!(count_sip_terms(4), 12);
    }

    #[test]
    fn test_fit_sip_with_known_distortion() {
        use super::super::projection::Wcs;
        use crate::core::types::RaDec;

        // Create a WCS at center of image
        let wcs = Wcs::new(
            (512.0, 512.0),
            RaDec::from_degrees(180.0, 45.0),
            [[1.0 / 3600.0, 0.0], [0.0, 1.0 / 3600.0]], // 1 arcsec/pixel
        );

        // Create synthetic star positions with known quadratic distortion
        // True distortion: dx = 1e-5 * u², dy = 1e-5 * v²
        let mut detected_pixels = Vec::new();
        let mut catalog_sky = Vec::new();

        for i in -3..=3 {
            for j in -3..=3 {
                // Undistorted pixel positions (relative to center)
                let u = i as f64 * 100.0;
                let v = j as f64 * 100.0;

                // Apply distortion to get "detected" positions
                let distortion_x = 1e-5 * u * u;
                let distortion_y = 1e-5 * v * v;
                let x_det = 512.0 + u + distortion_x;
                let y_det = 512.0 + v + distortion_y;

                // The catalog sky position is where it would be without distortion
                let sky = wcs.pixel_to_sky(512.0 + u, 512.0 + v);

                detected_pixels.push((x_det, y_det));
                catalog_sky.push(sky);
            }
        }

        // Fit SIP distortion
        let sip = fit_sip_distortion(&detected_pixels, &catalog_sky, &wcs, 2);

        // Check that we recovered approximately the right coefficients
        // Note: sign is inverted because SIP corrects *away* the distortion
        // The distortion we added was +1e-5*u², so A_2_0 should be ~-1e-5
        let a_2_0 = sip.a[2][0];
        let b_0_2 = sip.b[0][2];

        assert!(
            (a_2_0 + 1e-5).abs() < 1e-6,
            "A_2_0 = {}, expected ~-1e-5",
            a_2_0
        );
        assert!(
            (b_0_2 + 1e-5).abs() < 1e-6,
            "B_0_2 = {}, expected ~-1e-5",
            b_0_2
        );
    }
}
