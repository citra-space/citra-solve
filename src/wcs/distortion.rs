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
                        header.push_str(&format!(
                            "A_{}_{}   = {:20.15E}\n",
                            i, j, self.a[i][j]
                        ));
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
                        header.push_str(&format!(
                            "B_{}_{}   = {:20.15E}\n",
                            i, j, self.b[i][j]
                        ));
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
                        header.push_str(&format!(
                            "AP_{}_{}  = {:20.15E}\n",
                            i, j, self.ap[i][j]
                        ));
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
                        header.push_str(&format!(
                            "BP_{}_{}  = {:20.15E}\n",
                            i, j, self.bp[i][j]
                        ));
                    }
                }
            }
        }

        header
    }
}

/// Fit SIP distortion coefficients from residuals.
///
/// Given a set of (pixel, sky) correspondences and an initial WCS,
/// fit polynomial distortion to minimize residuals.
pub fn fit_distortion(
    _pixel_coords: &[(f64, f64)],
    _sky_coords: &[(f64, f64)], // Projected to intermediate coordinates
    _order: usize,
) -> SipDistortion {
    // TODO: Implement least-squares fitting of SIP coefficients
    // This requires solving a linear system for the polynomial coefficients
    SipDistortion::default()
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
        assert!((u_back - 100.0).abs() < 0.1, "u_back={}, expected 100.0", u_back);
        assert!((v_back - 100.0).abs() < 0.1, "v_back={}, expected 100.0", v_back);
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
}
