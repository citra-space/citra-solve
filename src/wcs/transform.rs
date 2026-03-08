//! Coordinate transformation utilities.

use super::distortion::SipDistortion;
use super::projection::Wcs;
use crate::core::types::RaDec;

/// Extended WCS with optional distortion.
#[derive(Debug, Clone)]
pub struct WcsWithDistortion {
    /// Base WCS (TAN projection).
    pub wcs: Wcs,
    /// Optional SIP distortion.
    pub distortion: Option<SipDistortion>,
}

impl WcsWithDistortion {
    /// Create a new WCS without distortion.
    pub fn new(wcs: Wcs) -> Self {
        Self {
            wcs,
            distortion: None,
        }
    }

    /// Create a new WCS with distortion.
    pub fn with_distortion(wcs: Wcs, distortion: SipDistortion) -> Self {
        Self {
            wcs,
            distortion: Some(distortion),
        }
    }

    /// Transform pixel coordinates to sky coordinates.
    pub fn pixel_to_sky(&self, x: f64, y: f64) -> RaDec {
        let (u, v) = if let Some(ref dist) = self.distortion {
            // Apply forward distortion
            let u = x - self.wcs.crpix().0;
            let v = y - self.wcs.crpix().1;
            let (u_prime, v_prime) = dist.apply_forward(u, v);
            (u_prime + self.wcs.crpix().0, v_prime + self.wcs.crpix().1)
        } else {
            (x, y)
        };

        self.wcs.pixel_to_sky(u, v)
    }

    /// Transform sky coordinates to pixel coordinates.
    pub fn sky_to_pixel(&self, radec: &RaDec) -> (f64, f64) {
        let (x, y) = self.wcs.sky_to_pixel(radec);

        if let Some(ref dist) = self.distortion {
            // Apply inverse distortion
            let u_prime = x - self.wcs.crpix().0;
            let v_prime = y - self.wcs.crpix().1;
            let (u, v) = dist.apply_inverse(u_prime, v_prime);
            (u + self.wcs.crpix().0, v + self.wcs.crpix().1)
        } else {
            (x, y)
        }
    }

    /// Get pixel scale in arcseconds per pixel.
    pub fn pixel_scale_arcsec(&self) -> f64 {
        self.wcs.pixel_scale_arcsec()
    }

    /// Get rotation in degrees.
    pub fn rotation_deg(&self) -> f64 {
        self.wcs.rotation_deg()
    }

    /// Generate FITS header.
    pub fn to_fits_header(&self) -> String {
        let mut header = self.wcs.to_fits_header();

        if let Some(ref dist) = self.distortion {
            if !dist.is_zero() {
                // Update CTYPE to indicate SIP
                header = header.replace("RA---TAN", "RA---TAN-SIP");
                header = header.replace("DEC--TAN", "DEC--TAN-SIP");
                header.push_str(&dist.to_fits_header());
            }
        }

        header
    }
}

/// Transform a list of pixel coordinates to sky coordinates.
pub fn pixels_to_sky(wcs: &Wcs, pixels: &[(f64, f64)]) -> Vec<RaDec> {
    pixels
        .iter()
        .map(|(x, y)| wcs.pixel_to_sky(*x, *y))
        .collect()
}

/// Transform a list of sky coordinates to pixel coordinates.
pub fn sky_to_pixels(wcs: &Wcs, sky: &[RaDec]) -> Vec<(f64, f64)> {
    sky.iter().map(|radec| wcs.sky_to_pixel(radec)).collect()
}

/// Compute residuals between detected and predicted positions.
pub fn compute_residuals(wcs: &Wcs, detected: &[(f64, f64)], catalog: &[RaDec]) -> Vec<(f64, f64)> {
    detected
        .iter()
        .zip(catalog.iter())
        .map(|((det_x, det_y), cat_radec)| {
            let (pred_x, pred_y) = wcs.sky_to_pixel(cat_radec);
            (det_x - pred_x, det_y - pred_y)
        })
        .collect()
}

/// Compute RMS of residuals in pixels.
pub fn residual_rms(residuals: &[(f64, f64)]) -> f64 {
    if residuals.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = residuals.iter().map(|(dx, dy)| dx * dx + dy * dy).sum();
    (sum_sq / residuals.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wcs_with_distortion_no_distortion() {
        let wcs = Wcs::new(
            (512.0, 512.0),
            RaDec::from_degrees(180.0, 45.0),
            [[0.001, 0.0], [0.0, 0.001]],
        );
        let wcs_dist = WcsWithDistortion::new(wcs.clone());

        // Should behave identically to base WCS
        let sky1 = wcs.pixel_to_sky(100.0, 100.0);
        let sky2 = wcs_dist.pixel_to_sky(100.0, 100.0);

        assert!((sky1.ra - sky2.ra).abs() < 1e-10);
        assert!((sky1.dec - sky2.dec).abs() < 1e-10);
    }

    #[test]
    fn test_residual_rms() {
        let residuals = vec![(3.0, 4.0), (0.0, 0.0)];
        let rms = residual_rms(&residuals);
        // sqrt((25 + 0) / 2) = sqrt(12.5) ≈ 3.54
        assert!((rms - (12.5f64).sqrt()).abs() < 1e-10);
    }
}
