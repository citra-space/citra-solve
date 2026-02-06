//! TAN (gnomonic) projection and WCS implementation.

use crate::core::types::RaDec;
use std::f64::consts::PI;

/// World Coordinate System transformation.
///
/// Implements the FITS WCS standard with TAN (gnomonic) projection.
#[derive(Debug, Clone)]
pub struct Wcs {
    /// Reference pixel coordinates (CRPIX1, CRPIX2).
    crpix: (f64, f64),
    /// Reference sky coordinates (CRVAL1, CRVAL2) in radians.
    crval: RaDec,
    /// CD matrix (2x2): transforms pixel offsets to intermediate world coordinates.
    /// CD = [[CD1_1, CD1_2], [CD2_1, CD2_2]]
    cd: [[f64; 2]; 2],
    /// Inverse CD matrix (precomputed for efficiency).
    cd_inv: [[f64; 2]; 2],
}

impl Wcs {
    /// Create a new WCS with the given parameters.
    ///
    /// # Arguments
    /// * `crpix` - Reference pixel (1-indexed in FITS, but we use 0-indexed here)
    /// * `crval` - Reference sky position
    /// * `cd` - CD matrix in degrees per pixel
    pub fn new(crpix: (f64, f64), crval: RaDec, cd: [[f64; 2]; 2]) -> Self {
        let cd_inv = invert_2x2(&cd);
        Self {
            crpix,
            crval,
            cd,
            cd_inv,
        }
    }

    /// Get the reference pixel.
    #[inline]
    pub fn crpix(&self) -> (f64, f64) {
        self.crpix
    }

    /// Get the reference sky coordinate.
    #[inline]
    pub fn crval(&self) -> RaDec {
        self.crval
    }

    /// Get the CD matrix.
    #[inline]
    pub fn cd(&self) -> &[[f64; 2]; 2] {
        &self.cd
    }

    /// Create a new WCS with updated CRVAL.
    pub fn with_crval(&self, crval: RaDec) -> Self {
        Self::new(self.crpix, crval, self.cd)
    }

    /// Create a new WCS with updated CRPIX.
    pub fn with_crpix(&self, crpix: (f64, f64)) -> Self {
        Self::new(crpix, self.crval, self.cd)
    }

    /// Create a new WCS with updated CD matrix.
    pub fn with_cd(&self, cd: [[f64; 2]; 2]) -> Self {
        Self::new(self.crpix, self.crval, cd)
    }

    /// Transform pixel coordinates to sky coordinates.
    pub fn pixel_to_sky(&self, x: f64, y: f64) -> RaDec {
        // Pixel offset from reference
        let dx = x - self.crpix.0;
        let dy = y - self.crpix.1;

        // Apply CD matrix to get intermediate world coordinates (in degrees)
        let xi_deg = self.cd[0][0] * dx + self.cd[0][1] * dy;
        let eta_deg = self.cd[1][0] * dx + self.cd[1][1] * dy;

        // Convert to radians
        let xi = xi_deg.to_radians();
        let eta = eta_deg.to_radians();

        // TAN (gnomonic) deprojection
        self.deproject_tan(xi, eta)
    }

    /// Transform sky coordinates to pixel coordinates.
    pub fn sky_to_pixel(&self, radec: &RaDec) -> (f64, f64) {
        // TAN projection
        let (xi, eta) = self.project_tan(radec);

        // Convert to degrees
        let xi_deg = xi.to_degrees();
        let eta_deg = eta.to_degrees();

        // Apply inverse CD matrix
        let dx = self.cd_inv[0][0] * xi_deg + self.cd_inv[0][1] * eta_deg;
        let dy = self.cd_inv[1][0] * xi_deg + self.cd_inv[1][1] * eta_deg;

        // Add reference pixel
        (self.crpix.0 + dx, self.crpix.1 + dy)
    }

    /// TAN (gnomonic) projection: sky to intermediate coordinates.
    ///
    /// Returns (xi, eta) in radians. For points behind the projection plane
    /// (more than 90° from tangent point), returns very large values that will
    /// project far outside any reasonable image.
    fn project_tan(&self, radec: &RaDec) -> (f64, f64) {
        let ra0 = self.crval.ra;
        let dec0 = self.crval.dec;
        let ra = radec.ra;
        let dec = radec.dec;

        let cos_dec = dec.cos();
        let sin_dec = dec.sin();
        let cos_dec0 = dec0.cos();
        let sin_dec0 = dec0.sin();
        let cos_dra = (ra - ra0).cos();
        let sin_dra = (ra - ra0).sin();

        // Denominator (cosine of angular distance from reference)
        let denom = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_dra;

        // Point is behind the projection plane (>90° from tangent point)
        // Return very large values to ensure it projects far outside image
        if denom < 1e-10 {
            return (1e10, 1e10);
        }

        // Standard TAN projection
        let xi = cos_dec * sin_dra / denom;
        let eta = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_dra) / denom;

        (xi, eta)
    }

    /// TAN (gnomonic) deprojection: intermediate coordinates to sky.
    fn deproject_tan(&self, xi: f64, eta: f64) -> RaDec {
        let ra0 = self.crval.ra;
        let dec0 = self.crval.dec;

        let cos_dec0 = dec0.cos();
        let sin_dec0 = dec0.sin();

        // Radial distance from center
        let rho = (xi * xi + eta * eta).sqrt();

        // Handle center point
        if rho < 1e-10 {
            return self.crval;
        }

        // Angular distance from center
        let c = rho.atan();
        let cos_c = c.cos();
        let sin_c = c.sin();

        // Declination
        let dec = (cos_c * sin_dec0 + eta * sin_c * cos_dec0 / rho).asin();

        // Right ascension
        let ra = ra0 + (xi * sin_c).atan2(rho * cos_dec0 * cos_c - eta * sin_dec0 * sin_c);

        RaDec::new(ra, dec).normalize()
    }

    /// Get the pixel scale in arcseconds per pixel (average of X and Y).
    pub fn pixel_scale_arcsec(&self) -> f64 {
        let scale_x = (self.cd[0][0].powi(2) + self.cd[1][0].powi(2)).sqrt();
        let scale_y = (self.cd[0][1].powi(2) + self.cd[1][1].powi(2)).sqrt();
        ((scale_x + scale_y) / 2.0) * 3600.0 // degrees to arcsec
    }

    /// Get the rotation angle in degrees (position angle of +Y axis).
    pub fn rotation_deg(&self) -> f64 {
        // Rotation is the angle of the Y-axis in sky coordinates
        let angle = self.cd[0][1].atan2(self.cd[1][1]);
        angle.to_degrees()
    }

    /// Generate a FITS WCS header string.
    pub fn to_fits_header(&self) -> String {
        let mut header = String::new();

        // WCS type
        header.push_str("WCSAXES =                    2 / Number of WCS axes\n");
        header.push_str("CTYPE1  = 'RA---TAN'           / TAN projection\n");
        header.push_str("CTYPE2  = 'DEC--TAN'           / TAN projection\n");

        // Reference pixel (FITS uses 1-indexed)
        header.push_str(&format!(
            "CRPIX1  = {:20.10} / Reference pixel X\n",
            self.crpix.0 + 1.0
        ));
        header.push_str(&format!(
            "CRPIX2  = {:20.10} / Reference pixel Y\n",
            self.crpix.1 + 1.0
        ));

        // Reference coordinates (in degrees)
        header.push_str(&format!(
            "CRVAL1  = {:20.10} / Reference RA (deg)\n",
            self.crval.ra_deg()
        ));
        header.push_str(&format!(
            "CRVAL2  = {:20.10} / Reference Dec (deg)\n",
            self.crval.dec_deg()
        ));

        // CD matrix
        header.push_str(&format!(
            "CD1_1   = {:20.15} / CD matrix element\n",
            self.cd[0][0]
        ));
        header.push_str(&format!(
            "CD1_2   = {:20.15} / CD matrix element\n",
            self.cd[0][1]
        ));
        header.push_str(&format!(
            "CD2_1   = {:20.15} / CD matrix element\n",
            self.cd[1][0]
        ));
        header.push_str(&format!(
            "CD2_2   = {:20.15} / CD matrix element\n",
            self.cd[1][1]
        ));

        header
    }
}

/// Invert a 2x2 matrix.
fn invert_2x2(m: &[[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if det.abs() < 1e-20 {
        // Return identity on singular matrix
        return [[1.0, 0.0], [0.0, 1.0]];
    }
    let inv_det = 1.0 / det;
    [
        [m[1][1] * inv_det, -m[0][1] * inv_det],
        [-m[1][0] * inv_det, m[0][0] * inv_det],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_4;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_wcs_roundtrip() {
        // Create a simple WCS at RA=180, Dec=45, scale=1"/pixel
        let scale = 1.0 / 3600.0; // 1 arcsec per pixel in degrees
        let wcs = Wcs::new(
            (512.0, 512.0),
            RaDec::from_degrees(180.0, 45.0),
            [[scale, 0.0], [0.0, scale]],
        );

        // Test roundtrip at various points
        let test_points = [
            (512.0, 512.0), // Center
            (0.0, 0.0),     // Corner
            (1024.0, 0.0),
            (1024.0, 1024.0),
            (0.0, 1024.0),
            (256.0, 768.0), // Random
        ];

        for (x, y) in test_points {
            let sky = wcs.pixel_to_sky(x, y);
            let (x2, y2) = wcs.sky_to_pixel(&sky);
            assert!(
                approx_eq(x, x2, 1e-6),
                "X mismatch: {} vs {} for ({}, {})",
                x,
                x2,
                x,
                y
            );
            assert!(
                approx_eq(y, y2, 1e-6),
                "Y mismatch: {} vs {} for ({}, {})",
                y,
                y2,
                x,
                y
            );
        }
    }

    #[test]
    fn test_pixel_scale() {
        let scale = 2.5 / 3600.0; // 2.5 arcsec per pixel
        let wcs = Wcs::new(
            (512.0, 512.0),
            RaDec::from_degrees(0.0, 0.0),
            [[scale, 0.0], [0.0, scale]],
        );

        let pixel_scale = wcs.pixel_scale_arcsec();
        assert!(approx_eq(pixel_scale, 2.5, 1e-6));
    }

    #[test]
    fn test_fits_header_generation() {
        let wcs = Wcs::new(
            (512.0, 512.0),
            RaDec::from_degrees(180.0, 45.0),
            [[-0.001, 0.0], [0.0, 0.001]],
        );

        let header = wcs.to_fits_header();
        assert!(header.contains("CTYPE1"));
        assert!(header.contains("CRPIX1"));
        assert!(header.contains("CRVAL1"));
        assert!(header.contains("CD1_1"));
    }

    #[test]
    fn test_invert_2x2() {
        let m = [[2.0, 1.0], [1.0, 1.0]];
        let inv = invert_2x2(&m);

        // m * inv should be identity
        let i00 = m[0][0] * inv[0][0] + m[0][1] * inv[1][0];
        let i01 = m[0][0] * inv[0][1] + m[0][1] * inv[1][1];
        let i10 = m[1][0] * inv[0][0] + m[1][1] * inv[1][0];
        let i11 = m[1][0] * inv[0][1] + m[1][1] * inv[1][1];

        assert!(approx_eq(i00, 1.0, 1e-10));
        assert!(approx_eq(i01, 0.0, 1e-10));
        assert!(approx_eq(i10, 0.0, 1e-10));
        assert!(approx_eq(i11, 1.0, 1e-10));
    }
}
