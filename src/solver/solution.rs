//! Solution representation and output.

use crate::core::types::{RaDec, CatalogStar};
use crate::wcs::Wcs;

/// A successful plate solve solution.
#[derive(Debug, Clone)]
pub struct Solution {
    /// The WCS transformation.
    pub wcs: Wcs,
    /// Center of the image in sky coordinates.
    pub center: RaDec,
    /// Field of view width in degrees.
    pub fov_width_deg: f64,
    /// Field of view height in degrees.
    pub fov_height_deg: f64,
    /// Rotation angle (position angle of +Y axis) in degrees.
    pub rotation_deg: f64,
    /// Pixel scale in arcseconds per pixel.
    pub pixel_scale_arcsec: f64,
    /// RMS residual of the fit in arcseconds.
    pub rms_arcsec: f64,
    /// Number of stars matched.
    pub num_matched_stars: usize,
    /// Log-odds confidence measure.
    pub log_odds: f64,
    /// Matched stars (detected index, catalog star).
    pub matched_stars: Vec<(usize, CatalogStar)>,
}

impl Solution {
    /// Create a new solution from WCS and metadata.
    pub fn new(
        wcs: Wcs,
        image_width: u32,
        image_height: u32,
        rms_arcsec: f64,
        log_odds: f64,
        matched_stars: Vec<(usize, CatalogStar)>,
    ) -> Self {
        let center = wcs.pixel_to_sky(image_width as f64 / 2.0, image_height as f64 / 2.0);
        let pixel_scale = wcs.pixel_scale_arcsec();
        let fov_width_deg = image_width as f64 * pixel_scale / 3600.0;
        let fov_height_deg = image_height as f64 * pixel_scale / 3600.0;
        let rotation_deg = wcs.rotation_deg();

        Self {
            wcs,
            center,
            fov_width_deg,
            fov_height_deg,
            rotation_deg,
            pixel_scale_arcsec: pixel_scale,
            rms_arcsec,
            num_matched_stars: matched_stars.len(),
            log_odds,
            matched_stars,
        }
    }

    /// Get the sky position of a pixel.
    pub fn pixel_to_sky(&self, x: f64, y: f64) -> RaDec {
        self.wcs.pixel_to_sky(x, y)
    }

    /// Get the pixel position of a sky coordinate.
    pub fn sky_to_pixel(&self, radec: &RaDec) -> (f64, f64) {
        self.wcs.sky_to_pixel(radec)
    }

    /// Generate a FITS WCS header.
    pub fn to_fits_header(&self) -> String {
        self.wcs.to_fits_header()
    }

    /// Get the corner coordinates of the image.
    pub fn corners(&self, image_width: u32, image_height: u32) -> [RaDec; 4] {
        [
            self.wcs.pixel_to_sky(0.0, 0.0),
            self.wcs.pixel_to_sky(image_width as f64, 0.0),
            self.wcs.pixel_to_sky(image_width as f64, image_height as f64),
            self.wcs.pixel_to_sky(0.0, image_height as f64),
        ]
    }
}

impl std::fmt::Display for Solution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Plate Solve Solution:")?;
        writeln!(
            f,
            "  Center: RA={:.6}° Dec={:.6}°",
            self.center.ra_deg(),
            self.center.dec_deg()
        )?;
        writeln!(
            f,
            "  FOV: {:.3}° x {:.3}°",
            self.fov_width_deg, self.fov_height_deg
        )?;
        writeln!(f, "  Rotation: {:.2}°", self.rotation_deg)?;
        writeln!(f, "  Scale: {:.4}\"/pixel", self.pixel_scale_arcsec)?;
        writeln!(f, "  RMS: {:.2}\"", self.rms_arcsec)?;
        writeln!(f, "  Matched stars: {}", self.num_matched_stars)?;
        writeln!(f, "  Log-odds: {:.1}", self.log_odds)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solution_display() {
        let wcs = Wcs::new(
            (512.0, 384.0),
            RaDec::from_degrees(180.0, 45.0),
            [[-0.001, 0.0], [0.0, 0.001]],
        );
        let solution = Solution::new(wcs, 1024, 768, 1.5, 25.0, vec![]);
        let display = format!("{}", solution);
        assert!(display.contains("Plate Solve Solution"));
        assert!(display.contains("Center"));
    }
}
