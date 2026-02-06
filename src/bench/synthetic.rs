//! Synthetic star field generation for testing.

use crate::core::types::{RaDec, Vec3, DetectedStar};
use crate::core::math;
use crate::wcs::Wcs;
use crate::catalog::Index;

use std::f64::consts::PI;

/// A synthetic star field for testing.
#[derive(Debug, Clone)]
pub struct SyntheticField {
    /// Center of the field.
    pub center: RaDec,
    /// Field of view in degrees.
    pub fov_deg: f64,
    /// Image dimensions.
    pub width: u32,
    pub height: u32,
    /// Detected stars (with noise applied).
    pub detected_stars: Vec<DetectedStar>,
    /// Ground truth WCS.
    pub true_wcs: Wcs,
    /// Indices of catalog stars that should be visible.
    pub visible_catalog_indices: Vec<u32>,
}

/// Configuration for synthetic field generation.
#[derive(Debug, Clone)]
pub struct SyntheticConfig {
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Gaussian noise in pixels (sigma).
    pub position_noise_pixels: f64,
    /// Fraction of true stars to drop (0.0 - 1.0).
    pub missing_star_rate: f64,
    /// Number of false stars to add.
    pub false_star_count: usize,
    /// Minimum star flux for detection.
    pub min_flux: f64,
    /// Rotation angle in degrees.
    pub rotation_deg: f64,
}

impl Default for SyntheticConfig {
    fn default() -> Self {
        Self {
            seed: 12345,
            position_noise_pixels: 0.5,
            missing_star_rate: 0.0,
            false_star_count: 0,
            min_flux: 10.0,
            rotation_deg: 0.0,
        }
    }
}

impl SyntheticConfig {
    /// Create a "noisy" configuration for robustness testing.
    pub fn noisy() -> Self {
        Self {
            position_noise_pixels: 1.5,
            missing_star_rate: 0.1,
            false_star_count: 5,
            ..Default::default()
        }
    }

    /// Create a "challenging" configuration.
    pub fn challenging() -> Self {
        Self {
            position_noise_pixels: 2.0,
            missing_star_rate: 0.25,
            false_star_count: 20,
            ..Default::default()
        }
    }
}

/// Simple LCG random number generator for reproducibility.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Gaussian random number (Box-Muller transform).
    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-10);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    fn next_bool(&mut self, probability: f64) -> bool {
        self.next_f64() < probability
    }
}

/// Generate a synthetic field from a catalog.
pub fn generate_field(
    index: &Index,
    center: RaDec,
    fov_deg: f64,
    width: u32,
    height: u32,
    config: &SyntheticConfig,
) -> SyntheticField {
    let mut rng = SimpleRng::new(config.seed);

    // Create WCS for this field
    let pixel_scale_deg = fov_deg / width.max(height) as f64;
    let rotation_rad = config.rotation_deg.to_radians();
    let cos_r = rotation_rad.cos();
    let sin_r = rotation_rad.sin();

    let cd = [
        [-pixel_scale_deg * cos_r, pixel_scale_deg * sin_r],
        [pixel_scale_deg * sin_r, pixel_scale_deg * cos_r],
    ];

    let crpix = (width as f64 / 2.0, height as f64 / 2.0);
    let wcs = Wcs::new(crpix, center, cd);

    // Find all catalog stars in the field
    let fov_radius_rad = (fov_deg * 1.5).to_radians(); // Extra margin
    let mut visible_stars = Vec::new();

    for (idx, star) in index.stars() {
        let pos = star.to_radec();
        let sep = math::angular_separation(&center, &pos);

        if sep < fov_radius_rad {
            // Project to pixel coordinates
            let (px, py) = wcs.sky_to_pixel(&pos);

            // Check if within image bounds (with margin)
            if px >= -10.0 && px < width as f64 + 10.0 && py >= -10.0 && py < height as f64 + 10.0 {
                // Flux based on magnitude (higher flux = brighter = lower mag)
                let flux = 10.0f64.powf((7.0 - star.magnitude() as f64) / 2.5) * 100.0;

                if flux >= config.min_flux {
                    visible_stars.push((idx, px, py, flux));
                }
            }
        }
    }

    // Sort by brightness (flux descending)
    visible_stars.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

    // Generate detected stars with noise and dropouts
    let mut detected_stars = Vec::new();
    let mut visible_indices = Vec::new();

    for (idx, px, py, flux) in visible_stars {
        // Random dropout
        if config.missing_star_rate > 0.0 && rng.next_bool(config.missing_star_rate) {
            continue;
        }

        // Add position noise
        let noise_x = rng.next_gaussian() * config.position_noise_pixels;
        let noise_y = rng.next_gaussian() * config.position_noise_pixels;

        let noisy_x = px + noise_x;
        let noisy_y = py + noise_y;

        // Skip if noise pushed it out of bounds
        if noisy_x < 0.0 || noisy_x >= width as f64 || noisy_y < 0.0 || noisy_y >= height as f64 {
            continue;
        }

        // Add flux noise (~5%)
        let flux_noise = 1.0 + rng.next_gaussian() * 0.05;
        let noisy_flux = (flux * flux_noise).max(config.min_flux);

        detected_stars.push(DetectedStar::new(noisy_x, noisy_y, noisy_flux));
        visible_indices.push(idx);
    }

    // Add false stars (noise, hot pixels, etc.)
    for _ in 0..config.false_star_count {
        let x = rng.next_f64() * width as f64;
        let y = rng.next_f64() * height as f64;
        // False stars tend to be faint
        let flux = config.min_flux + rng.next_f64() * config.min_flux * 2.0;

        detected_stars.push(DetectedStar::new(x, y, flux));
    }

    // Re-sort by brightness (false stars mixed in)
    detected_stars.sort_by(|a, b| {
        b.flux
            .partial_cmp(&a.flux)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    SyntheticField {
        center,
        fov_deg,
        width,
        height,
        detected_stars,
        true_wcs: wcs,
        visible_catalog_indices: visible_indices,
    }
}

/// Generate a random sky position.
pub fn random_sky_position(seed: u64) -> RaDec {
    let mut rng = SimpleRng::new(seed);
    let ra = rng.next_f64() * 2.0 * PI;
    let dec = (rng.next_f64() * 2.0 - 1.0).asin();
    RaDec::new(ra, dec)
}

/// Generate multiple test fields at different sky positions.
pub fn generate_test_suite(
    index: &Index,
    num_fields: usize,
    fov_deg: f64,
    width: u32,
    height: u32,
    config: &SyntheticConfig,
) -> Vec<SyntheticField> {
    let mut fields = Vec::with_capacity(num_fields);

    for i in 0..num_fields {
        let center = random_sky_position(config.seed.wrapping_add(i as u64 * 1000));
        let field_config = SyntheticConfig {
            seed: config.seed.wrapping_add(i as u64),
            ..*config
        };
        fields.push(generate_field(index, center, fov_deg, width, height, &field_config));
    }

    fields
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_rng() {
        let mut rng = SimpleRng::new(42);
        let v1 = rng.next_f64();
        let v2 = rng.next_f64();
        assert!(v1 >= 0.0 && v1 < 1.0);
        assert!(v2 >= 0.0 && v2 < 1.0);
        assert_ne!(v1, v2);

        // Test reproducibility
        let mut rng2 = SimpleRng::new(42);
        assert_eq!(rng2.next_f64(), v1);
    }

    #[test]
    fn test_random_sky_position() {
        let pos1 = random_sky_position(1);
        let pos2 = random_sky_position(2);

        // Should be valid coordinates
        assert!(pos1.ra >= 0.0 && pos1.ra < 2.0 * PI);
        assert!(pos1.dec >= -PI / 2.0 && pos1.dec <= PI / 2.0);

        // Different seeds should give different positions
        assert!((pos1.ra - pos2.ra).abs() > 0.001 || (pos1.dec - pos2.dec).abs() > 0.001);
    }
}
