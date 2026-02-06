//! Centroid extraction for star detection.

use crate::core::types::DetectedStar;
use image::{DynamicImage, GenericImageView, Luma};
use std::path::Path;

/// Configuration for star extraction.
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    /// Sigma above background for detection threshold
    pub sigma_threshold: f64,
    /// Minimum distance between detected stars (pixels)
    pub min_separation: f64,
    /// Maximum number of stars to return
    pub max_stars: usize,
    /// Radius for centroid computation (pixels)
    pub centroid_radius: usize,
    /// Minimum star flux (after background subtraction)
    pub min_flux: f64,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            sigma_threshold: 5.0,
            min_separation: 10.0,  // Increased to avoid double detections from PSF artifacts
            max_stars: 200,
            centroid_radius: 5,
            min_flux: 100.0,
        }
    }
}

/// Extract stars from an image file.
pub fn extract_stars<P: AsRef<Path>>(
    path: P,
    config: &ExtractionConfig,
) -> Result<Vec<DetectedStar>, String> {
    let img = image::open(path).map_err(|e| format!("Failed to open image: {}", e))?;
    extract_stars_from_image(&img, config)
}

/// Extract stars from a loaded image.
pub fn extract_stars_from_image(
    img: &DynamicImage,
    config: &ExtractionConfig,
) -> Result<Vec<DetectedStar>, String> {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    // Compute background statistics (median and stddev estimate)
    let (background, noise) = estimate_background(&gray);

    // Detection threshold
    let threshold = background + config.sigma_threshold * noise;

    // Find local maxima above threshold
    let mut candidates: Vec<(u32, u32, f64)> = Vec::new();
    let margin = config.centroid_radius as u32;

    for y in margin..(height - margin) {
        for x in margin..(width - margin) {
            let val = gray.get_pixel(x, y).0[0] as f64;

            if val < threshold {
                continue;
            }

            // Check if local maximum in 3x3 neighborhood
            let mut is_max = true;
            'outer: for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = (x as i32 + dx) as u32;
                    let ny = (y as i32 + dy) as u32;
                    if gray.get_pixel(nx, ny).0[0] as f64 >= val {
                        is_max = false;
                        break 'outer;
                    }
                }
            }

            if is_max {
                candidates.push((x, y, val));
            }
        }
    }

    // Sort by brightness (descending)
    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    // Extract centroids with non-maximum suppression
    let mut stars: Vec<DetectedStar> = Vec::new();
    let sep_sq = config.min_separation * config.min_separation;

    for (px, py, _peak) in candidates {
        // Compute intensity-weighted centroid
        let (cx, cy, flux) = compute_centroid(&gray, px, py, config.centroid_radius, background);

        if flux < config.min_flux {
            continue;
        }

        // Check separation from existing stars using CENTROID position
        let too_close = stars.iter().any(|s| {
            let dx = s.x - cx;
            let dy = s.y - cy;
            dx * dx + dy * dy < sep_sq
        });

        if too_close {
            continue;
        }

        stars.push(DetectedStar::new(cx, cy, flux));

        if stars.len() >= config.max_stars {
            break;
        }
    }

    // Sort by flux (brightest first)
    stars.sort_by(|a, b| b.flux.partial_cmp(&a.flux).unwrap());

    Ok(stars)
}

/// Estimate background level and noise using median and MAD.
fn estimate_background(img: &image::GrayImage) -> (f64, f64) {
    let (width, height) = img.dimensions();

    // Sample pixels for efficiency on large images
    let mut samples: Vec<u8> = Vec::with_capacity(10000);
    let step = ((width * height) as usize / 10000).max(1);

    for (i, pixel) in img.pixels().enumerate() {
        if i % step == 0 {
            samples.push(pixel.0[0]);
        }
    }

    samples.sort_unstable();

    // Median
    let median = samples[samples.len() / 2] as f64;

    // MAD (Median Absolute Deviation) for robust noise estimate
    let mut deviations: Vec<f64> = samples.iter().map(|&v| (v as f64 - median).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = deviations[deviations.len() / 2];

    // Convert MAD to standard deviation estimate (factor of 1.4826 for normal distribution)
    let sigma = mad * 1.4826;

    (median, sigma.max(1.0))
}

/// Compute intensity-weighted centroid around a peak.
fn compute_centroid(
    img: &image::GrayImage,
    px: u32,
    py: u32,
    radius: usize,
    background: f64,
) -> (f64, f64, f64) {
    let r = radius as i32;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_w = 0.0;

    for dy in -r..=r {
        for dx in -r..=r {
            let x = px as i32 + dx;
            let y = py as i32 + dy;

            if x < 0 || y < 0 {
                continue;
            }

            let x = x as u32;
            let y = y as u32;

            if x >= img.width() || y >= img.height() {
                continue;
            }

            let val = img.get_pixel(x, y).0[0] as f64;
            let weight = (val - background).max(0.0);

            sum_x += x as f64 * weight;
            sum_y += y as f64 * weight;
            sum_w += weight;
        }
    }

    if sum_w > 0.0 {
        (sum_x / sum_w, sum_y / sum_w, sum_w)
    } else {
        (px as f64, py as f64, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extraction_config_default() {
        let config = ExtractionConfig::default();
        assert_eq!(config.sigma_threshold, 5.0);
        assert_eq!(config.max_stars, 200);
    }
}
