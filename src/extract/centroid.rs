//! Centroid extraction for star detection.

use crate::core::types::DetectedStar;
use image::DynamicImage;
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
    /// Minimum peak contrast above local annulus background.
    pub min_peak_contrast: f64,
    /// Minimum RMS spot radius (pixels) to reject hot pixels.
    pub min_rms_radius: f64,
    /// Maximum RMS spot radius (pixels) to reject broad cloud blobs.
    pub max_rms_radius: f64,
    /// Maximum spot eccentricity (0=circular, 1=line-like).
    pub max_eccentricity: f64,
    /// Minimum normalized peak sharpness relative to local contrast.
    pub min_peak_sharpness: f64,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            sigma_threshold: 5.5,
            min_separation: 10.0,
            max_stars: 40,
            centroid_radius: 5,
            min_flux: 110.0,
            min_peak_contrast: 6.0,
            min_rms_radius: 0.6,
            max_rms_radius: 3.8,
            max_eccentricity: 0.9,
            min_peak_sharpness: 0.12,
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
    let mut candidates: Vec<(u32, u32, f64, f64)> = Vec::new();
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
                // Local contrast gate: reject broad cloud structures and gradients.
                let local_bg = local_annulus_mean(
                    &gray,
                    x as i32,
                    y as i32,
                    config.centroid_radius as i32 + 2,
                    config.centroid_radius as i32 + 6,
                );
                let contrast = val - local_bg;
                if contrast < config.min_peak_contrast {
                    continue;
                }
                let sharpness = local_peak_sharpness(&gray, x, y, val, local_bg);
                if sharpness < config.min_peak_sharpness {
                    continue;
                }
                candidates.push((x, y, val, contrast));
            }
        }
    }

    // Sort by brightness (descending)
    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    // Extract centroids with non-maximum suppression
    let mut stars: Vec<DetectedStar> = Vec::new();
    let sep_sq = config.min_separation * config.min_separation;

    for (px, py, _peak, contrast) in candidates {
        // Compute intensity-weighted centroid
        let (cx, cy, flux, rms_radius, eccentricity) =
            compute_centroid(&gray, px, py, config.centroid_radius, background);

        if flux < config.min_flux {
            continue;
        }
        if rms_radius < config.min_rms_radius || rms_radius > config.max_rms_radius {
            continue;
        }
        if eccentricity > config.max_eccentricity {
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

        // Quality-weighted brightness helps prioritize star-like centroids
        // over broad cloud structures when selecting top stars for solving.
        let compactness = 1.0 / (1.0 + 0.7 * rms_radius * rms_radius);
        let roundness = (1.0 - 0.4 * eccentricity * eccentricity).max(0.2);
        let contrast_boost = (1.0 + contrast / 20.0).min(2.5);
        let quality_flux = flux * compactness * roundness * contrast_boost;

        stars.push(DetectedStar::new(cx, cy, quality_flux));

        if stars.len() >= config.max_stars {
            break;
        }
    }

    // Sort by flux (brightest first)
    stars.sort_by(|a, b| b.flux.partial_cmp(&a.flux).unwrap());

    Ok(stars)
}

/// Compute a normalized peak sharpness score in [~0, 1+].
///
/// Broad cloud structures tend to have low sharpness while true star peaks
/// usually stand out from their immediate 4-neighborhood.
fn local_peak_sharpness(img: &image::GrayImage, x: u32, y: u32, peak: f64, local_bg: f64) -> f64 {
    if x == 0 || y == 0 || x + 1 >= img.width() || y + 1 >= img.height() {
        return 0.0;
    }
    let n1 = img.get_pixel(x - 1, y).0[0] as f64;
    let n2 = img.get_pixel(x + 1, y).0[0] as f64;
    let n3 = img.get_pixel(x, y - 1).0[0] as f64;
    let n4 = img.get_pixel(x, y + 1).0[0] as f64;
    let neighbors_mean = (n1 + n2 + n3 + n4) * 0.25;
    let denom = (peak - local_bg).max(1.0);
    ((peak - neighbors_mean) / denom).max(0.0)
}

/// Compute local annulus mean around (cx, cy).
fn local_annulus_mean(img: &image::GrayImage, cx: i32, cy: i32, r_in: i32, r_out: i32) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    let r_in2 = (r_in * r_in) as i64;
    let r_out2 = (r_out * r_out) as i64;

    for dy in -r_out..=r_out {
        for dx in -r_out..=r_out {
            let x = cx + dx;
            let y = cy + dy;
            if x < 0 || y < 0 || x >= img.width() as i32 || y >= img.height() as i32 {
                continue;
            }
            let d2 = (dx as i64 * dx as i64) + (dy as i64 * dy as i64);
            if d2 >= r_in2 && d2 <= r_out2 {
                sum += img.get_pixel(x as u32, y as u32).0[0] as f64;
                count += 1;
            }
        }
    }

    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
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
) -> (f64, f64, f64, f64, f64) {
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

    if sum_w <= 0.0 {
        return (px as f64, py as f64, 0.0, f64::INFINITY, 1.0);
    }

    let cx = sum_x / sum_w;
    let cy = sum_y / sum_w;

    // Second pass: RMS radius around centroid for compactness filtering.
    let mut sum_r2_w = 0.0;
    let mut sum_x2_w = 0.0;
    let mut sum_y2_w = 0.0;
    let mut sum_xy_w = 0.0;
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
            if weight <= 0.0 {
                continue;
            }

            let dx = x as f64 - cx;
            let dy = y as f64 - cy;
            sum_r2_w += (dx * dx + dy * dy) * weight;
            sum_x2_w += dx * dx * weight;
            sum_y2_w += dy * dy * weight;
            sum_xy_w += dx * dy * weight;
        }
    }

    let rms_radius = (sum_r2_w / sum_w).sqrt();
    let mxx = sum_x2_w / sum_w;
    let myy = sum_y2_w / sum_w;
    let mxy = sum_xy_w / sum_w;
    let trace = mxx + myy;
    let det = (mxx * myy - mxy * mxy).max(0.0);
    let disc = (0.25 * trace * trace - det).max(0.0).sqrt();
    let l1 = (0.5 * trace + disc).max(0.0);
    let l2 = (0.5 * trace - disc).max(0.0);
    let eccentricity = if l1 > 1e-12 {
        (1.0 - (l2 / l1)).max(0.0).sqrt()
    } else {
        0.0
    };

    (cx, cy, sum_w, rms_radius, eccentricity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extraction_config_default() {
        let config = ExtractionConfig::default();
        assert_eq!(config.sigma_threshold, 5.5);
        assert_eq!(config.min_separation, 10.0);
        assert_eq!(config.max_stars, 40);
        assert_eq!(config.min_flux, 110.0);
        assert_eq!(config.min_peak_contrast, 6.0);
        assert_eq!(config.min_rms_radius, 0.6);
        assert_eq!(config.max_rms_radius, 3.8);
        assert_eq!(config.max_eccentricity, 0.9);
        assert_eq!(config.min_peak_sharpness, 0.12);
    }
}
