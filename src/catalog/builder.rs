//! Index builder for creating pattern databases from star catalogs.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use bytemuck::bytes_of;

use super::error::CatalogError;
use super::index::{IndexHeader, DEFAULT_HASH_BINS_PER_DIM};
use super::star::{PackedPattern, PackedStar};
use crate::core::types::{RaDec, Vec3};

/// Configuration for building an index.
#[derive(Debug, Clone)]
pub struct BuildConfig {
    /// Minimum FOV in degrees (patterns smaller than this are excluded).
    pub fov_min_deg: f32,
    /// Maximum FOV in degrees (patterns larger than this are excluded).
    pub fov_max_deg: f32,
    /// Magnitude limit (exclude stars fainter than this).
    pub mag_limit: f32,
    /// Number of hash bins.
    pub num_bins: u32,
    /// Maximum number of stars to use for pattern generation.
    pub max_stars: usize,
    /// Maximum patterns per star (to limit index size).
    pub max_patterns_per_star: usize,
    /// Number of logarithmic angular scale bands.
    pub num_scale_bands: u8,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            fov_min_deg: 5.0,
            fov_max_deg: 30.0,
            mag_limit: 8.0,
            num_bins: 1_000_000, // 1M bins
            max_stars: 8000,
            // Memory-safe medium-density default tuned for high recall without
            // pathological bin density or multi-gigabyte indices.
            max_patterns_per_star: 1500,
            num_scale_bands: 8,
        }
    }
}

/// A star during index building (before packing).
#[derive(Debug, Clone)]
pub struct BuildStar {
    /// Original catalog ID (e.g., Hipparcos number).
    pub catalog_id: u32,
    /// Position.
    pub position: RaDec,
    /// Visual magnitude.
    pub magnitude: f32,
}

/// Index builder.
pub struct IndexBuilder {
    config: BuildConfig,
    stars: Vec<BuildStar>,
}

impl IndexBuilder {
    /// Create a new index builder with the given configuration.
    pub fn new(config: BuildConfig) -> Self {
        Self {
            config,
            stars: Vec::new(),
        }
    }

    /// Add a star to the catalog.
    pub fn add_star(&mut self, catalog_id: u32, ra: f64, dec: f64, magnitude: f32) {
        if magnitude <= self.config.mag_limit {
            self.stars.push(BuildStar {
                catalog_id,
                position: RaDec::new(ra, dec),
                magnitude,
            });
        }
    }

    /// Add multiple stars.
    pub fn add_stars(&mut self, stars: impl Iterator<Item = (u32, f64, f64, f32)>) {
        for (id, ra, dec, mag) in stars {
            self.add_star(id, ra, dec, mag);
        }
    }

    /// Build the index and write it to a file.
    pub fn build<P: AsRef<Path>>(&mut self, output_path: P) -> Result<BuildStats, CatalogError> {
        // Sort stars by brightness
        self.stars.sort_by(|a, b| {
            a.magnitude
                .partial_cmp(&b.magnitude)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to max_stars
        if self.stars.len() > self.config.max_stars {
            self.stars.truncate(self.config.max_stars);
        }

        let num_stars = self.stars.len();
        println!("Building index with {} stars", num_stars);

        // Precompute unit vectors
        let unit_vectors: Vec<Vec3> = self.stars.iter().map(|s| s.position.to_vec3()).collect();

        // Generate patterns and bin them
        let fov_min_rad = (self.config.fov_min_deg as f64).to_radians();
        let fov_max_rad = (self.config.fov_max_deg as f64).to_radians();

        let mut bins: HashMap<u32, Vec<PackedPattern>> = HashMap::new();
        let mut total_patterns = 0u64;
        let mut patterns_per_star = vec![0usize; num_stars];

        // Generate quads from combinations of 4 stars
        // Build neighbor lists first: for each star, stars within FOV distance
        println!("Generating patterns...");
        println!("  Building neighbor lists...");

        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); num_stars];
        for i in 0..num_stars {
            for j in (i + 1)..num_stars {
                let d = unit_vectors[i].angle_to(&unit_vectors[j]);
                if d <= fov_max_rad * 1.2 {
                    neighbors[i].push(j);
                    neighbors[j].push(i);
                }
            }
            if i % 1000 == 0 {
                println!("    Processed {}/{} stars", i, num_stars);
            }
        }

        // Generate patterns from neighbor combinations.
        //
        // Strategy: For each primary star i, iterate over j neighbors with
        // a stride to ensure DIVERSE j values are sampled (not just the nearest).
        // For each (i,j) pair, limit the number of (k,l) combinations to avoid
        // exhausting the pattern budget on a single j neighbor.
        println!("  Generating patterns...");
        let max_per_pair = 40; // Max patterns for each (i,j) pair

        for i in 0..num_stars {
            if i % 500 == 0 {
                println!("    Star {}/{}", i, num_stars);
            }

            if patterns_per_star[i] >= self.config.max_patterns_per_star {
                continue;
            }

            // Get neighbors of star i with index > i (to avoid duplicates)
            let mut ni: Vec<usize> = neighbors[i].iter().filter(|&&j| j > i).cloned().collect();
            // Sort by angular distance from i so sampled quads are geometrically
            // local and diverse (tetra3-style robustness to missing/false stars).
            ni.sort_by(|a, b| {
                let da = unit_vectors[i].angle_to(&unit_vectors[*a]);
                let db = unit_vectors[i].angle_to(&unit_vectors[*b]);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
            if ni.len() > 140 {
                let mut reduced = ni[..100].to_vec();
                let tail = &ni[100..];
                let stride = (tail.len() / 40).max(1);
                for idx in tail.iter().step_by(stride).take(40) {
                    reduced.push(*idx);
                }
                ni = reduced;
            }

            if ni.len() < 3 {
                continue;
            }

            let ni_len = ni.len();

            for ji in 0..ni_len {
                let j = ni[ji];
                if patterns_per_star[i] >= self.config.max_patterns_per_star {
                    break;
                }

                let d_ij = unit_vectors[i].angle_to(&unit_vectors[j]);
                if d_ij < fov_min_rad * 0.1 {
                    continue;
                }

                let mut pair_count = 0;

                let k_stride = 1usize;

                let mut ki = ji + 1;
                while ki < ni_len {
                    let k = ni[ki];
                    if k <= j {
                        ki += k_stride;
                        continue;
                    }
                    if patterns_per_star[i] >= self.config.max_patterns_per_star
                        || pair_count >= max_per_pair
                    {
                        break;
                    }

                    let d_ik = unit_vectors[i].angle_to(&unit_vectors[k]);
                    let d_jk = unit_vectors[j].angle_to(&unit_vectors[k]);
                    if d_ik > fov_max_rad * 1.5 || d_jk > fov_max_rad * 1.5 {
                        ki += k_stride;
                        continue;
                    }

                    let l_stride = 1usize;

                    let mut li = ki + 1;
                    while li < ni_len {
                        let l = ni[li];
                        if l <= k {
                            li += l_stride;
                            continue;
                        }
                        if patterns_per_star[i] >= self.config.max_patterns_per_star
                            || pair_count >= max_per_pair
                        {
                            break;
                        }

                        let d_il = unit_vectors[i].angle_to(&unit_vectors[l]);
                        let d_jl = unit_vectors[j].angle_to(&unit_vectors[l]);
                        let d_kl = unit_vectors[k].angle_to(&unit_vectors[l]);

                        let angular_distances = [d_ij, d_ik, d_il, d_jk, d_jl, d_kl];
                        let max_ang = angular_distances.iter().cloned().fold(0.0f64, f64::max);

                        if max_ang >= fov_min_rad && max_ang <= fov_max_rad {
                            // Use tangent-plane (gnomonic) distances for hash ratios.
                            // This matches what a camera with TAN projection produces,
                            // so pixel distance ratios match catalog distance ratios.
                            let centroid = unit_vectors[i]
                                .add(&unit_vectors[j])
                                .add(&unit_vectors[k])
                                .add(&unit_vectors[l])
                                .normalize();

                            let proj_i = unit_vectors[i].gnomonic_project(&centroid);
                            let proj_j = unit_vectors[j].gnomonic_project(&centroid);
                            let proj_k = unit_vectors[k].gnomonic_project(&centroid);
                            let proj_l = unit_vectors[l].gnomonic_project(&centroid);

                            let (proj_i, proj_j, proj_k, proj_l) =
                                match (proj_i, proj_j, proj_k, proj_l) {
                                    (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
                                    _ => {
                                        li += l_stride;
                                        continue;
                                    }
                                };

                            let tp_dist = |a: (f64, f64), b: (f64, f64)| -> f64 {
                                ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
                            };

                            let mut distances = [
                                tp_dist(proj_i, proj_j),
                                tp_dist(proj_i, proj_k),
                                tp_dist(proj_i, proj_l),
                                tp_dist(proj_j, proj_k),
                                tp_dist(proj_j, proj_l),
                                tp_dist(proj_k, proj_l),
                            ];
                            let max_dist = distances.iter().cloned().fold(0.0f64, f64::max);

                            for d in &mut distances {
                                *d /= max_dist;
                            }
                            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

                            // Filter degenerate patterns to reduce hash collisions
                            // and improve robustness to false/missing detections.
                            if distances[0] < 0.04 || distances[1] < 0.07 || distances[2] < 0.10 {
                                li += l_stride;
                                continue;
                            }

                            let ratios: [f64; 5] = [
                                distances[0],
                                distances[1],
                                distances[2],
                                distances[3],
                                distances[4],
                            ];
                            let tetra_signature =
                                crate::pattern::tetra::canonical_tetra_signature(&[
                                    proj_i, proj_j, proj_k, proj_l,
                                ]);
                            let Some(tetra_signature) = tetra_signature else {
                                li += l_stride;
                                continue;
                            };

                            let scale_band = compute_scale_band(
                                max_ang,
                                fov_min_rad,
                                fov_max_rad,
                                self.config.num_scale_bands,
                            );
                            let hash_bin =
                                compute_hash(&ratios, &tetra_signature, self.config.num_bins);
                            let pattern = PackedPattern::new(
                                [i as u16, j as u16, k as u16, l as u16],
                                ratios,
                                tetra_signature,
                                scale_band,
                                max_ang.to_degrees(),
                            );

                            bins.entry(hash_bin).or_default().push(pattern);
                            total_patterns += 1;
                            patterns_per_star[i] += 1;
                            patterns_per_star[j] += 1;
                            patterns_per_star[k] += 1;
                            patterns_per_star[l] += 1;
                            pair_count += 1;
                        }

                        li += l_stride;
                    } // end l loop

                    ki += k_stride;
                } // end k loop
            } // end j loop
        } // end i loop

        println!(
            "Generated {} patterns in {} bins",
            total_patterns,
            bins.len()
        );

        // Write index file
        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);

        // Write header
        let header = IndexHeader::new(
            num_stars as u32,
            total_patterns as u32,
            self.config.num_bins,
            self.config.fov_min_deg,
            self.config.fov_max_deg,
            self.config.mag_limit,
            self.config.num_scale_bands.max(1),
            DEFAULT_HASH_BINS_PER_DIM,
        );
        writer.write_all(bytes_of(&header))?;

        // Write stars
        for star in &self.stars {
            let packed = PackedStar::new(
                star.position.ra,
                star.position.dec,
                star.magnitude,
                (star.catalog_id & 0xFFFF) as u16,
            );
            writer.write_all(bytes_of(&packed))?;
        }

        // Build bin offsets and pattern array
        let mut bin_offsets = vec![0u32; self.config.num_bins as usize + 1];
        let mut all_patterns = Vec::with_capacity(total_patterns as usize);

        let mut current_offset = 0u32;
        for bin in 0..self.config.num_bins {
            bin_offsets[bin as usize] = current_offset;
            if let Some(patterns) = bins.get(&bin) {
                for p in patterns {
                    all_patterns.push(*p);
                }
                current_offset += patterns.len() as u32;
            }
        }
        bin_offsets[self.config.num_bins as usize] = current_offset;

        // Write bin offsets
        for offset in &bin_offsets {
            writer.write_all(&offset.to_le_bytes())?;
        }

        // Write patterns
        for pattern in &all_patterns {
            writer.write_all(bytes_of(pattern))?;
        }

        writer.flush()?;

        let max_patterns_per_bin = bins
            .values()
            .map(|patterns| patterns.len() as u32)
            .max()
            .unwrap_or(0);

        Ok(BuildStats {
            num_stars: num_stars as u32,
            num_patterns: total_patterns as u32,
            num_bins_used: bins.len() as u32,
            max_patterns_per_bin,
            avg_patterns_per_bin: if bins.is_empty() {
                0.0
            } else {
                total_patterns as f64 / bins.len() as f64
            },
        })
    }
}

/// Statistics from index building.
#[derive(Debug)]
pub struct BuildStats {
    pub num_stars: u32,
    pub num_patterns: u32,
    pub num_bins_used: u32,
    pub max_patterns_per_bin: u32,
    pub avg_patterns_per_bin: f64,
}

fn compute_scale_band(max_angle_rad: f64, fov_min_rad: f64, fov_max_rad: f64, num_bands: u8) -> u8 {
    if num_bands <= 1 {
        return 0;
    }

    let min_scale = fov_min_rad.max(1e-6);
    let max_scale = fov_max_rad.max(min_scale * 1.01);
    let angle = max_angle_rad.clamp(min_scale, max_scale);
    let ratio = (angle / min_scale).ln() / (max_scale / min_scale).ln();
    let idx = (ratio.clamp(0.0, 0.999_999) * num_bands as f64).floor() as u8;
    idx.min(num_bands.saturating_sub(1))
}

/// Compute the hash bin for a pattern.
fn compute_hash(ratios: &[f64; 5], tetra_signature: &[f64; 4], num_bins: u32) -> u32 {
    crate::pattern::hash::compute_hash(ratios, tetra_signature, num_bins)
}

/// Compute hash with tolerance, returning all bins that might match.
pub fn compute_hash_with_tolerance(
    ratios: &[f64; 5],
    tetra_signature: &[f64; 4],
    num_bins: u32,
    ratio_tolerance: f64,
    tetra_tolerance: f64,
) -> Vec<u32> {
    crate::pattern::hash::query_hash_bins(
        ratios,
        tetra_signature,
        num_bins,
        ratio_tolerance,
        tetra_tolerance,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_hash_deterministic() {
        let ratios = [0.1, 0.2, 0.3, 0.4, 0.5];
        let tetra = [0.1, 0.2, 0.8, 0.3];
        let h1 = compute_hash(&ratios, &tetra, 1_000_000);
        let h2 = compute_hash(&ratios, &tetra, 1_000_000);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_compute_hash_with_tolerance() {
        let ratios = [0.5, 0.5, 0.5, 0.5, 0.5];
        let tetra = [0.2, 0.2, 0.8, 0.3];
        let hashes = compute_hash_with_tolerance(&ratios, &tetra, 1_000_000, 0.01, 0.02);
        // Should return multiple bins
        assert!(hashes.len() > 1);

        // Original hash should be included
        let original = compute_hash(&ratios, &tetra, 1_000_000);
        assert!(hashes.contains(&original));
    }
}
