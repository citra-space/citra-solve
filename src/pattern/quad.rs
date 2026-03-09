//! Quad (4-star pattern) extraction from detected stars.

use crate::core::types::DetectedStar;

/// A quad pattern extracted from detected stars.
#[derive(Debug, Clone)]
pub struct Quad {
    /// Indices into the detected star array.
    pub star_indices: [usize; 4],
    /// Normalized edge ratios (5 values, sorted, excluding the longest=1.0).
    pub ratios: [f64; 5],
    /// The longest edge distance (in pixels), used for scale estimation.
    pub max_edge_pixels: f64,
    /// Canonical tetra signature [x1, y1, x2, y2] for collision-resistant matching.
    pub tetra_signature: [f64; 4],
}

impl Quad {
    /// Get the indices of the 4 stars.
    #[inline]
    pub fn indices(&self) -> [usize; 4] {
        self.star_indices
    }

    /// Get the 5 normalized edge ratios.
    #[inline]
    pub fn ratios(&self) -> &[f64; 5] {
        &self.ratios
    }

    /// Get canonical tetra signature values.
    #[inline]
    pub fn tetra(&self) -> &[f64; 4] {
        &self.tetra_signature
    }
}

/// Generate quad patterns from detected stars.
///
/// Stars should be sorted by brightness (brightest first) before calling.
/// Returns up to `max_quads` patterns from the `max_stars` brightest stars.
///
/// Generates all valid quads and then uniformly subsamples to ensure
/// diverse star combinations are covered across the full brightness range.
pub fn generate_quads(stars: &[DetectedStar], max_stars: usize, max_quads: usize) -> Vec<Quad> {
    let n = stars.len().min(max_stars);
    if n < 4 {
        return Vec::new();
    }

    // Precompute pairwise distances
    let mut distances = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = stars[i].distance_to(&stars[j]);
            distances[i][j] = d;
            distances[j][i] = d;
        }
    }

    // Count total valid quads to compute stride
    let total_combinations = combinations_count(n, 4);

    if total_combinations <= max_quads {
        // Generate all quads if the total count is manageable
        let mut quads = Vec::with_capacity(total_combinations);
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    for l in (k + 1)..n {
                        if let Some(quad) =
                            make_quad_from_indices(i, j, k, l, stars, &distances)
                        {
                            quads.push(quad);
                        }
                    }
                }
            }
        }
        return quads;
    }

    // Use deterministic uniform subsampling: enumerate all quads with a stride
    // to get approximately max_quads evenly distributed across the combinatorial space.
    let stride = (total_combinations / max_quads).max(1);
    let mut quads = Vec::with_capacity(max_quads);
    let mut count = 0usize;

    for i in 0..n {
        if quads.len() >= max_quads {
            break;
        }
        for j in (i + 1)..n {
            if quads.len() >= max_quads {
                break;
            }
            for k in (j + 1)..n {
                if quads.len() >= max_quads {
                    break;
                }
                for l in (k + 1)..n {
                    if quads.len() >= max_quads {
                        break;
                    }
                    if count % stride == 0 {
                        if let Some(quad) =
                            make_quad_from_indices(i, j, k, l, stars, &distances)
                        {
                            quads.push(quad);
                        }
                    }
                    count += 1;
                }
            }
        }
    }

    quads
}

/// Compute C(n, k) - number of k-combinations from n elements.
fn combinations_count(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let mut result = 1usize;
    for i in 0..k {
        result = result.saturating_mul(n - i) / (i + 1);
    }
    result
}

/// Generate quads prioritizing patterns with the brightest stars.
///
/// This variant focuses on creating patterns that include the very brightest
/// stars, which are more likely to be matched reliably.
pub fn generate_quads_brightness_priority(
    stars: &[DetectedStar],
    max_stars: usize,
    max_quads: usize,
    min_edge_pixels: f64,
    max_edge_pixels: f64,
) -> Vec<Quad> {
    let n = stars.len().min(max_stars);
    if n < 4 {
        return Vec::new();
    }

    let mut quads = Vec::with_capacity(max_quads);

    // Precompute pairwise distances
    let mut distances = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = stars[i].distance_to(&stars[j]);
            distances[i][j] = d;
            distances[j][i] = d;
        }
    }

    // Generate quads, prioritizing those containing the brightest stars
    // Use a "rings" approach: first try quads from stars 0-3, then 0-7, then 0-15, etc.
    let mut ring_size = 4;
    while ring_size <= n && quads.len() < max_quads {
        for i in 0..ring_size {
            if quads.len() >= max_quads {
                break;
            }
            for j in (i + 1)..ring_size {
                if quads.len() >= max_quads {
                    break;
                }
                for k in (j + 1)..ring_size {
                    if quads.len() >= max_quads {
                        break;
                    }
                    for l in (k + 1)..ring_size {
                        if quads.len() >= max_quads {
                            break;
                        }

                        // Skip if we already processed this combination in a smaller ring
                        if ring_size > 4 {
                            let prev_ring = ring_size / 2;
                            if i < prev_ring && j < prev_ring && k < prev_ring && l < prev_ring {
                                continue;
                            }
                        }

                        if let Some(quad) =
                            make_quad_from_indices(i, j, k, l, stars, &distances)
                        {
                            // Check edge size constraints
                            if quad.max_edge_pixels >= min_edge_pixels
                                && quad.max_edge_pixels <= max_edge_pixels
                            {
                                quads.push(quad);
                            }
                        }
                    }
                }
            }
        }
        ring_size *= 2;
    }

    quads
}

/// Create a quad from 4 star indices.
fn make_quad_from_indices(
    i: usize,
    j: usize,
    k: usize,
    l: usize,
    stars: &[DetectedStar],
    distances: &[Vec<f64>],
) -> Option<Quad> {
    // Get all 6 pairwise distances
    let d_ij = distances[i][j];
    let d_ik = distances[i][k];
    let d_il = distances[i][l];
    let d_jk = distances[j][k];
    let d_jl = distances[j][l];
    let d_kl = distances[k][l];

    let mut edge_distances = [d_ij, d_ik, d_il, d_jk, d_jl, d_kl];

    // Find maximum distance
    let max_dist = edge_distances.iter().cloned().fold(0.0f64, f64::max);

    // Skip degenerate quads
    if max_dist < 1.0 {
        return None;
    }

    // Normalize by max distance
    for d in &mut edge_distances {
        *d /= max_dist;
    }

    // Sort to create rotation/reflection invariant representation
    edge_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Reject highly degenerate quads (near-collinear/duplicate-star geometry).
    // Tetra3-style indexing is significantly more robust with well-conditioned
    // patterns because they hash more uniquely under centroid noise.
    if edge_distances[0] < 0.04 || edge_distances[1] < 0.07 || edge_distances[2] < 0.10 {
        return None;
    }

    // Take first 5 ratios (the 6th is always ~1.0)
    let ratios = [
        edge_distances[0],
        edge_distances[1],
        edge_distances[2],
        edge_distances[3],
        edge_distances[4],
    ];

    let points = [
        (stars[i].x, stars[i].y),
        (stars[j].x, stars[j].y),
        (stars[k].x, stars[k].y),
        (stars[l].x, stars[l].y),
    ];
    let tetra_signature = crate::pattern::tetra::canonical_tetra_signature(&points)?;

    Some(Quad {
        star_indices: [i, j, k, l],
        ratios,
        max_edge_pixels: max_dist,
        tetra_signature,
    })
}

/// Compute the centroid of a quad in pixel coordinates.
pub fn quad_centroid(stars: &[DetectedStar], quad: &Quad) -> (f64, f64) {
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for &idx in &quad.star_indices {
        sum_x += stars[idx].x;
        sum_y += stars[idx].y;
    }
    (sum_x / 4.0, sum_y / 4.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_stars() -> Vec<DetectedStar> {
        vec![
            DetectedStar::new(0.0, 0.0, 1000.0),
            DetectedStar::new(100.0, 0.0, 900.0),
            DetectedStar::new(100.0, 100.0, 800.0),
            DetectedStar::new(0.0, 100.0, 700.0),
            DetectedStar::new(50.0, 50.0, 600.0),
        ]
    }

    #[test]
    fn test_generate_quads_basic() {
        let stars = make_test_stars();
        let quads = generate_quads(&stars, 5, 10);

        // Should generate some quads
        assert!(!quads.is_empty());

        // Each quad should have valid indices
        for quad in &quads {
            for &idx in &quad.star_indices {
                assert!(idx < stars.len());
            }
            // Indices should be unique
            let mut indices = quad.star_indices.to_vec();
            indices.sort();
            indices.dedup();
            assert_eq!(indices.len(), 4);
        }
    }

    #[test]
    fn test_quad_ratios_invariant() {
        let stars = make_test_stars();
        let quads = generate_quads(&stars, 5, 10);

        for quad in &quads {
            // Ratios should be sorted
            for i in 0..4 {
                assert!(quad.ratios[i] <= quad.ratios[i + 1]);
            }
            // Ratios should be in [0, 1]
            for &r in &quad.ratios {
                assert!(r >= 0.0 && r <= 1.0);
            }
        }
    }

    #[test]
    fn test_quad_scale_invariance() {
        // Create two sets of stars at different scales
        let stars1 = vec![
            DetectedStar::new(0.0, 0.0, 100.0),
            DetectedStar::new(10.0, 0.0, 100.0),
            DetectedStar::new(10.0, 10.0, 100.0),
            DetectedStar::new(0.0, 10.0, 100.0),
        ];
        let stars2 = vec![
            DetectedStar::new(0.0, 0.0, 100.0),
            DetectedStar::new(100.0, 0.0, 100.0),
            DetectedStar::new(100.0, 100.0, 100.0),
            DetectedStar::new(0.0, 100.0, 100.0),
        ];

        let quads1 = generate_quads(&stars1, 4, 1);
        let quads2 = generate_quads(&stars2, 4, 1);

        assert_eq!(quads1.len(), 1);
        assert_eq!(quads2.len(), 1);

        // Ratios should be identical (scale invariant)
        for i in 0..5 {
            assert!((quads1[0].ratios[i] - quads2[0].ratios[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_max_quads_limit() {
        let stars = make_test_stars();
        let quads = generate_quads(&stars, 5, 2);
        assert_eq!(quads.len(), 2);
    }

    #[test]
    fn test_too_few_stars() {
        let stars = vec![
            DetectedStar::new(0.0, 0.0, 100.0),
            DetectedStar::new(10.0, 0.0, 100.0),
            DetectedStar::new(10.0, 10.0, 100.0),
        ];
        let quads = generate_quads(&stars, 3, 10);
        assert!(quads.is_empty());
    }
}
