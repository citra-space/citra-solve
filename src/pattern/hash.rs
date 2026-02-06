//! Geometric hashing for pattern matching.

/// Default number of bins per dimension for hashing.
pub const BINS_PER_DIM: u32 = 100;

/// Compute the hash bin for a set of edge ratios.
///
/// Uses a 5D quantization scheme where each ratio is mapped to one of
/// BINS_PER_DIM bins, then combined into a single hash value.
#[inline]
pub fn compute_hash(ratios: &[f64; 5], num_bins: u32) -> u32 {
    let mut hash: u64 = 0;
    let mut multiplier: u64 = 1;

    for &r in ratios.iter() {
        let bin = ((r * BINS_PER_DIM as f64) as u64).min(BINS_PER_DIM as u64 - 1);
        hash += bin * multiplier;
        multiplier *= BINS_PER_DIM as u64;
    }

    (hash % num_bins as u64) as u32
}

/// Query all hash bins that could match the given ratios within tolerance.
///
/// Returns a list of bin indices to query. The tolerance is specified as
/// a fraction of the ratio range (0-1), typically 0.01-0.05.
pub fn query_hash_bins(ratios: &[f64; 5], num_bins: u32, tolerance: f64) -> Vec<u32> {
    let delta = (tolerance * BINS_PER_DIM as f64).ceil() as i32;

    // Compute center bins for each dimension
    let centers: Vec<i32> = ratios
        .iter()
        .map(|&r| (r * BINS_PER_DIM as f64) as i32)
        .collect();

    // Generate all combinations within the tolerance hypercube
    let mut hashes = Vec::new();

    fn recurse(
        centers: &[i32],
        idx: usize,
        current: &mut [u32; 5],
        delta: i32,
        num_bins: u32,
        hashes: &mut Vec<u32>,
    ) {
        if idx == 5 {
            // Compute hash from current bin combination
            let mut hash: u64 = 0;
            let mut multiplier: u64 = 1;
            for &bin in current.iter() {
                hash += bin as u64 * multiplier;
                multiplier *= BINS_PER_DIM as u64;
            }
            hashes.push((hash % num_bins as u64) as u32);
            return;
        }

        let center = centers[idx];
        let min_bin = (center - delta).max(0) as u32;
        let max_bin = (center + delta).min(BINS_PER_DIM as i32 - 1) as u32;

        for bin in min_bin..=max_bin {
            current[idx] = bin;
            recurse(centers, idx + 1, current, delta, num_bins, hashes);
        }
    }

    let mut current = [0u32; 5];
    recurse(&centers, 0, &mut current, delta, num_bins, &mut hashes);

    // Remove duplicates (can occur due to modulo)
    hashes.sort_unstable();
    hashes.dedup();
    hashes
}

/// Compute the L2 distance between two sets of ratios.
#[inline]
pub fn ratio_distance(a: &[f64; 5], b: &[f64; 5]) -> f64 {
    let mut sum_sq = 0.0;
    for i in 0..5 {
        let diff = a[i] - b[i];
        sum_sq += diff * diff;
    }
    sum_sq.sqrt()
}

/// Check if two ratio sets match within tolerance.
#[inline]
pub fn ratios_match(a: &[f64; 5], b: &[f64; 5], tolerance: f64) -> bool {
    ratio_distance(a, b) < tolerance
}

/// Fine-grained ratio comparison that checks each dimension.
#[inline]
pub fn ratios_match_per_dim(a: &[f64; 5], b: &[f64; 5], tolerance: f64) -> bool {
    for i in 0..5 {
        if (a[i] - b[i]).abs() > tolerance {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_deterministic() {
        let ratios = [0.1, 0.2, 0.3, 0.4, 0.5];
        let h1 = compute_hash(&ratios, 1_000_000);
        let h2 = compute_hash(&ratios, 1_000_000);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_different_ratios() {
        let r1 = [0.1, 0.2, 0.3, 0.4, 0.5];
        let r2 = [0.15, 0.25, 0.35, 0.45, 0.55];
        let h1 = compute_hash(&r1, 1_000_000);
        let h2 = compute_hash(&r2, 1_000_000);
        // These should (usually) hash to different bins
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_query_bins_includes_center() {
        let ratios = [0.5, 0.5, 0.5, 0.5, 0.5];
        let center_hash = compute_hash(&ratios, 1_000_000);
        let bins = query_hash_bins(&ratios, 1_000_000, 0.01);

        assert!(bins.contains(&center_hash));
    }

    #[test]
    fn test_query_bins_expands_with_tolerance() {
        let ratios = [0.5, 0.5, 0.5, 0.5, 0.5];
        let bins_small = query_hash_bins(&ratios, 1_000_000, 0.01);
        let bins_large = query_hash_bins(&ratios, 1_000_000, 0.05);

        assert!(bins_large.len() > bins_small.len());
    }

    #[test]
    fn test_ratio_distance() {
        let a = [0.1, 0.2, 0.3, 0.4, 0.5];
        let b = [0.1, 0.2, 0.3, 0.4, 0.5];
        assert!((ratio_distance(&a, &b)).abs() < 1e-10);

        let c = [0.2, 0.3, 0.4, 0.5, 0.6];
        let d = ratio_distance(&a, &c);
        assert!(d > 0.0);
    }

    #[test]
    fn test_ratios_match() {
        let a = [0.1, 0.2, 0.3, 0.4, 0.5];
        let b = [0.101, 0.201, 0.301, 0.401, 0.501];

        assert!(ratios_match(&a, &b, 0.01));
        assert!(!ratios_match(&a, &b, 0.001));
    }
}
