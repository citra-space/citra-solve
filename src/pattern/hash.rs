//! Geometric hashing for pattern matching.

/// Number of bins for tetra signature dimensions.
pub const TETRA_BINS_PER_DIM: u32 = 64;
/// Number of bins for the ratio anchor dimension.
pub const RATIO_BINS_PER_DIM: u32 = 80;

/// One hashed probe candidate with its integer-grid distance to center.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HashProbe {
    pub bin: u32,
    pub l1_distance: u16,
}

#[inline]
fn quantize_ratio_anchor(r: f64) -> u32 {
    (r.clamp(0.0, 0.999_999) * RATIO_BINS_PER_DIM as f64).floor() as u32
}

#[inline]
fn quantize_tetra(v: f64) -> u32 {
    // Tetra coordinates are typically in [-2, 2] after baseline normalization.
    let norm = ((v.clamp(-2.0, 2.0) + 2.0) * 0.25).clamp(0.0, 0.999_999);
    (norm * TETRA_BINS_PER_DIM as f64).floor() as u32
}

#[inline]
fn hash_from_bins(tetra_bins: [u32; 4], ratio_anchor_bin: u32, num_bins: u32) -> u32 {
    let mut h: u64 = 0x9E37_79B9_7F4A_7C15;
    for &b in &tetra_bins {
        h ^= b as u64 + 0x9E37_79B9 + (h << 6) + (h >> 2);
    }
    h ^= ratio_anchor_bin as u64 + 0x85EB_CA77 + (h << 6) + (h >> 2);
    (h % num_bins as u64) as u32
}

/// Compute hash bin for a quad using tetra-primary signature and one ratio anchor.
#[inline]
pub fn compute_hash(ratios: &[f64; 5], tetra: &[f64; 4], num_bins: u32) -> u32 {
    let tetra_bins = [
        quantize_tetra(tetra[0]),
        quantize_tetra(tetra[1]),
        quantize_tetra(tetra[2]),
        quantize_tetra(tetra[3]),
    ];
    let ratio_anchor = quantize_ratio_anchor(ratios[2]);
    hash_from_bins(tetra_bins, ratio_anchor, num_bins)
}

/// Query all hash bins that could match within ratio/tetra tolerance.
pub fn query_hash_bins(
    ratios: &[f64; 5],
    tetra: &[f64; 4],
    num_bins: u32,
    ratio_tolerance: f64,
    tetra_tolerance: f64,
) -> Vec<u32> {
    query_hash_bins_ranked(
        ratios,
        tetra,
        num_bins,
        ratio_tolerance,
        tetra_tolerance,
        usize::MAX,
    )
    .into_iter()
    .map(|p| p.bin)
    .collect()
}

/// Query candidate bins ranked by integer-grid distance from hash center.
pub fn query_hash_bins_ranked(
    ratios: &[f64; 5],
    tetra: &[f64; 4],
    num_bins: u32,
    ratio_tolerance: f64,
    tetra_tolerance: f64,
    max_bins: usize,
) -> Vec<HashProbe> {
    if num_bins == 0 || max_bins == 0 {
        return Vec::new();
    }

    let centers_t = [
        quantize_tetra(tetra[0]) as i32,
        quantize_tetra(tetra[1]) as i32,
        quantize_tetra(tetra[2]) as i32,
        quantize_tetra(tetra[3]) as i32,
    ];
    let center_r = quantize_ratio_anchor(ratios[2]) as i32;

    let delta_t = (tetra_tolerance.max(0.0) * TETRA_BINS_PER_DIM as f64).ceil() as i32;
    let delta_r = (ratio_tolerance.max(0.0) * RATIO_BINS_PER_DIM as f64).ceil() as i32;

    let mut probes = Vec::new();

    fn recurse(
        idx: usize,
        centers_t: &[i32; 4],
        center_r: i32,
        current_t: &mut [u32; 4],
        current_r: &mut u32,
        l1_distance: i32,
        delta_t: i32,
        delta_r: i32,
        num_bins: u32,
        probes: &mut Vec<HashProbe>,
    ) {
        if idx == 5 {
            probes.push(HashProbe {
                bin: hash_from_bins(*current_t, *current_r, num_bins),
                l1_distance: l1_distance as u16,
            });
            return;
        }

        if idx < 4 {
            let center = centers_t[idx];
            let min_bin = (center - delta_t).max(0) as u32;
            let max_bin = (center + delta_t).min(TETRA_BINS_PER_DIM as i32 - 1) as u32;
            for bin in min_bin..=max_bin {
                current_t[idx] = bin;
                recurse(
                    idx + 1,
                    centers_t,
                    center_r,
                    current_t,
                    current_r,
                    l1_distance + (bin as i32 - center).abs(),
                    delta_t,
                    delta_r,
                    num_bins,
                    probes,
                );
            }
            return;
        }

        let min_bin = (center_r - delta_r).max(0) as u32;
        let max_bin = (center_r + delta_r).min(RATIO_BINS_PER_DIM as i32 - 1) as u32;
        for bin in min_bin..=max_bin {
            *current_r = bin;
            recurse(
                idx + 1,
                centers_t,
                center_r,
                current_t,
                current_r,
                l1_distance + (bin as i32 - center_r).abs(),
                delta_t,
                delta_r,
                num_bins,
                probes,
            );
        }
    }

    let mut current_t = [0u32; 4];
    let mut current_r = 0u32;
    recurse(
        0,
        &centers_t,
        center_r,
        &mut current_t,
        &mut current_r,
        0,
        delta_t,
        delta_r,
        num_bins,
        &mut probes,
    );

    probes.sort_by(|a, b| {
        a.l1_distance
            .cmp(&b.l1_distance)
            .then_with(|| a.bin.cmp(&b.bin))
    });
    probes.dedup_by(|a, b| a.bin == b.bin);
    if probes.len() > max_bins {
        probes.truncate(max_bins);
    }
    probes
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
        let tetra = [0.15, 0.21, 0.77, 0.41];
        let h1 = compute_hash(&ratios, &tetra, 1_000_000);
        let h2 = compute_hash(&ratios, &tetra, 1_000_000);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_changes_with_tetra() {
        let ratios = [0.1, 0.2, 0.3, 0.4, 0.5];
        let t1 = [0.1, 0.2, 0.8, 0.3];
        let t2 = [0.2, 0.1, 0.7, 0.4];
        let h1 = compute_hash(&ratios, &t1, 1_000_000);
        let h2 = compute_hash(&ratios, &t2, 1_000_000);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_query_bins_includes_center() {
        let ratios = [0.5, 0.5, 0.5, 0.5, 0.5];
        let tetra = [0.2, 0.2, 0.8, 0.2];
        let center_hash = compute_hash(&ratios, &tetra, 1_000_000);
        let bins = query_hash_bins(&ratios, &tetra, 1_000_000, 0.01, 0.02);
        assert!(bins.contains(&center_hash));
    }

    #[test]
    fn test_query_bins_expands_with_tolerance() {
        let ratios = [0.5, 0.5, 0.5, 0.5, 0.5];
        let tetra = [0.2, 0.2, 0.8, 0.2];
        let bins_small = query_hash_bins(&ratios, &tetra, 1_000_000, 0.01, 0.02);
        let bins_large = query_hash_bins(&ratios, &tetra, 1_000_000, 0.03, 0.04);
        assert!(bins_large.len() > bins_small.len());
    }

    #[test]
    fn test_query_bins_ranked_prefers_center() {
        let ratios = [0.4, 0.5, 0.6, 0.7, 0.8];
        let tetra = [0.11, 0.24, 0.73, 0.35];
        let center = compute_hash(&ratios, &tetra, 1_000_000);
        let probes = query_hash_bins_ranked(&ratios, &tetra, 1_000_000, 0.02, 0.03, 32);
        assert!(!probes.is_empty());
        assert_eq!(probes[0].bin, center);
        assert_eq!(probes[0].l1_distance, 0);
    }
}
