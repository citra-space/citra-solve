//! Debug pattern matching to find where correct match ranks.

use citra_solve::catalog::Index;
use citra_solve::core::types::RaDec;
use citra_solve::extract::{extract_stars, ExtractionConfig};
use citra_solve::pattern::{generate_quads, PatternMatcher};
use citra_solve::wcs::Wcs;

fn main() {
    let image_path = "exposure_5.jpg";
    let index_path = "hipparcos_deep.idx";

    // Exact WCS
    let crval = RaDec::from_degrees(14.3012483353, -9.12424543834);
    let crpix = (903.227925618, 575.047943115);
    let cd = [
        [-0.0159317827922, 0.00630093735133],
        [-0.00629294246889, -0.0159290388907],
    ];
    let wcs = Wcs::new(crpix, crval, cd);

    let index = Index::open(index_path).expect("Failed to open index");
    println!("Index: {} patterns\n", index.num_patterns());

    // Extract stars
    let config = ExtractionConfig {
        max_stars: 50,
        sigma_threshold: 5.0,
        ..Default::default()
    };
    let stars = extract_stars(image_path, &config).expect("Failed to extract");
    println!("Extracted {} stars\n", stars.len());

    // Build catalog star list in FOV (pixel matching)
    let mut cat_in_fov: Vec<(u32, f64, f64)> = Vec::new();
    for (idx, star) in index.stars() {
        let pos = star.to_radec();
        let (px, py) = wcs.sky_to_pixel(&pos);
        if px >= 0.0 && px <= 1456.0 && py >= 0.0 && py <= 1088.0 {
            cat_in_fov.push((idx, px, py));
        }
    }

    // Match detected to catalog
    let mut matched_pairs: Vec<(usize, u32)> = Vec::new();
    for (det_idx, star) in stars.iter().enumerate() {
        for &(cat_idx, cx, cy) in &cat_in_fov {
            let dx = star.x - cx;
            let dy = star.y - cy;
            let dist = (dx*dx + dy*dy).sqrt();
            if dist < 15.0 {
                matched_pairs.push((det_idx, cat_idx));
                break;
            }
        }
    }
    println!("{} detected stars match catalog\n", matched_pairs.len());

    // Generate quads
    let quads = generate_quads(&stars, 30, 100);
    println!("Generated {} quads\n", quads.len());

    // Find quad [0, 1, 2, 3] and its pattern distance
    let matcher = PatternMatcher::new(&index)
        .with_bin_tolerance(0.03)
        .with_ratio_tolerance(0.03);

    // Find matches for the first quad (should be [51, 376, 306, 272])
    let quad0 = &quads[0];
    println!("Quad 0: detected stars {:?}", quad0.star_indices);
    println!("  Ratios: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]\n",
        quad0.ratios[0], quad0.ratios[1], quad0.ratios[2], quad0.ratios[3], quad0.ratios[4]);

    // Get catalog stars for this quad
    let mut cat_indices: Vec<u32> = Vec::new();
    for &det_idx in &quad0.star_indices {
        if let Some(&(_, cat_idx)) = matched_pairs.iter().find(|&&(d, _)| d == det_idx) {
            cat_indices.push(cat_idx);
        } else {
            println!("  Detected star {} has no catalog match!", det_idx);
        }
    }
    println!("  Catalog stars: {:?}\n", cat_indices);

    // Find all pattern matches
    let matches = matcher.find_matches(quad0);
    println!("Found {} pattern matches\n", matches.len());

    // Sort by pattern distance
    let mut matches_sorted: Vec<_> = matches.iter().collect();
    matches_sorted.sort_by(|a, b| a.ratio_distance.partial_cmp(&b.ratio_distance).unwrap());

    // Find where the correct match is
    let target: std::collections::HashSet<u16> = cat_indices.iter().map(|&c| c as u16).collect();

    println!("Top 20 matches by pattern distance:");
    for (rank, m) in matches_sorted.iter().take(20).enumerate() {
        let stars: std::collections::HashSet<u16> =
            m.catalog_pattern.star_indices.iter().cloned().collect();
        let overlap = stars.intersection(&target).count();
        let marker = if overlap == 4 { "<<<< CORRECT" } else { "" };
        let r = m.catalog_pattern.ratios();
        println!("  {:3}: dist={:.6} stars={:?} {} {}",
            rank, m.ratio_distance, m.catalog_pattern.star_indices,
            if overlap > 0 { format!("(overlap: {})", overlap) } else { "".to_string() },
            marker);
    }

    // Find rank of correct match
    for (rank, m) in matches_sorted.iter().enumerate() {
        let stars: std::collections::HashSet<u16> =
            m.catalog_pattern.star_indices.iter().cloned().collect();
        let overlap = stars.intersection(&target).count();
        if overlap == 4 {
            println!("\n*** Correct match [51, 272, 306, 376] at rank {} (distance: {:.6}) ***",
                rank, m.ratio_distance);
            break;
        }
    }
}
