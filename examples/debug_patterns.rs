//! Debug pattern matching by checking if true matches exist.

use chameleon::catalog::Index;
use chameleon::core::types::RaDec;
use chameleon::core::math::angular_separation;
use chameleon::extract::{extract_stars, ExtractionConfig};
use chameleon::pattern::{generate_quads, Quad, PatternMatcher};
use chameleon::wcs::Wcs;

fn main() {
    let image_path = "exposure_5.jpg";
    let index_path = "hipparcos_deep.idx";

    // Exact WCS from astrometry.net
    let crval = RaDec::from_degrees(14.3012483353, -9.12424543834);
    let crpix = (903.227925618, 575.047943115);
    let cd = [
        [-0.0159317827922, 0.00630093735133],
        [-0.00629294246889, -0.0159290388907],
    ];
    let wcs = Wcs::new(crpix, crval, cd);

    println!("Debug: Pattern Matching Analysis\n");

    // Load index
    let index = Index::open(index_path).expect("Failed to open index");
    println!("Index: {} stars, {} patterns\n", index.num_stars(), index.num_patterns());

    // Extract stars
    let extract_config = ExtractionConfig {
        max_stars: 50,
        sigma_threshold: 5.0,
        ..Default::default()
    };
    let stars = extract_stars(image_path, &extract_config).expect("Failed to extract stars");
    println!("Extracted {} stars\n", stars.len());

    // Build list of catalog stars in FOV (using pixel coordinates)
    let mut cat_in_fov: Vec<(u32, f64, f64)> = Vec::new();
    for (idx, star) in index.stars() {
        let pos = star.to_radec();
        let (px, py) = wcs.sky_to_pixel(&pos);
        if px >= 0.0 && px <= 1456.0 && py >= 0.0 && py <= 1088.0 {
            cat_in_fov.push((idx, px, py));
        }
    }
    println!("Catalog stars in FOV: {}\n", cat_in_fov.len());

    // Find catalog stars that match our detected stars (using pixel distance)
    println!("Matching detected stars to catalog using pixel distance:");
    let match_threshold_px = 15.0; // pixels
    let mut matched_pairs: Vec<(usize, u32)> = Vec::new(); // (detected_idx, catalog_idx)

    for (det_idx, star) in stars.iter().enumerate() {
        // Find nearest catalog star by pixel distance
        let mut best_dist = f64::MAX;
        let mut best_cat_idx: u32 = 0;

        for &(cat_idx, cx, cy) in &cat_in_fov {
            let dx = star.x - cx;
            let dy = star.y - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < best_dist {
                best_dist = dist;
                best_cat_idx = cat_idx;
            }
        }

        if best_dist < match_threshold_px {
            matched_pairs.push((det_idx, best_cat_idx));
            if matched_pairs.len() <= 15 {
                println!("  Detected {} ({:.0},{:.0}) -> Catalog {} (dist={:.1}px)",
                    det_idx, star.x, star.y, best_cat_idx, best_dist);
            }
        }
    }
    println!("Found {} true star correspondences\n", matched_pairs.len());

    if matched_pairs.len() < 4 {
        println!("Not enough matched stars to form quads");
        return;
    }

    // Generate quads from detected stars
    let image_quads = generate_quads(&stars, 30, 100);
    println!("Generated {} image quads\n", image_quads.len());

    // Check which image quads consist entirely of matched stars
    println!("Image quads with all stars matched to catalog:");
    let mut good_quads = 0;
    let matcher = PatternMatcher::new(&index)
        .with_bin_tolerance(0.05)
        .with_ratio_tolerance(0.05);

    for (q_idx, quad) in image_quads.iter().enumerate() {
        // Check if all 4 stars in the quad have catalog matches
        let mut all_matched = true;
        let mut catalog_indices: Vec<u32> = Vec::new();

        for &star_idx in &quad.star_indices {
            if let Some(&(_, cat_idx)) = matched_pairs.iter().find(|&&(d, _)| d == star_idx) {
                catalog_indices.push(cat_idx);
            } else {
                all_matched = false;
                break;
            }
        }

        if all_matched {
            good_quads += 1;
            if good_quads <= 5 {
                println!("  Quad {}: detected {:?} -> catalog {:?}",
                    q_idx, quad.star_indices, catalog_indices);
                println!("    Ratios: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                    quad.ratios[0], quad.ratios[1], quad.ratios[2], quad.ratios[3], quad.ratios[4]);

                // Check if a matching pattern exists in index
                let matches = matcher.find_matches(quad);

                println!("    Pattern matches in index: {}", matches.len());

                // Check if any match involves the correct catalog stars
                let mut found_correct = false;
                for m in &matches {
                    let pattern_stars: Vec<u16> = m.catalog_pattern.star_indices.iter().cloned().collect();
                    // Check if any of our catalog indices are in this pattern
                    let overlap: Vec<_> = catalog_indices.iter()
                        .filter(|&&c| pattern_stars.contains(&(c as u16)))
                        .collect();
                    if overlap.len() >= 2 {
                        found_correct = true;
                        println!("    FOUND: pattern stars {:?}, overlap {:?}",
                            pattern_stars, overlap);
                    }
                }
                if !found_correct && !matches.is_empty() {
                    println!("    Pattern matches are from different regions (false positives)");
                }
                println!();
            }
        }
    }
    println!("Total quads with all matched stars: {}\n", good_quads);

    // Show the catalog star indices we're looking for
    let matched_cat_set: std::collections::HashSet<u32> = matched_pairs.iter().map(|&(_, c)| c).collect();
    println!("Looking for index patterns containing catalog stars: {:?}\n", matched_cat_set);

    // Check for patterns with matching ratios at different tolerances
    println!("Checking ratio matching at different tolerances:");
    if good_quads > 0 {
        // Take the first good quad
        for (q_idx, quad) in image_quads.iter().enumerate() {
            let mut all_matched = true;
            let mut catalog_indices: Vec<u32> = Vec::new();
            for &star_idx in &quad.star_indices {
                if let Some(&(_, cat_idx)) = matched_pairs.iter().find(|&&(d, _)| d == star_idx) {
                    catalog_indices.push(cat_idx);
                } else {
                    all_matched = false;
                    break;
                }
            }

            if all_matched {
                println!("\n  Testing quad {}: catalog stars {:?}", q_idx, catalog_indices);
                println!("    Image ratios: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                    quad.ratios[0], quad.ratios[1], quad.ratios[2], quad.ratios[3], quad.ratios[4]);

                // Also compute what the catalog quad ratios would be
                let mut cat_positions: Vec<(f64, f64)> = Vec::new();
                for &cat_idx in &catalog_indices {
                    let star = index.get_star(cat_idx).unwrap();
                    let pos = star.to_radec();
                    // Project to pixel coords using known WCS
                    let (px, py) = wcs.sky_to_pixel(&pos);
                    cat_positions.push((px, py));
                }

                // Compute distances and ratios from catalog positions
                let mut distances = Vec::new();
                for i in 0..4 {
                    for j in (i+1)..4 {
                        let dx = cat_positions[i].0 - cat_positions[j].0;
                        let dy = cat_positions[i].1 - cat_positions[j].1;
                        let d = (dx*dx + dy*dy).sqrt();
                        distances.push(d);
                    }
                }
                let max_d = distances.iter().cloned().fold(0.0f64, f64::max);
                for d in &mut distances {
                    *d /= max_d;
                }
                distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
                println!("    Catalog ratios (projected): [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                    distances[0], distances[1], distances[2], distances[3], distances[4]);

                // Compute difference
                let diff: f64 = (0..5).map(|i| (quad.ratios[i] - distances[i]).abs()).sum();
                println!("    Sum of ratio differences: {:.4}", diff);

                for tol in [0.01, 0.02, 0.03, 0.05, 0.10] {
                    let m = PatternMatcher::new(&index)
                        .with_bin_tolerance(tol)
                        .with_ratio_tolerance(tol);
                    let matches = m.find_matches(quad);
                    let correct_matches: Vec<_> = matches.iter()
                        .filter(|m| {
                            let stars: Vec<u16> = m.catalog_pattern.star_indices.iter().cloned().collect();
                            catalog_indices.iter().filter(|&&c| stars.contains(&(c as u16))).count() >= 3
                        })
                        .collect();
                    println!("    Tolerance {:.2}: {} matches, {} correct",
                        tol, matches.len(), correct_matches.len());
                }

                break; // Only test first good quad
            }
        }
    }
}
