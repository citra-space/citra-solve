//! Debug the solver pipeline step by step.

use citra_solve::catalog::Index;
use citra_solve::core::types::RaDec;
use citra_solve::extract::{extract_stars, ExtractionConfig};
use citra_solve::pattern::{generate_quads, PatternMatcher};
use citra_solve::solver::hypothesis::generate_hypotheses;
use citra_solve::solver::verify::{verify_hypothesis, VerifyConfig};
use citra_solve::wcs::Wcs;

fn main() {
    let image_path = "exposure_5.jpg";
    let index_path = "hipparcos_deep.idx";

    println!("Debug: Solver Pipeline\n");

    // Load index
    let index = Index::open(index_path).expect("Failed to open index");
    let (fov_min, fov_max) = index.fov_range_deg();
    println!(
        "Index: {} stars, {} patterns",
        index.num_stars(),
        index.num_patterns()
    );
    println!("FOV range: {:.1}° - {:.1}°\n", fov_min, fov_max);

    // Extract stars (use same settings as debug_matching)
    let extract_config = ExtractionConfig {
        max_stars: 50,
        sigma_threshold: 5.0,
        ..Default::default()
    };

    let stars = extract_stars(image_path, &extract_config).expect("Failed to extract stars");
    println!("Extracted {} stars\n", stars.len());

    // Generate quads (use same settings as debug_matching: 30 stars, 100 quads)
    let quads = generate_quads(&stars, 30, 100);
    println!("Generated {} quads\n", quads.len());

    // Show Quad 0 details and identify its catalog stars using the known WCS
    let true_crval = RaDec::from_degrees(14.3012483353, -9.12424543834);
    let true_crpix = (903.227925618, 575.047943115);
    let true_cd = [
        [-0.0159317827922, 0.00630093735133],
        [-0.00629294246889, -0.0159290388907],
    ];
    let true_wcs = Wcs::new(true_crpix, true_crval, true_cd);

    // Build catalog star list in FOV
    let mut cat_in_fov: Vec<(u32, f64, f64)> = Vec::new();
    for (idx, star) in index.stars() {
        let pos = star.to_radec();
        let (px, py) = true_wcs.sky_to_pixel(&pos);
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
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < 15.0 {
                matched_pairs.push((det_idx, cat_idx));
                break;
            }
        }
    }

    println!("Quad 0: detected stars {:?}", quads[0].star_indices);
    let quad0_cat: Vec<u32> = quads[0]
        .star_indices
        .iter()
        .filter_map(|&det_idx| {
            matched_pairs
                .iter()
                .find(|&&(d, _)| d == det_idx)
                .map(|&(_, c)| c)
        })
        .collect();
    println!("  Catalog stars: {:?}", quad0_cat);
    println!("  Ratios: {:?}\n", quads[0].ratios);

    // Pattern matching
    let matcher = PatternMatcher::new(&index)
        .with_bin_tolerance(0.03)
        .with_ratio_tolerance(0.03);

    // First check matches specifically for Quad 0
    let quad0_matches = matcher.find_matches(&quads[0]);
    println!("Quad 0 specific matches: {} total", quad0_matches.len());

    // Find rank of correct match
    let target: std::collections::HashSet<u16> = [51u16, 272, 306, 376].iter().cloned().collect();
    for (rank, m) in quad0_matches.iter().take(20).enumerate() {
        let stars: std::collections::HashSet<u16> =
            m.catalog_pattern.star_indices.iter().cloned().collect();
        let overlap = stars.intersection(&target).count();
        let marker = if overlap == 4 { " <<<< CORRECT" } else { "" };
        println!(
            "  Rank {:2}: dist={:.6} stars={:?} overlap={}{}",
            rank, m.ratio_distance, m.catalog_pattern.star_indices, overlap, marker
        );
    }

    let matches = matcher.find_matches_batch(&quads, 10);
    println!("\nBatch total: {} pattern matches", matches.len());

    // Check if correct match is in the batch
    let mut found_correct = false;
    for (i, m) in matches.iter().enumerate() {
        let stars: std::collections::HashSet<u16> =
            m.catalog_pattern.star_indices.iter().cloned().collect();
        let overlap = stars.intersection(&target).count();
        if overlap == 4 {
            println!("Found correct match [51,272,306,376] in batch at position {}, detected_quad_idx={}",
                i, m.detected_quad_idx);
            found_correct = true;
        }
    }
    if !found_correct {
        println!("WARNING: Correct match NOT in batch!");
    }
    println!();

    if matches.is_empty() {
        println!("No matches found - cannot proceed");
        return;
    }

    // Take top matches - need enough to include correct match (at position 332)
    let top_matches: Vec<_> = matches.into_iter().take(500).collect();
    println!(
        "Using top {} matches for hypothesis generation\n",
        top_matches.len()
    );

    // Generate hypotheses
    let image_width = 1456u32;
    let image_height = 1088u32;

    let hypotheses = generate_hypotheses(
        &stars,
        &quads,
        &top_matches,
        &index,
        image_width,
        image_height,
    );

    println!("Generated {} hypotheses\n", hypotheses.len());

    if hypotheses.is_empty() {
        println!("No hypotheses generated - cannot proceed");
        return;
    }

    // Show first few hypotheses
    println!("First 5 hypotheses:");
    for (i, h) in hypotheses.iter().take(5).enumerate() {
        let center = h.wcs.crval();
        let scale = h.wcs.pixel_scale_arcsec();
        println!(
            "  {}: center=({:.4}°, {:.4}°), scale={:.3}\"/px, pattern_dist={:.4}",
            i,
            center.ra_deg(),
            center.dec_deg(),
            scale,
            h.pattern_distance
        );
    }

    // Look for hypothesis with the correct catalog stars [51, 272, 306, 376]
    let target_stars: std::collections::HashSet<u32> =
        [51u32, 272, 306, 376].iter().cloned().collect();
    println!("\nSearching for hypothesis with correct catalog stars [51, 272, 306, 376]:");
    for (i, h) in hypotheses.iter().enumerate() {
        let match_ids: std::collections::HashSet<u32> =
            h.star_matches.iter().map(|(_, c)| c.id).collect();
        let overlap = match_ids.intersection(&target_stars).count();
        if overlap >= 3 {
            let center = h.wcs.crval();
            let scale = h.wcs.pixel_scale_arcsec();
            println!(
                "  Hypothesis {}: overlap={}, center=({:.4}°, {:.4}°), scale={:.3}\"/px",
                i,
                overlap,
                center.ra_deg(),
                center.dec_deg(),
                scale
            );
            println!("    Star IDs: {:?}", match_ids);
        }
    }

    // Also look for hypotheses with reasonable scale (60-65 arcsec/px) and near correct position
    println!("\nSearching for hypothesis with reasonable scale (55-70 arcsec/px):");
    for (i, h) in hypotheses.iter().enumerate() {
        let scale = h.wcs.pixel_scale_arcsec();
        if scale > 55.0 && scale < 70.0 {
            let center = h.wcs.crval();
            println!(
                "  Hypothesis {}: center=({:.4}°, {:.4}°), scale={:.3}\"/px, pattern_dist={:.4}",
                i,
                center.ra_deg(),
                center.dec_deg(),
                scale,
                h.pattern_distance
            );
            println!(
                "    First star match IDs: {:?}",
                h.star_matches.iter().map(|(_, c)| c.id).collect::<Vec<_>>()
            );
        }
    }

    // Verify first few hypotheses
    // Use larger sigma to account for projection errors in wide-FOV images
    println!("\nVerifying hypotheses:");
    let verify_config = VerifyConfig {
        max_match_distance_pixels: 20.0, // Relax matching
        position_sigma_pixels: 10.0,     // Larger sigma for wide FOV
        ..Default::default()
    };

    for (i, hypothesis) in hypotheses.iter().take(10).enumerate() {
        let result = verify_hypothesis(
            hypothesis,
            &stars,
            &index,
            image_width,
            image_height,
            &verify_config,
        );

        println!(
            "  Hypothesis {}: matched={}/{}, rms={:.2}px, log_odds={:.2}",
            i,
            result.num_matched,
            result.num_expected,
            result.rms_residual_pixels,
            result.hypothesis.log_odds
        );

        if result.num_matched > 0 && i == 0 {
            // Show first few matches
            println!("    First matches:");
            for (j, &(det_idx, _cat)) in result.hypothesis.star_matches.iter().take(3).enumerate() {
                let det = &stars[det_idx];
                println!("      Match {}: detected ({:.1}, {:.1})", j, det.x, det.y);
            }
        }
    }

    // Find best hypothesis
    let mut best = &hypotheses[0];
    let mut best_result = verify_hypothesis(
        best,
        &stars,
        &index,
        image_width,
        image_height,
        &verify_config,
    );
    let mut best_idx = 0;

    for (i, h) in hypotheses.iter().enumerate() {
        let r = verify_hypothesis(h, &stars, &index, image_width, image_height, &verify_config);
        if r.hypothesis.log_odds > best_result.hypothesis.log_odds {
            best = h;
            best_result = r;
            best_idx = i;
        }
    }

    println!("\nBest hypothesis (index {}):", best_idx);
    let center = best.wcs.crval();
    println!(
        "  Center: RA={:.4}°, Dec={:.4}°",
        center.ra_deg(),
        center.dec_deg()
    );
    println!("  Scale: {:.2} arcsec/px", best.wcs.pixel_scale_arcsec());
    println!(
        "  Matched: {}/{} stars",
        best_result.num_matched, best_result.num_expected
    );
    println!(
        "  RMS residual: {:.2} pixels",
        best_result.rms_residual_pixels
    );
    println!("  Log-odds: {:.2}", best_result.hypothesis.log_odds);
    println!(
        "  Star IDs: {:?}",
        best.star_matches
            .iter()
            .map(|(_, c)| c.id)
            .collect::<Vec<_>>()
    );

    // Also verify hypothesis with the correct stars specifically
    println!("\nLooking for hypothesis with exact stars [51, 376, 306, 272]:");
    for (i, h) in hypotheses.iter().enumerate() {
        let ids: Vec<u32> = h.star_matches.iter().map(|(_, c)| c.id).collect();
        if ids.contains(&51) && ids.contains(&272) && ids.contains(&306) && ids.contains(&376) {
            let r = verify_hypothesis(h, &stars, &index, image_width, image_height, &verify_config);
            let center = h.wcs.crval();
            println!("  Hypothesis {} with stars {:?}:", i, ids);
            println!(
                "    Center: RA={:.4}°, Dec={:.4}°",
                center.ra_deg(),
                center.dec_deg()
            );
            println!("    Scale: {:.2} arcsec/px", h.wcs.pixel_scale_arcsec());
            println!("    Matched: {}/{} stars", r.num_matched, r.num_expected);
            println!("    RMS residual: {:.2} pixels", r.rms_residual_pixels);
            println!("    Log-odds: {:.2}", r.hypothesis.log_odds);
        }
    }

    // Compare to true WCS
    println!("\nTrue WCS (from astrometry.net):");
    println!("  Center: RA=14.3012°, Dec=-9.1242°");
    println!("  Scale: 61.67 arcsec/px");

    // Check what the best WCS gives for image center
    let image_center_x = image_width as f64 / 2.0;
    let image_center_y = image_height as f64 / 2.0;
    let sky_at_center = best.wcs.pixel_to_sky(image_center_x, image_center_y);
    println!(
        "\nBest WCS applied to image center ({}, {}):",
        image_center_x, image_center_y
    );
    println!(
        "  Sky position: RA={:.4}°, Dec={:.4}°",
        sky_at_center.ra_deg(),
        sky_at_center.dec_deg()
    );

    // Compare with true position at image center
    let true_center_sky = true_wcs.pixel_to_sky(image_center_x, image_center_y);
    println!(
        "  True sky at center: RA={:.4}°, Dec={:.4}°",
        true_center_sky.ra_deg(),
        true_center_sky.dec_deg()
    );
}
