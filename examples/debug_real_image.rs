//! Debug pattern matching on a real image.

use citra_solve::catalog::Index;
use citra_solve::extract::{extract_stars, ExtractionConfig};
use citra_solve::pattern::{generate_quads, PatternMatcher, compute_hash};

fn main() {
    let image_path = "exposure_5.jpg";
    let index_path = "hipparcos.idx";

    println!("Debug: Real Image Pattern Matching\n");

    // Load index
    let index = Index::open(index_path).expect("Failed to open index");
    let (fov_min, fov_max) = index.fov_range_deg();
    println!("Index: {} stars, {} patterns, {} bins",
        index.num_stars(), index.num_patterns(), index.num_bins());
    println!("FOV range: {:.1}° - {:.1}°\n", fov_min, fov_max);

    // Extract stars
    let extract_config = ExtractionConfig {
        max_stars: 100,
        sigma_threshold: 3.0,  // Lower threshold
        min_flux: 50.0,        // Lower min flux
        ..Default::default()
    };

    let stars = extract_stars(image_path, &extract_config).expect("Failed to extract stars");
    println!("Extracted {} stars\n", stars.len());

    // Show brightest stars
    println!("Top 15 stars:");
    for (i, star) in stars.iter().take(15).enumerate() {
        println!("  {}: ({:.1}, {:.1}) flux={:.0}", i, star.x, star.y, star.flux);
    }

    // Generate quads
    println!("\nGenerating quads...");
    let quads = generate_quads(&stars, 50, 200);
    println!("Generated {} quads\n", quads.len());

    // Show first few quads
    println!("First 5 quads:");
    for (i, quad) in quads.iter().take(5).enumerate() {
        let hash = compute_hash(&quad.ratios, index.num_bins());
        println!("  Quad {}: stars={:?}", i, quad.star_indices);
        println!("    Ratios: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
            quad.ratios[0], quad.ratios[1], quad.ratios[2], quad.ratios[3], quad.ratios[4]);
        println!("    Max edge: {:.1} px, Hash bin: {}", quad.max_edge_pixels, hash);
    }

    // Try matching at various tolerances
    println!("\nPattern matching:");
    for tol in [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2] {
        let matcher = PatternMatcher::new(&index)
            .with_bin_tolerance(tol)
            .with_ratio_tolerance(tol);

        let matches = matcher.find_matches_batch(&quads, 10);
        println!("  Tolerance {:.2}: {} matches", tol, matches.len());

        if !matches.is_empty() && matches.len() < 50 {
            println!("    First match:");
            let m = &matches[0];
            println!("      Detected quad: {:?}", quads[m.detected_quad_idx].star_indices);
            println!("      Catalog pattern: {:?}", m.catalog_pattern.stars());
            println!("      Ratio distance: {:.4}", m.ratio_distance);
        }
    }

    // Estimate pixel scale from max edge
    // If we have a 20° FOV over 1456 pixels diagonal, that's ~1 arcmin/pixel
    // Let's compute actual quad sizes and estimate the angular scale
    println!("\nQuad scale analysis:");
    let img_diagonal = (1456.0_f64.powi(2) + 1088.0_f64.powi(2)).sqrt();
    println!("  Image diagonal: {:.1} px", img_diagonal);

    // Compute average quad max edge
    let avg_max_edge: f64 = quads.iter().take(20).map(|q| q.max_edge_pixels).sum::<f64>() / 20.0;
    println!("  Average quad max edge: {:.1} px", avg_max_edge);

    // If the FOV is X degrees, then pixel scale is X/diagonal deg/px
    // For various FOV assumptions:
    for fov_deg in [5.0, 10.0, 15.0, 20.0, 25.0, 30.0] {
        let pixel_scale_deg = fov_deg / img_diagonal;
        let avg_quad_angular = avg_max_edge * pixel_scale_deg;
        println!("  If FOV={:.0}°: pixel_scale={:.4}°/px, avg quad={:.1}° (index range: {:.1}-{:.1}°)",
            fov_deg, pixel_scale_deg, avg_quad_angular, fov_min, fov_max);
    }
}
