//! Debug script to understand why solving fails.

use citra_solve::catalog::builder::{BuildConfig, IndexBuilder};
use citra_solve::catalog::Index;
use citra_solve::core::types::DetectedStar;
use citra_solve::pattern::{compute_hash, generate_quads, PatternMatcher};
use citra_solve::wcs::Wcs;

fn main() {
    println!("Debug: Building small index...\n");

    let index_path = "/tmp/chameleon_debug.idx";

    // Build a smaller, simpler index - very fast settings
    let build_config = BuildConfig {
        fov_min_deg: 10.0,
        fov_max_deg: 30.0,
        mag_limit: 6.0,
        num_bins: 100_000,
        max_stars: 200, // Very small for fast testing
        max_patterns_per_star: 20,
    };

    let mut builder = IndexBuilder::new(build_config);

    // Generate synthetic stars clustered in one region
    // This ensures we have many stars in our test FOV
    let seed: u64 = 42;
    let mut rng_state = seed;
    let mut rng = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng_state >> 33) as f64 / (1u64 << 31) as f64
    };

    // Center of our cluster
    let center_ra = 1.0; // radians
    let center_dec = 0.5; // radians

    // Generate stars within ~20° of center
    for i in 0..500 {
        let ra = center_ra + (rng() - 0.5) * 0.7; // ~20° spread
        let dec = center_dec + (rng() - 0.5) * 0.7;
        let dec = dec.clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
        let mag = 1.0 + rng() * 5.0;
        builder.add_star(i as u32, ra, dec, mag as f32);
    }

    let stats = builder.build(index_path).expect("Failed to build index");
    println!(
        "Built index: {} stars, {} patterns\n",
        stats.num_stars, stats.num_patterns
    );

    // Load index
    let index = Index::open(index_path).expect("Failed to open index");

    println!("Creating test field from catalog stars...\n");

    // Create WCS for 20° FOV
    let center_radec = citra_solve::core::types::RaDec::new(center_ra, center_dec);
    // CD matrix is in DEGREES per pixel (FITS convention)
    let pixel_scale_deg: f64 = 20.0 / 1024.0; // 20 deg FOV for 1024 px
    let wcs = Wcs::new(
        (512.0, 512.0),
        center_radec,
        [[-pixel_scale_deg, 0.0], [0.0, pixel_scale_deg]],
    );

    println!(
        "WCS: center=({:.2}°, {:.2}°), scale={:.4}°/px\n",
        center_radec.ra_deg(),
        center_radec.dec_deg(),
        pixel_scale_deg
    );

    // Project all catalog stars and take ones in FOV
    let mut detected: Vec<DetectedStar> = Vec::new();
    for (idx, star) in index.stars() {
        let radec = star.to_radec();
        let (px, py) = wcs.sky_to_pixel(&radec);

        // Check if in image with margin
        if px >= 50.0 && px < 974.0 && py >= 50.0 && py < 718.0 {
            detected.push(DetectedStar::new(
                px,
                py,
                1000.0 - star.magnitude() as f64 * 100.0,
            ));
            if detected.len() <= 10 {
                println!(
                    "  Star {}: ({:.1}, {:.1}) px, mag={:.1}",
                    idx,
                    px,
                    py,
                    star.magnitude()
                );
            }
        }
    }
    println!("... {} stars total in FOV", detected.len());

    if detected.len() < 4 {
        println!("\nNot enough stars in FOV");
        std::fs::remove_file(index_path).ok();
        return;
    }

    // Sort by brightness
    detected.sort_by(|a, b| b.flux.partial_cmp(&a.flux).unwrap());

    // Generate quads
    println!(
        "\nGenerating quads from {} detected stars...",
        detected.len()
    );
    let quads = generate_quads(&detected, 20, 100);
    println!("Generated {} quads", quads.len());

    // Show first few quads' ratios
    for (i, quad) in quads.iter().take(3).enumerate() {
        println!("\nDetected quad {}:", i);
        println!("  Stars: {:?}", quad.star_indices);
        println!(
            "  Ratios: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
            quad.ratios[0], quad.ratios[1], quad.ratios[2], quad.ratios[3], quad.ratios[4]
        );
        println!("  Max edge: {:.1} px", quad.max_edge_pixels);

        let hash = compute_hash(&quad.ratios, 100_000);
        println!("  Hash bin: {}", hash);
    }

    // Check some catalog patterns
    println!("\nFirst few catalog patterns:");
    for bin in 0..index.num_bins() {
        if let Ok(patterns) = index.get_patterns_in_bin(bin) {
            if !patterns.is_empty() {
                for (i, p) in patterns.iter().take(3).enumerate() {
                    let ratios = p.ratios();
                    println!("  Pattern {} (bin {}): stars={:?}, ratios=[{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                        i, bin, p.stars(), ratios[0], ratios[1], ratios[2], ratios[3], ratios[4]);
                }
                break;
            }
        }
    }

    // Try matching with varying tolerance
    println!("\nTrying pattern matching...");

    for tol in [0.01, 0.02, 0.05, 0.1, 0.2] {
        let matcher = PatternMatcher::new(&index)
            .with_bin_tolerance(tol)
            .with_ratio_tolerance(tol);

        let matches = matcher.find_matches_batch(&quads, 10);
        println!("  Tolerance {:.2}: {} matches", tol, matches.len());

        if !matches.is_empty() && tol == 0.1 {
            println!("    First match:");
            println!("      Detected quad idx: {}", matches[0].detected_quad_idx);
            println!(
                "      Catalog stars: {:?}",
                matches[0].catalog_pattern.stars()
            );
            println!(
                "      Catalog ratios: {:?}",
                matches[0].catalog_pattern.ratios()
            );
            println!("      Distance: {:.6}", matches[0].ratio_distance);
        }
    }

    // Cleanup
    std::fs::remove_file(index_path).ok();
}
