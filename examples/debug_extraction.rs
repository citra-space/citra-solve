//! Debug star extraction - compare detected vs catalog stars.

use citra_solve::catalog::Index;
use citra_solve::core::math::angular_separation;
use citra_solve::core::types::RaDec;
use citra_solve::extract::{extract_stars, ExtractionConfig};
use citra_solve::wcs::Wcs;

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

    println!("Debug: Star Extraction Analysis\n");

    // Load index
    let index = Index::open(index_path).expect("Failed to open index");
    println!("Index: {} stars (mag limit ~8.0)\n", index.num_stars());

    // Find all catalog stars in the field of view
    println!("Catalog stars in field of view (projected to pixels):");
    let mut cat_in_fov: Vec<(u32, f64, f64, f32)> = Vec::new();

    for (idx, star) in index.stars() {
        let pos = star.to_radec();
        let (px, py) = wcs.sky_to_pixel(&pos);

        // Check if in image bounds (with margin)
        if px >= -50.0 && px <= 1506.0 && py >= -50.0 && py <= 1138.0 {
            cat_in_fov.push((idx, px, py, star.magnitude()));
        }
    }

    // Sort by magnitude
    cat_in_fov.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap());

    println!("Found {} catalog stars in FOV\n", cat_in_fov.len());
    println!("Top 20 brightest:");
    for (idx, px, py, mag) in cat_in_fov.iter().take(20) {
        println!("  Cat {} at ({:6.1}, {:6.1}) mag={:.1}", idx, px, py, mag);
    }
    println!();

    // Extract stars with different thresholds
    for sigma in [3.0, 5.0, 8.0, 10.0] {
        let config = ExtractionConfig {
            max_stars: 100,
            sigma_threshold: sigma,
            min_separation: 10.0,
            ..Default::default()
        };
        let stars = extract_stars(image_path, &config).expect("Failed to extract");

        // Count how many match catalog
        let mut matches = 0;
        for star in &stars {
            let sky = wcs.pixel_to_sky(star.x, star.y);
            for (_, cat_star) in index.stars() {
                let cat_pos = cat_star.to_radec();
                let sep = angular_separation(&sky, &cat_pos).to_degrees() * 3600.0;
                if sep < 60.0 {
                    matches += 1;
                    break;
                }
            }
        }

        println!(
            "Sigma={:.1}: extracted {} stars, {} match catalog ({:.0}%)",
            sigma,
            stars.len(),
            matches,
            100.0 * matches as f64 / stars.len() as f64
        );
    }
    println!();

    // Now extract with best config and show matches
    let config = ExtractionConfig {
        max_stars: 100,
        sigma_threshold: 5.0,
        min_separation: 10.0,
        ..Default::default()
    };
    let stars = extract_stars(image_path, &config).expect("Failed to extract");

    println!("Detailed analysis with sigma=5.0:");
    println!("Detected stars and nearest catalog match:\n");

    let mut total_matched = 0;
    for (i, star) in stars.iter().enumerate() {
        let sky = wcs.pixel_to_sky(star.x, star.y);

        // Find nearest catalog star
        let mut best_sep = f64::MAX;
        let mut best_cat: Option<(u32, f32)> = None;

        for (idx, cat_star) in index.stars() {
            let cat_pos = cat_star.to_radec();
            let sep = angular_separation(&sky, &cat_pos).to_degrees() * 3600.0;
            if sep < best_sep {
                best_sep = sep;
                best_cat = Some((idx, cat_star.magnitude()));
            }
        }

        let status = if best_sep < 60.0 { "MATCH" } else { "miss" };
        if best_sep < 60.0 {
            total_matched += 1;
        }

        if i < 30 || best_sep < 60.0 {
            if let Some((cat_idx, mag)) = best_cat {
                println!("  Det {:2}: ({:6.1},{:6.1}) flux={:6.0} -> nearest Cat {} (mag {:.1}) sep={:5.1}\" {}",
                    i, star.x, star.y, star.flux, cat_idx, mag, best_sep, status);
            }
        }
    }

    println!(
        "\nTotal: {} detected, {} matched to catalog ({:.0}%)",
        stars.len(),
        total_matched,
        100.0 * total_matched as f64 / stars.len() as f64
    );

    // Also check: how many of the bright catalog stars were detected?
    println!("\nCatalog stars NOT detected (mag < 6.0 in FOV):");
    let bright_cat: Vec<_> = cat_in_fov.iter().filter(|(_, _, _, m)| *m < 6.0).collect();

    for (idx, cx, cy, mag) in &bright_cat {
        // Check if any detected star is close
        let mut found = false;
        for star in &stars {
            let dx = star.x - cx;
            let dy = star.y - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < 20.0 {
                found = true;
                break;
            }
        }
        if !found && *cx > 0.0 && *cx < 1456.0 && *cy > 0.0 && *cy < 1088.0 {
            println!(
                "  Cat {} at ({:.1}, {:.1}) mag={:.1} - NOT DETECTED",
                idx, cx, cy, mag
            );
        }
    }
}
