//! Debug centroids by comparing detected positions to projected catalog positions.

use citra_solve::catalog::Index;
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
    let pixel_scale = 61.67; // arcsec per pixel

    println!("Debug: Centroid Analysis\n");
    println!("Pixel scale: {:.2}\"/px\n", pixel_scale);

    // Load index and find catalog stars in FOV
    let index = Index::open(index_path).expect("Failed to open index");

    let mut cat_in_fov: Vec<(u32, f64, f64, f32)> = Vec::new();
    for (idx, star) in index.stars() {
        let pos = star.to_radec();
        let (px, py) = wcs.sky_to_pixel(&pos);
        if px >= 0.0 && px <= 1456.0 && py >= 0.0 && py <= 1088.0 {
            cat_in_fov.push((idx, px, py, star.magnitude()));
        }
    }
    cat_in_fov.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap());

    // Extract stars
    let config = ExtractionConfig {
        max_stars: 100,
        sigma_threshold: 5.0,
        min_separation: 10.0,
        ..Default::default()
    };
    let stars = extract_stars(image_path, &config).expect("Failed to extract");

    println!("Catalog stars in FOV: {}", cat_in_fov.len());
    println!("Extracted stars: {}\n", stars.len());

    // For each catalog star, find nearest detected star
    println!("Catalog stars (brightest first) and nearest detection:");
    println!(
        "{:>6} {:>8} {:>8} {:>4}  {:>8} {:>8} {:>6} {:>8}",
        "Cat", "Cat_X", "Cat_Y", "Mag", "Det_X", "Det_Y", "Dist", "Status"
    );
    println!("{}", "-".repeat(75));

    let mut matched = 0;
    let mut total_err = 0.0;

    for (idx, cx, cy, mag) in cat_in_fov.iter().take(30) {
        // Find nearest detected star by pixel distance
        let mut best_dist = f64::MAX;
        let mut best_det: Option<(f64, f64)> = None;

        for star in &stars {
            let dx = star.x - cx;
            let dy = star.y - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < best_dist {
                best_dist = dist;
                best_det = Some((star.x, star.y));
            }
        }

        let status = if best_dist < 5.0 {
            matched += 1;
            total_err += best_dist;
            "GOOD"
        } else if best_dist < 15.0 {
            matched += 1;
            total_err += best_dist;
            "OK"
        } else {
            "MISS"
        };

        if let Some((dx, dy)) = best_det {
            println!(
                "{:>6} {:>8.1} {:>8.1} {:>4.1}  {:>8.1} {:>8.1} {:>6.1} {:>8}",
                idx, cx, cy, mag, dx, dy, best_dist, status
            );
        } else {
            println!(
                "{:>6} {:>8.1} {:>8.1} {:>4.1}  {:>8} {:>8} {:>6} {:>8}",
                idx, cx, cy, mag, "-", "-", "-", "MISS"
            );
        }
    }

    println!(
        "\nMatched: {} of {} (threshold: 15 pixels)",
        matched,
        30.min(cat_in_fov.len())
    );
    if matched > 0 {
        println!(
            "Average error: {:.2} pixels ({:.1}\")",
            total_err / matched as f64,
            total_err / matched as f64 * pixel_scale
        );
    }

    // Now show detected stars that DON'T match any catalog star
    println!("\nDetected stars with no catalog match within 20 pixels:");
    let mut unmatched = 0;
    for (i, star) in stars.iter().enumerate() {
        let mut nearest_dist = f64::MAX;
        for (_, cx, cy, _) in &cat_in_fov {
            let dx = star.x - cx;
            let dy = star.y - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < nearest_dist {
                nearest_dist = dist;
            }
        }

        if nearest_dist > 20.0 {
            unmatched += 1;
            if unmatched <= 20 {
                println!(
                    "  Det {:2}: ({:6.1},{:6.1}) flux={:5.0}  nearest_cat={:.0}px",
                    i, star.x, star.y, star.flux, nearest_dist
                );
            }
        }
    }
    println!(
        "Total unmatched detections: {} of {}",
        unmatched,
        stars.len()
    );
}
