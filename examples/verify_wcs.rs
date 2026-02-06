//! Verify WCS implementation against astrometry.net solution.

use citra_solve::catalog::Index;
use citra_solve::core::types::RaDec;
use citra_solve::core::math::angular_separation;
use citra_solve::extract::{extract_stars, ExtractionConfig};
use citra_solve::wcs::Wcs;

fn main() {
    let image_path = "exposure_5.jpg";
    let index_path = "hipparcos.idx";

    // Exact WCS from astrometry.net wcs.fits
    let crval = RaDec::from_degrees(14.3012483353, -9.12424543834);
    let crpix = (903.227925618, 575.047943115);
    let cd = [
        [-0.0159317827922, 0.00630093735133],
        [-0.00629294246889, -0.0159290388907],
    ];

    let wcs = Wcs::new(crpix, crval, cd);

    println!("Astrometry.net WCS (exact from FITS):");
    println!("  CRVAL: ({:.6}°, {:.6}°)", 14.3012483353, -9.12424543834);
    println!("  CRPIX: ({:.2}, {:.2})", 903.23, 575.05);
    println!("  CD matrix:");
    println!("    [{:+.8}, {:+.8}]", cd[0][0], cd[0][1]);
    println!("    [{:+.8}, {:+.8}]", cd[1][0], cd[1][1]);
    println!("  Pixel scale: {:.2}\"/px", wcs.pixel_scale_arcsec());
    println!();

    // Load index
    let index = Index::open(index_path).expect("Failed to open index");
    println!("Index: {} stars\n", index.num_stars());

    // Extract stars
    let extract_config = ExtractionConfig {
        max_stars: 100,
        sigma_threshold: 5.0,
        ..Default::default()
    };
    let stars = extract_stars(image_path, &extract_config).expect("Failed to extract stars");
    println!("Extracted {} stars\n", stars.len());

    // Project each detected star to sky and find nearest catalog star
    println!("Matching detected stars to catalog:");
    let match_threshold = 120.0; // arcseconds

    let mut matched = 0;
    let mut total_sep = 0.0;

    for (i, star) in stars.iter().take(30).enumerate() {
        let sky = wcs.pixel_to_sky(star.x, star.y);

        // Find nearest catalog star
        let mut best_sep = f64::MAX;
        let mut best_idx = 0;
        let mut best_mag = 0.0f32;

        for (idx, cat_star) in index.stars() {
            let cat_pos = cat_star.to_radec();
            let sep = angular_separation(&sky, &cat_pos);
            if sep < best_sep {
                best_sep = sep;
                best_idx = idx;
                best_mag = cat_star.magnitude();
            }
        }

        let sep_arcsec = best_sep.to_degrees() * 3600.0;
        let status = if sep_arcsec < match_threshold { "MATCH" } else { "miss" };

        if sep_arcsec < match_threshold {
            matched += 1;
            total_sep += sep_arcsec;
        }

        println!("  Star {:2} ({:6.1},{:6.1}) -> ({:7.3}°,{:7.3}°) nearest: star {} (mag {:.1}) sep={:5.1}\" {}",
            i, star.x, star.y, sky.ra_deg(), sky.dec_deg(), best_idx, best_mag, sep_arcsec, status);
    }

    println!("\n========================================");
    println!("Matched {}/30 stars (threshold: {}\")", matched, match_threshold);
    if matched > 0 {
        println!("Average separation: {:.1}\"", total_sep / matched as f64);
    }
    println!("========================================");

    // Also test a few specific pixel positions
    println!("\nTest pixel positions:");
    let test_pixels = [
        (0.0, 0.0, "top-left"),
        (1456.0, 0.0, "top-right"),
        (1456.0, 1088.0, "bottom-right"),
        (0.0, 1088.0, "bottom-left"),
        (728.0, 544.0, "center"),
        (crpix.0, crpix.1, "CRPIX"),
    ];

    for (x, y, name) in test_pixels {
        let sky = wcs.pixel_to_sky(x, y);
        println!("  {} ({:.0},{:.0}) -> ({:.4}°, {:.4}°)", name, x, y, sky.ra_deg(), sky.dec_deg());
    }
}
