//! Verify the astrometry.net solution by projecting detected stars.

use citra_solve::catalog::Index;
use citra_solve::core::math::angular_separation;
use citra_solve::core::types::RaDec;
use citra_solve::extract::{extract_stars, ExtractionConfig};
use citra_solve::wcs::Wcs;

fn main() {
    let image_path = "exposure_5.jpg";
    let index_path = "hipparcos.idx";

    // Load index first
    let index = Index::open(index_path).expect("Failed to open index");

    // Astrometry.net solution
    // Center: (16.912, -7.531) deg
    // Size: 24.9 x 18.6 deg for 1456 x 1088 pixels
    // Pixel scale: 61.7 arcsec/pixel
    // Orientation: Up is 21.6 degrees E of N

    let crval = RaDec::from_degrees(16.912, -7.531);
    let pixel_scale = 61.7 / 3600.0; // degrees per pixel
    let crpix = (1456.0 / 2.0, 1088.0 / 2.0);

    println!("Testing different rotation interpretations...\n");

    // Try different rotation interpretations and CD matrix signs
    let test_cases = [
        ("Standard, rot=-21.6", -21.6, false, false),
        ("Standard, rot=21.6", 21.6, false, false),
        ("Standard, rot=0", 0.0, false, false),
        ("Flip RA, rot=-21.6", -21.6, true, false),
        ("Flip RA, rot=21.6", 21.6, true, false),
        ("Flip Dec, rot=-21.6", -21.6, false, true),
        ("Flip both, rot=-21.6", -21.6, true, true),
        ("Flip both, rot=21.6", 21.6, true, true),
    ];

    for (name, rot_deg, flip_ra, flip_dec) in test_cases {
        let rotation = (rot_deg as f64).to_radians();
        let cos_r = rotation.cos();
        let sin_r = rotation.sin();

        let ra_sign = if flip_ra { 1.0 } else { -1.0 };
        let dec_sign = if flip_dec { -1.0 } else { 1.0 };

        let cd = [
            [ra_sign * pixel_scale * cos_r, ra_sign * pixel_scale * sin_r],
            [
                dec_sign * pixel_scale * sin_r,
                dec_sign * pixel_scale * cos_r,
            ],
        ];

        let wcs = Wcs::new(crpix, crval, cd);

        // Test with brightest star at (1273.5, 991.7)
        let sky = wcs.pixel_to_sky(1273.5, 991.7);

        // Find nearest catalog star
        let mut best_sep = f64::MAX;
        for (_idx, cat_star) in index.stars() {
            let cat_pos = cat_star.to_radec();
            let sep = angular_separation(&sky, &cat_pos);
            if sep < best_sep {
                best_sep = sep;
            }
        }
        let sep_arcsec = best_sep.to_degrees() * 3600.0;
        println!(
            "{}: star -> ({:.2}°, {:.2}°), sep={:.0}\"",
            name,
            sky.ra_deg(),
            sky.dec_deg(),
            sep_arcsec
        );
    }

    println!("\n--- Now testing with best configuration ---\n");

    // Based on results, use the best configuration
    // Standard astrometry: RA increases left (negative X), Dec increases up (positive Y)
    let rotation = (-21.6_f64).to_radians();
    let cos_r = rotation.cos();
    let sin_r = rotation.sin();

    let cd = [
        [-pixel_scale * cos_r, -pixel_scale * sin_r],
        [pixel_scale * sin_r, pixel_scale * cos_r],
    ];

    let wcs = Wcs::new(crpix, crval, cd);

    println!("Astrometry.net WCS (adjusted):");
    println!("  CRVAL: ({:.4}°, {:.4}°)", 16.912, -7.531);
    println!("  Pixel scale: {:.2}\"/px", 61.7);
    println!();

    // Extract stars
    let extract_config = ExtractionConfig {
        max_stars: 100,
        sigma_threshold: 5.0,
        ..Default::default()
    };
    let stars = extract_stars(image_path, &extract_config).expect("Failed to extract stars");
    println!("Extracted {} stars\n", stars.len());

    // Project and match
    println!("Matching detected stars to catalog (threshold 120\"):");
    let match_threshold = 120.0;

    let mut matched = 0;
    let mut total_sep = 0.0;

    for (i, star) in stars.iter().take(20).enumerate() {
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
        let status = if sep_arcsec < match_threshold {
            "MATCH"
        } else {
            "miss"
        };

        if sep_arcsec < match_threshold {
            matched += 1;
            total_sep += sep_arcsec;
        }

        println!(
            "  Star {:2} ({:6.1},{:6.1}) -> ({:6.2}°,{:6.2}°) sep={:5.0}\" {}",
            i,
            star.x,
            star.y,
            sky.ra_deg(),
            sky.dec_deg(),
            sep_arcsec,
            status
        );
    }

    println!("\nMatched {}/20 stars", matched);
    if matched > 0 {
        println!("Average separation: {:.1}\"", total_sep / matched as f64);
    }
}
