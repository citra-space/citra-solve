//! Debug SIP distortion fitting.
//!
//! Tests whether adding SIP distortion coefficients improves the WCS accuracy.

use chameleon::catalog::Index;
use chameleon::core::types::RaDec;
use chameleon::extract::{extract_stars, ExtractionConfig};
use chameleon::pattern::{generate_quads, PatternMatcher};
use chameleon::solver::hypothesis::generate_hypotheses;
use chameleon::solver::verify::{verify_hypothesis, VerifyConfig};
use chameleon::solver::refine::{fit_sip, refine_linear_wcs};
use chameleon::wcs::Wcs;

fn main() {
    let image_path = "exposure_5.jpg";
    let index_path = "hipparcos_deep.idx";

    println!("Debug: SIP Distortion Fitting\n");

    // Load index
    let index = Index::open(index_path).expect("Failed to open index");
    println!("Index: {} stars, {} patterns", index.num_stars(), index.num_patterns());

    // Known true WCS for comparison
    let true_crval = RaDec::from_degrees(14.3012483353, -9.12424543834);
    let true_crpix = (903.227925618, 575.047943115);
    let true_cd = [
        [-0.0159317827922, 0.00630093735133],
        [-0.00629294246889, -0.0159290388907],
    ];
    let true_wcs = Wcs::new(true_crpix, true_crval, true_cd);
    println!("True WCS scale: {:.2} arcsec/px\n", true_wcs.pixel_scale_arcsec());

    // Extract stars
    let extract_config = ExtractionConfig {
        max_stars: 50,
        sigma_threshold: 5.0,
        ..Default::default()
    };
    let stars = extract_stars(image_path, &extract_config).expect("Failed to extract stars");
    println!("Extracted {} stars", stars.len());

    // Generate quads and find matches
    let quads = generate_quads(&stars, 30, 100);
    println!("Generated {} quads", quads.len());

    let matcher = PatternMatcher::new(&index)
        .with_bin_tolerance(0.03)
        .with_ratio_tolerance(0.03);

    let matches = matcher.find_matches_batch(&quads, 10);
    println!("Found {} pattern matches", matches.len());

    let top_matches: Vec<_> = matches.into_iter().take(500).collect();

    // Generate and verify hypotheses
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

    // Find the best hypothesis
    let verify_config = VerifyConfig {
        max_match_distance_pixels: 20.0,
        position_sigma_pixels: 10.0,
        ..Default::default()
    };

    let mut best_result = None;
    let mut best_log_odds = f64::NEG_INFINITY;

    for hypothesis in &hypotheses {
        let result = verify_hypothesis(
            hypothesis,
            &stars,
            &index,
            image_width,
            image_height,
            &verify_config,
        );
        if result.hypothesis.log_odds > best_log_odds {
            best_log_odds = result.hypothesis.log_odds;
            best_result = Some(result);
        }
    }

    let best = best_result.expect("No valid hypothesis found");
    println!("Best hypothesis:");
    println!("  Matched: {} stars", best.num_matched);
    println!("  RMS residual: {:.2} pixels", best.rms_residual_pixels);
    println!("  Log-odds: {:.2}", best.hypothesis.log_odds);

    let initial_wcs = &best.hypothesis.wcs;
    let center = initial_wcs.crval();
    println!("  Center: RA={:.4}°, Dec={:.4}°", center.ra_deg(), center.dec_deg());
    println!("  Scale: {:.2} arcsec/px\n", initial_wcs.pixel_scale_arcsec());

    // First, refine the linear WCS using all matched stars
    println!("=== Step 1: Refine Linear WCS ===\n");

    let refined = refine_linear_wcs(
        &stars,
        &best.hypothesis.star_matches,
        initial_wcs,
        image_width,
        image_height,
    );

    let wcs = if let Some(result) = refined {
        println!("Linear WCS refinement:");
        println!("  RMS before: {:.3} pixels", result.rms_before_pixels);
        println!("  RMS after:  {:.3} pixels", result.rms_after_pixels);
        println!("  Stars used: {}", result.num_stars_used);

        let center = result.wcs.crval();
        println!("  Refined center: RA={:.4}°, Dec={:.4}°", center.ra_deg(), center.dec_deg());
        println!("  Refined scale: {:.2} arcsec/px", result.wcs.pixel_scale_arcsec());

        // Compare with true WCS
        let img_center = (image_width as f64 / 2.0, image_height as f64 / 2.0);
        let sky_refined = result.wcs.pixel_to_sky(img_center.0, img_center.1);
        let sky_true = true_wcs.pixel_to_sky(img_center.0, img_center.1);
        let err = angular_distance_arcsec(&sky_refined, &sky_true);
        println!("  Error at image center: {:.1}\"", err);
        println!();

        result.wcs
    } else {
        println!("Linear refinement failed, using initial WCS\n");
        initial_wcs.clone()
    };

    // Now fit SIP distortion
    println!("=== Step 2: Fit SIP Distortion ===\n");

    for order in 2..=4 {
        let sip_result = fit_sip(
            &stars,
            &best.hypothesis.star_matches,
            &wcs,
            order,
        );

        println!("SIP Order {}:", order);
        println!("  RMS before: {:.3} pixels", sip_result.rms_before_pixels);
        println!("  RMS after:  {:.3} pixels", sip_result.rms_after_pixels);
        let improvement = (1.0 - sip_result.rms_after_pixels / sip_result.rms_before_pixels) * 100.0;
        println!("  Improvement: {:.1}%", improvement);
        println!("  Stars used: {}", sip_result.num_stars_used);

        // Test at image corners
        let corners = [
            (0.0, 0.0),
            (image_width as f64, 0.0),
            (image_width as f64, image_height as f64),
            (0.0, image_height as f64),
            (image_width as f64 / 2.0, image_height as f64 / 2.0),
        ];

        println!("  Position comparison:");
        for (x, y) in &corners {
            let sky_linear = wcs.pixel_to_sky(*x, *y);
            let sky_sip = sip_result.wcs.pixel_to_sky(*x, *y);
            let sky_true = true_wcs.pixel_to_sky(*x, *y);

            // Compute error in arcsec
            let err_linear = angular_distance_arcsec(&sky_linear, &sky_true);
            let err_sip = angular_distance_arcsec(&sky_sip, &sky_true);

            println!("    ({:4.0},{:4.0}): linear err={:5.1}\", SIP err={:5.1}\"",
                x, y, err_linear, err_sip);
        }
        println!();
    }

    // Output FITS header for best solution
    println!("=== FITS Header (Order 3 SIP) ===");
    let sip_result = fit_sip(&stars, &best.hypothesis.star_matches, &wcs, 3);
    println!("{}", sip_result.wcs.to_fits_header());
}

fn angular_distance_arcsec(a: &RaDec, b: &RaDec) -> f64 {
    use chameleon::core::math::angular_separation;
    angular_separation(a, b).to_degrees() * 3600.0
}
