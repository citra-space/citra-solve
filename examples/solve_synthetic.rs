//! End-to-end test of the plate solver using synthetic data.
//!
//! This example:
//! 1. Builds a synthetic star index
//! 2. Generates a synthetic star field
//! 3. Attempts to solve it
//! 4. Reports accuracy metrics

use chameleon::catalog::builder::{BuildConfig, IndexBuilder};
use chameleon::catalog::Index;
use chameleon::bench::synthetic::{generate_field, SyntheticConfig};
use chameleon::core::types::RaDec;
use chameleon::core::math::angular_separation_arcsec;
use chameleon::solver::{Solver, SolverConfig};

use std::time::Instant;

fn main() {
    println!("===========================================");
    println!("  Chameleon Plate Solver - E2E Test");
    println!("===========================================\n");

    // Step 1: Build a synthetic index
    println!("Step 1: Building synthetic star index...");
    let index_path = "/tmp/chameleon_e2e_test.idx";

    let build_config = BuildConfig {
        fov_min_deg: 10.0,
        fov_max_deg: 30.0,
        mag_limit: 7.0,
        num_bins: 500_000,
        max_stars: 2000,
        max_patterns_per_star: 50,
    };

    let mut builder = IndexBuilder::new(build_config);

    // Generate synthetic stars uniformly on the sphere
    let num_catalog_stars = 5000;
    let seed: u64 = 12345;
    let mut rng_state = seed;

    let mut rng = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng_state >> 33) as f64 / (1u64 << 31) as f64
    };

    for i in 0..num_catalog_stars {
        let ra = rng() * 2.0 * std::f64::consts::PI;
        let dec = (rng() * 2.0 - 1.0).asin();
        let mag = 1.0 + rng().powf(0.3) * 8.0;
        builder.add_star(i as u32, ra, dec, mag as f32);
    }

    let build_start = Instant::now();
    let stats = builder.build(index_path).expect("Failed to build index");
    let build_time = build_start.elapsed();

    println!("  Built index with {} stars, {} patterns in {:?}",
        stats.num_stars, stats.num_patterns, build_time);

    // Step 2: Load the index
    println!("\nStep 2: Loading index...");
    let index = Index::open(index_path).expect("Failed to open index");
    println!("  Loaded index: {} stars, {} patterns",
        index.num_stars(), index.num_patterns());

    // Step 3: Generate synthetic test fields and solve them
    println!("\nStep 3: Testing plate solver...\n");

    let solver_config = SolverConfig::default();
    let solver = Solver::new(&index, solver_config);

    let image_width = 1024;
    let image_height = 768;
    let fov_deg = 20.0;

    let test_cases = [
        ("Clean field", SyntheticConfig::default()),
        ("Noisy field", SyntheticConfig {
            position_noise_pixels: 1.0,
            ..Default::default()
        }),
        ("Missing stars", SyntheticConfig {
            missing_star_rate: 0.2,
            ..Default::default()
        }),
        ("False stars", SyntheticConfig {
            false_star_count: 10,
            ..Default::default()
        }),
        ("Challenging", SyntheticConfig {
            position_noise_pixels: 1.5,
            missing_star_rate: 0.15,
            false_star_count: 5,
            ..Default::default()
        }),
    ];

    let mut total_solved = 0;
    let mut total_tests = 0;

    for (name, config) in test_cases.iter() {
        println!("  Testing: {}", name);

        // Generate several fields with different random positions
        let mut solved = 0;
        let num_trials = 5;

        for trial in 0..num_trials {
            let trial_config = SyntheticConfig {
                seed: config.seed + trial as u64 * 1000,
                ..config.clone()
            };

            // Pick a random sky position
            let mut trial_rng = trial_config.seed;
            trial_rng = trial_rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let ra = (trial_rng as f64 / u64::MAX as f64) * 2.0 * std::f64::consts::PI;
            trial_rng = trial_rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let dec = ((trial_rng as f64 / u64::MAX as f64) * 2.0 - 1.0).asin();
            let center = RaDec::new(ra, dec);

            // Generate field
            let field = generate_field(
                &index,
                center,
                fov_deg,
                image_width,
                image_height,
                &trial_config,
            );

            if field.detected_stars.len() < 4 {
                println!("    Trial {}: Skipped (only {} stars visible)",
                    trial + 1, field.detected_stars.len());
                continue;
            }

            // Solve
            let solve_start = Instant::now();
            let result = solver.solve(&field.detected_stars, image_width, image_height);
            let solve_time = solve_start.elapsed();

            match result {
                Ok(solution) => {
                    let error = angular_separation_arcsec(&field.center, &solution.center);
                    println!("    Trial {}: SOLVED in {:?}, error={:.1}\", {} matched",
                        trial + 1, solve_time, error, solution.num_matched_stars);
                    solved += 1;
                }
                Err(e) => {
                    println!("    Trial {}: FAILED - {}", trial + 1, e);
                }
            }
        }

        println!("    Result: {}/{} solved\n", solved, num_trials);
        total_solved += solved;
        total_tests += num_trials;
    }

    // Summary
    println!("===========================================");
    println!("  SUMMARY");
    println!("===========================================");
    println!("  Total solve rate: {}/{} ({:.1}%)",
        total_solved, total_tests,
        100.0 * total_solved as f64 / total_tests as f64);
    println!("===========================================\n");

    // Cleanup
    std::fs::remove_file(index_path).ok();
}
