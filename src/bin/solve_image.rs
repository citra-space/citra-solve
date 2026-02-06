//! Solve a real image using the Chameleon plate solver.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --bin solve-image -- --image exposure.jpg --index hipparcos.idx
//! ```

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use citra_solve::catalog::Index;
use citra_solve::extract::{extract_stars, ExtractionConfig};
use citra_solve::solver::{Solver, SolverConfig};
use image::GenericImageView;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 || args.contains(&"--help".to_string()) || args.contains(&"-h".to_string()) {
        print_usage();
        return ExitCode::from(0);
    }

    // Parse arguments
    let mut image_path: Option<PathBuf> = None;
    let mut index_path: Option<PathBuf> = None;
    let mut solver_config = SolverConfig::default();
    let mut extract_config = ExtractionConfig::default();
    let mut verbose = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--image" | "-i" => {
                i += 1;
                if i < args.len() {
                    image_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--index" | "-x" => {
                i += 1;
                if i < args.len() {
                    index_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--sigma" => {
                i += 1;
                if i < args.len() {
                    extract_config.sigma_threshold = args[i].parse().unwrap_or(5.0);
                }
            }
            "--max-stars" => {
                i += 1;
                if i < args.len() {
                    let n: usize = args[i].parse().unwrap_or(100);
                    extract_config.max_stars = n;
                    solver_config.max_stars = n;
                }
            }
            "--tolerance" => {
                i += 1;
                if i < args.len() {
                    let t: f64 = args[i].parse().unwrap_or(0.05);
                    solver_config.bin_tolerance = t;
                    solver_config.ratio_tolerance = t;
                }
            }
            "--verbose" | "-v" => {
                verbose = true;
            }
            _ => {
                if !args[i].starts_with('-') && image_path.is_none() {
                    image_path = Some(PathBuf::from(&args[i]));
                } else if !args[i].starts_with('-') && index_path.is_none() {
                    index_path = Some(PathBuf::from(&args[i]));
                }
            }
        }
        i += 1;
    }

    let image_path = match image_path {
        Some(p) => p,
        None => {
            eprintln!("Error: --image is required");
            print_usage();
            return ExitCode::from(1);
        }
    };

    let index_path = index_path.unwrap_or_else(|| PathBuf::from("hipparcos.idx"));

    println!("Chameleon Plate Solver");
    println!("======================");
    println!("Image: {}", image_path.display());
    println!("Index: {}", index_path.display());
    println!();

    // Load index
    println!("Loading index...");
    let index = match Index::open(&index_path) {
        Ok(idx) => idx,
        Err(e) => {
            eprintln!("Failed to load index: {}", e);
            return ExitCode::from(1);
        }
    };
    println!("  {} stars, {} patterns", index.num_stars(), index.num_patterns());

    // Extract stars from image
    println!("\nExtracting stars...");
    let extract_start = Instant::now();
    let stars = match extract_stars(&image_path, &extract_config) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to extract stars: {}", e);
            return ExitCode::from(1);
        }
    };
    let extract_time = extract_start.elapsed();
    println!("  Extracted {} stars in {:?}", stars.len(), extract_time);

    if verbose {
        println!("\n  Top 10 brightest stars:");
        for (i, star) in stars.iter().take(10).enumerate() {
            println!("    {}: ({:.1}, {:.1}) flux={:.0}", i + 1, star.x, star.y, star.flux);
        }
    }

    if stars.len() < 4 {
        eprintln!("\nError: Need at least 4 stars to solve, found {}", stars.len());
        return ExitCode::from(1);
    }

    // Get image dimensions from the image crate
    let img = image::open(&image_path).expect("Already opened successfully");
    let (width, height) = img.dimensions();
    println!("\nImage size: {}x{}", width, height);

    // Solve
    println!("\nSolving...");
    // Save config values before moving
    let max_stars_used = solver_config.max_stars;
    let bin_tol = solver_config.bin_tolerance;
    let ratio_tol = solver_config.ratio_tolerance;

    let solver = Solver::new(&index, solver_config);
    let solve_start = Instant::now();
    let result = solver.solve(&stars, width, height);
    let solve_time = solve_start.elapsed();

    match result {
        Ok(solution) => {
            println!("\n========== SOLUTION ==========");
            println!("Center: RA = {:.6}° ({:.4}h), Dec = {:.6}°",
                solution.center.ra_deg(),
                solution.center.ra_deg() / 15.0,
                solution.center.dec_deg());
            println!("Field of view: {:.3}° x {:.3}°",
                solution.fov_width_deg,
                solution.fov_height_deg);
            println!("Rotation: {:.2}°", solution.rotation_deg);
            println!("Pixel scale: {:.3}\"/px", solution.pixel_scale_arcsec);
            println!("Matched stars: {}", solution.num_matched_stars);
            println!("Solve time: {:?}", solve_time);
            println!("==============================\n");

            // Print WCS header
            if verbose {
                println!("WCS Header:");
                println!("{}", solution.wcs.to_fits_header());
            }

            ExitCode::from(0)
        }
        Err(e) => {
            eprintln!("\nFailed to solve: {}", e);
            eprintln!("Solve time: {:?}", solve_time);

            if verbose {
                eprintln!("\nDebug info:");
                eprintln!("  Stars used: {}", stars.len().min(max_stars_used));
                eprintln!("  Tolerance: bin={}, ratio={}", bin_tol, ratio_tol);
            }

            ExitCode::from(1)
        }
    }
}

fn print_usage() {
    println!("Chameleon Plate Solver");
    println!();
    println!("Usage: solve-image [OPTIONS] --image <FILE>");
    println!();
    println!("Options:");
    println!("  -i, --image <FILE>    Input image file (JPEG, PNG)");
    println!("  -x, --index <FILE>    Star pattern index [default: hipparcos.idx]");
    println!("  --sigma <N>           Detection threshold in sigma [default: 5.0]");
    println!("  --max-stars <N>       Maximum stars to use [default: 100]");
    println!("  --tolerance <T>       Pattern matching tolerance [default: 0.05]");
    println!("  -v, --verbose         Show detailed output");
    println!("  -h, --help            Show this help message");
}
