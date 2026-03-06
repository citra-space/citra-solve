//! Index builder CLI tool.
//!
//! Builds a Chameleon pattern index from a star catalog file.
//!
//! # Usage
//!
//! ```bash
//! # Build from Hipparcos catalog
//! build-index --catalog hip_main.dat --output hipparcos.idx
//!
//! # With custom parameters
//! build-index --catalog hip_main.dat --output wide.idx \
//!     --fov-min 10 --fov-max 30 --mag-limit 7.0
//! ```

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::ExitCode;

use citra_solve::catalog::builder::{BuildConfig, IndexBuilder};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 || args.contains(&"--help".to_string()) || args.contains(&"-h".to_string()) {
        print_usage();
        return ExitCode::from(0);
    }

    // Parse arguments
    let mut catalog_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut config = BuildConfig::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--catalog" | "-c" => {
                i += 1;
                if i < args.len() {
                    catalog_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--output" | "-o" => {
                i += 1;
                if i < args.len() {
                    output_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--fov-min" => {
                i += 1;
                if i < args.len() {
                    config.fov_min_deg = args[i].parse().unwrap_or(config.fov_min_deg);
                }
            }
            "--fov-max" => {
                i += 1;
                if i < args.len() {
                    config.fov_max_deg = args[i].parse().unwrap_or(config.fov_max_deg);
                }
            }
            "--mag-limit" => {
                i += 1;
                if i < args.len() {
                    config.mag_limit = args[i].parse().unwrap_or(config.mag_limit);
                }
            }
            "--max-stars" => {
                i += 1;
                if i < args.len() {
                    config.max_stars = args[i].parse().unwrap_or(config.max_stars);
                }
            }
            "--patterns-per-star" => {
                i += 1;
                if i < args.len() {
                    config.max_patterns_per_star =
                        args[i].parse().unwrap_or(config.max_patterns_per_star);
                }
            }
            "--synthetic" => {
                // Generate synthetic catalog for testing
                return build_synthetic_index(output_path, config);
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
            }
        }
        i += 1;
    }

    // Validate required arguments
    let catalog_path = match catalog_path {
        Some(p) => p,
        None => {
            eprintln!("Error: --catalog is required (or use --synthetic for test data)");
            print_usage();
            return ExitCode::from(1);
        }
    };

    let output_path = output_path.unwrap_or_else(|| PathBuf::from("chameleon.idx"));

    println!("Chameleon Index Builder");
    println!("=======================");
    println!("Catalog: {}", catalog_path.display());
    println!("Output:  {}", output_path.display());
    println!(
        "FOV:     {:.1}° - {:.1}°",
        config.fov_min_deg, config.fov_max_deg
    );
    println!("Mag limit: {:.1}", config.mag_limit);
    println!();

    // Build index
    match build_index(&catalog_path, &output_path, config) {
        Ok(()) => {
            println!("\nIndex built successfully!");
            ExitCode::from(0)
        }
        Err(e) => {
            eprintln!("\nError building index: {}", e);
            ExitCode::from(1)
        }
    }
}

fn print_usage() {
    println!("Chameleon Index Builder");
    println!();
    println!("Usage: build-index [OPTIONS]");
    println!();
    println!("Options:");
    println!("  -c, --catalog <FILE>    Input star catalog file (Hipparcos format)");
    println!("  -o, --output <FILE>     Output index file [default: chameleon.idx]");
    println!("  --fov-min <DEG>         Minimum FOV in degrees [default: 10.0]");
    println!("  --fov-max <DEG>         Maximum FOV in degrees [default: 30.0]");
    println!("  --mag-limit <MAG>       Magnitude limit [default: 7.0]");
    println!("  --max-stars <N>         Max stars for pattern generation [default: 5000]");
    println!("  --patterns-per-star <N> Max patterns per star [default: 5000]");
    println!("  --synthetic             Generate synthetic test catalog");
    println!("  -h, --help              Show this help message");
    println!();
    println!("Catalog formats:");
    println!("  Hipparcos: Standard hip_main.dat format");
    println!("  CSV: RA(deg),Dec(deg),Mag,ID");
}

fn build_index(
    catalog_path: &PathBuf,
    output_path: &PathBuf,
    config: BuildConfig,
) -> Result<(), String> {
    let mut builder = IndexBuilder::new(config);

    // Detect catalog format and parse
    let file = File::open(catalog_path).map_err(|e| format!("Failed to open catalog: {}", e))?;
    let reader = BufReader::new(file);

    let extension = catalog_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    let stars_added = if extension == "csv" {
        parse_csv_catalog(reader, &mut builder)?
    } else {
        // Assume Hipparcos format
        parse_hipparcos_catalog(reader, &mut builder)?
    };

    println!("Parsed {} stars from catalog", stars_added);

    if stars_added == 0 {
        return Err("No stars found in catalog".to_string());
    }

    // Build the index
    let stats = builder
        .build(output_path)
        .map_err(|e| format!("Build failed: {}", e))?;

    println!("\nBuild Statistics:");
    println!("  Stars in index: {}", stats.num_stars);
    println!("  Patterns generated: {}", stats.num_patterns);
    println!("  Hash bins used: {}", stats.num_bins_used);
    println!("  Avg patterns/bin: {:.2}", stats.avg_patterns_per_bin);

    Ok(())
}

fn parse_hipparcos_catalog<R: BufRead>(
    reader: R,
    builder: &mut IndexBuilder,
) -> Result<usize, String> {
    let mut count = 0;

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result.map_err(|e| format!("Read error at line {}: {}", line_num, e))?;

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Hipparcos format is fixed-width:
        // Columns 9-14: HIP number
        // Columns 52-63: RA (degrees)
        // Columns 65-76: Dec (degrees)
        // Columns 42-46: Vmag

        if line.len() < 76 {
            continue;
        }

        let hip_str = line.get(8..14).unwrap_or("").trim();
        let ra_str = line.get(51..63).unwrap_or("").trim();
        let dec_str = line.get(64..76).unwrap_or("").trim();
        let mag_str = line.get(41..46).unwrap_or("").trim();

        let hip: u32 = match hip_str.parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let ra: f64 = match ra_str.parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let dec: f64 = match dec_str.parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let mag: f32 = match mag_str.parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Convert degrees to radians
        let ra_rad = ra.to_radians();
        let dec_rad = dec.to_radians();

        builder.add_star(hip, ra_rad, dec_rad, mag);
        count += 1;
    }

    Ok(count)
}

fn parse_csv_catalog<R: BufRead>(reader: R, builder: &mut IndexBuilder) -> Result<usize, String> {
    let mut count = 0;

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result.map_err(|e| format!("Read error at line {}: {}", line_num, e))?;

        // Skip empty lines and headers
        if line.is_empty() || line.starts_with('#') || line.starts_with("RA") {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 3 {
            continue;
        }

        let ra: f64 = match parts[0].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let dec: f64 = match parts[1].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let mag: f32 = match parts[2].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let id: u32 = if parts.len() > 3 {
            parts[3].trim().parse().unwrap_or(count as u32)
        } else {
            count as u32
        };

        let ra_rad = ra.to_radians();
        let dec_rad = dec.to_radians();

        builder.add_star(id, ra_rad, dec_rad, mag);
        count += 1;
    }

    Ok(count)
}

fn build_synthetic_index(output_path: Option<PathBuf>, config: BuildConfig) -> ExitCode {
    println!("Generating synthetic star catalog for testing...");

    let output_path = output_path.unwrap_or_else(|| PathBuf::from("synthetic.idx"));

    let mut builder = IndexBuilder::new(config);

    // Generate a synthetic sky with randomly distributed stars
    use std::f64::consts::PI;

    let seed: u64 = 42;
    let mut rng_state = seed;

    // Simple LCG random number generator
    let mut rng = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng_state >> 33) as f64 / (1u64 << 31) as f64
    };

    // Generate ~10000 stars with realistic magnitude distribution
    let num_stars = 10000;
    for i in 0..num_stars {
        // Uniform on sphere: RA uniform, Dec = asin(uniform[-1,1])
        let ra = rng() * 2.0 * PI;
        let dec = (rng() * 2.0 - 1.0).asin();

        // Magnitude distribution (more faint stars than bright)
        let mag = 1.0 + rng().powf(0.3) * 8.0; // Range ~1 to ~9

        builder.add_star(i as u32, ra, dec, mag as f32);
    }

    println!("Generated {} synthetic stars", num_stars);

    match builder.build(&output_path) {
        Ok(stats) => {
            println!("\nSynthetic Index Built:");
            println!("  Output: {}", output_path.display());
            println!("  Stars: {}", stats.num_stars);
            println!("  Patterns: {}", stats.num_patterns);
            ExitCode::from(0)
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            ExitCode::from(1)
        }
    }
}
