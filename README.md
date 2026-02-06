# Chameleon

Efficient lost-in-space astrometric plate solver for embedded systems.

## Features

- **Low memory footprint**: ~16KB runtime heap, memory-mapped index files
- **Wide FOV support**: 10-30+ degree fields with SIP distortion modeling
- **Robust matching**: Handles false stars (noise, hot pixels) and missing stars
- **Fast solving**: O(1) hash table lookups with early termination
- **Pure Rust**: No external dependencies for core solving

## Quick Start

### Building an Index

First, build a star pattern index from a catalog:

```bash
# Build from Hipparcos catalog
cargo run --bin build-index -- --catalog hip_main.dat --output hipparcos.idx

# Or generate a synthetic test index
cargo run --bin build-index -- --synthetic --output test.idx
```

### Using the Solver

```rust
use chameleon::{Solver, SolverConfig, DetectedStar, Index};

// Load a pre-built index
let index = Index::open("hipparcos.idx")?;

// Create solver with default config
let solver = Solver::new(&index, SolverConfig::default());

// Your detected stars from image (sorted by brightness)
let stars = vec![
    DetectedStar::new(512.0, 384.0, 1000.0),  // x, y, flux
    DetectedStar::new(100.0, 200.0, 800.0),
    // ... more stars
];

// Solve!
match solver.solve(&stars, 1024, 768) {
    Ok(solution) => {
        println!("Center: RA={:.4}° Dec={:.4}°",
            solution.center.ra_deg(),
            solution.center.dec_deg());
        println!("FOV: {:.2}° x {:.2}°",
            solution.fov_width_deg,
            solution.fov_height_deg);
        println!("Rotation: {:.2}°", solution.rotation_deg);
        println!("Matched {} stars", solution.num_matched_stars);
    }
    Err(e) => eprintln!("Failed to solve: {}", e),
}
```

## Architecture

Chameleon uses a **geometric hashing** approach inspired by [astrometry.net](https://astrometry.net) and [tetra3](https://github.com/esa/tetra3):

1. **Pattern Generation**: Extract 4-star "quads" from detected stars
2. **Geometric Hash**: Compute scale/rotation invariant features (5 normalized edge ratios)
3. **Hash Lookup**: O(1) lookup in pre-built index
4. **Hypothesis Generation**: Generate WCS hypotheses from matched patterns
5. **Bayesian Verification**: Verify hypotheses using log-odds framework
6. **Refinement**: Iterative least-squares refinement with outlier rejection

### Module Structure

```
src/
├── core/          # Types (RaDec, Vec3) and math utilities
├── catalog/       # Index format, builder, and memory-mapped reader
├── pattern/       # Quad extraction and geometric hashing
├── solver/        # Hypothesis generation, verification, refinement
├── wcs/           # TAN projection and SIP distortion
└── bench/         # Benchmarking harness and synthetic data
```

## Index Format

The index file uses a compact binary format:

| Section | Description |
|---------|-------------|
| Header (64B) | Magic, version, star/pattern counts, FOV range |
| Stars | Packed 12-byte entries (RA, Dec, magnitude) |
| Bin offsets | u32 offsets for each hash bin |
| Patterns | Packed 18-byte entries (4 star indices + 5 ratios) |

Index files are memory-mapped for efficient access without loading into RAM.

## Configuration

### Solver Config

```rust
let config = SolverConfig {
    max_stars: 50,        // Use top N brightest stars
    max_quads: 100,       // Maximum patterns to generate
    max_matches: 50,      // Maximum matches to verify
    bin_tolerance: 0.02,  // Hash bin query tolerance
    ratio_tolerance: 0.01, // Pattern matching tolerance
    timeout_ms: 30000,    // Solver timeout
    ..Default::default()
};
```

Presets:
- `SolverConfig::default()` - Balanced accuracy/speed
- `SolverConfig::fast()` - Quick solving, lower accuracy
- `SolverConfig::thorough()` - Maximum accuracy, slower

### Build Config

```rust
let config = BuildConfig {
    fov_min_deg: 10.0,    // Minimum pattern size
    fov_max_deg: 30.0,    // Maximum pattern size
    mag_limit: 7.0,       // Magnitude cutoff
    num_bins: 1_000_000,  // Hash table size
    max_stars: 5000,      // Stars to use for patterns
    ..Default::default()
};
```

## Benchmarking

Run benchmarks against synthetic data:

```rust
use chameleon::bench::{BenchmarkSuite, SyntheticConfig};

let mut suite = BenchmarkSuite::new();
suite.run_synthetic(
    &index,
    100,  // num fields
    20.0, // FOV degrees
    1024, 768, // image size
    &SyntheticConfig::default(),
    SolverConfig::default(),
);
suite.print_report();
```

Compare with astrometry.net:

```rust
use chameleon::bench::comparison::{run_astrometry_net, compare_solutions};

// Run astrometry.net
let wcs_path = run_astrometry_net("image.fits", 5.0, 30.0, 60)?;
let astrometry_wcs = parse_astrometry_wcs(&wcs_path)?;

// Compare solutions
let comparison = compare_solutions(&chameleon_solution, &astrometry_wcs);
println!("Position difference: {:.2}\"", comparison.position_diff_arcsec.unwrap());
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Solve rate | >95% (clean fields) |
| Position accuracy | <10 arcsec |
| Solve time | <100ms (after index load) |
| Runtime memory | ~16 KB heap |
| Index size | ~10 MB (10-30° FOV, mag 7) |

## Catalog Support

Currently supported:
- Hipparcos (~118k stars) - via `build-index` tool
- CSV format (RA,Dec,Mag,ID)
- Synthetic catalogs (for testing)

Planned:
- Gaia DR3 subset
- Tycho-2

## License

MIT

## Acknowledgments

- [astrometry.net](https://astrometry.net) - Original plate solving algorithms
- [tetra3](https://github.com/esa/tetra3) - TETRA algorithm implementation
- [FITS WCS standard](https://fits.gsfc.nasa.gov/fits_wcs.html)
