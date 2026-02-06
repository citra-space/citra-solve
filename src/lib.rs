//! # Chameleon
//!
//! Efficient lost-in-space astrometric plate solver for embedded systems.
//!
//! Chameleon uses geometric hashing of 4-star patterns (quads) to achieve
//! fast, memory-efficient plate solving suitable for wide field-of-view
//! systems with significant optical distortion.
//!
//! ## Features
//!
//! - **Low memory footprint**: ~16KB runtime heap, memory-mapped indices
//! - **Wide FOV support**: 10+ degree fields with SIP distortion modeling
//! - **Robust matching**: Handles false stars (noise) and missing stars
//! - **Fast solving**: O(1) hash table lookups, early termination
//!
//! ## Quick Start
//!
//! ```no_run
//! use chameleon::{Solver, SolverConfig, DetectedStar, Index};
//!
//! // Load a pre-built index
//! let index = Index::open("hipparcos.idx").unwrap();
//!
//! // Create solver with default config
//! let solver = Solver::new(&index, SolverConfig::default());
//!
//! // Your detected stars from image
//! let stars = vec![
//!     DetectedStar { x: 512.0, y: 384.0, flux: 1000.0 },
//!     // ... more stars
//! ];
//!
//! // Solve!
//! match solver.solve(&stars, 1024, 768) {
//!     Ok(solution) => println!("Solved: RA={}, Dec={}", solution.center.ra_deg(), solution.center.dec_deg()),
//!     Err(e) => eprintln!("Failed to solve: {}", e),
//! }
//! ```

pub mod core;
pub mod catalog;
pub mod pattern;
pub mod solver;
pub mod wcs;
pub mod bench;

#[cfg(feature = "extract")]
pub mod extract;

// Re-export main types for convenience
pub use crate::core::types::{RaDec, Vec3, DetectedStar};
pub use crate::catalog::index::Index;
pub use crate::solver::solver::{Solver, SolverConfig};
pub use crate::solver::solution::Solution;
pub use crate::wcs::Wcs;

#[cfg(feature = "extract")]
pub use crate::extract::{extract_stars, ExtractionConfig};

/// Error types for the library
pub mod error {
    pub use crate::catalog::error::CatalogError;
    pub use crate::solver::error::SolveError;
}
