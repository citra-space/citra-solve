//! Star detection and centroid extraction from images.
//!
//! This module provides basic star detection using local maximum finding
//! and intensity-weighted centroid computation.

#[cfg(feature = "extract")]
mod centroid;

#[cfg(feature = "extract")]
pub use centroid::{extract_stars, ExtractionConfig};
