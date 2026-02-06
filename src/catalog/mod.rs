//! Star catalog and index management.
//!
//! This module handles loading and querying pre-built star pattern indices.
//! Indices are memory-mapped for efficient access on embedded systems.

pub mod error;
pub mod star;
pub mod index;
pub mod reader;
pub mod builder;

pub use error::CatalogError;
pub use index::Index;
pub use star::PackedStar;
