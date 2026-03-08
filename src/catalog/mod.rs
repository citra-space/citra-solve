//! Star catalog and index management.
//!
//! This module handles loading and querying pre-built star pattern indices.
//! Indices are memory-mapped for efficient access on embedded systems.

pub mod builder;
pub mod error;
pub mod index;
pub mod reader;
pub mod star;

pub use error::CatalogError;
pub use index::Index;
pub use star::PackedStar;
