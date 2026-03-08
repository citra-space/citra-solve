//! Star pattern extraction and matching.

pub mod hash;
pub mod matcher;
pub mod quad;

pub use hash::{compute_hash, query_hash_bins};
pub use matcher::{PatternMatch, PatternMatcher};
pub use quad::{generate_quads, Quad};
