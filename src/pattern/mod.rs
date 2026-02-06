//! Star pattern extraction and matching.

pub mod quad;
pub mod hash;
pub mod matcher;

pub use quad::{Quad, generate_quads};
pub use hash::{compute_hash, query_hash_bins};
pub use matcher::{PatternMatcher, PatternMatch};
