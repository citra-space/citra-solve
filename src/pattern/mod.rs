//! Star pattern extraction and matching.

pub mod hash;
pub mod matcher;
pub mod quad;
pub mod tetra;

pub use hash::{compute_hash, query_hash_bins, query_hash_bins_ranked, HashProbe};
pub use matcher::{PatternMatch, PatternMatcher};
pub use quad::{generate_quads, Quad};
pub use tetra::canonical_tetra_signature;
