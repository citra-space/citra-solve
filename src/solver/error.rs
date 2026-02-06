//! Error types for plate solving.

use thiserror::Error;

/// Errors that can occur during plate solving.
#[derive(Error, Debug)]
pub enum SolveError {
    /// Not enough stars detected to attempt solving.
    #[error("not enough stars: need at least {0}, got {1}")]
    NotEnoughStars(usize, usize),

    /// No matching patterns found in the index.
    #[error("no pattern matches found")]
    NoMatches,

    /// No hypothesis passed verification.
    #[error("no solution verified: best log-odds was {0:.2}")]
    VerificationFailed(f64),

    /// Solution refinement failed to converge.
    #[error("refinement failed to converge")]
    RefinementFailed,

    /// Timeout exceeded.
    #[error("solver timeout after {0}ms")]
    Timeout(u32),

    /// Catalog/index error.
    #[error("catalog error: {0}")]
    Catalog(#[from] crate::catalog::CatalogError),
}
