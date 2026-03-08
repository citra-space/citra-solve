//! Plate solving core: hypothesis generation, verification, and refinement.

pub mod error;
pub mod hypothesis;
pub mod refine;
pub mod solution;
pub mod solver;
pub mod verify;

pub use error::SolveError;
pub use refine::{fit_sip, refine_linear_wcs, LinearRefineResult, SipRefineResult};
pub use solution::Solution;
pub use solver::{Solver, SolverConfig};
pub use verify::CatalogStarIndex;
