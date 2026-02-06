//! Plate solving core: hypothesis generation, verification, and refinement.

pub mod error;
pub mod hypothesis;
pub mod verify;
pub mod refine;
pub mod solution;
pub mod solver;

pub use error::SolveError;
pub use solution::Solution;
pub use solver::{Solver, SolverConfig};
pub use refine::{fit_sip, SipRefineResult, refine_linear_wcs, LinearRefineResult};
