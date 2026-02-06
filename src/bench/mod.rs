//! Benchmarking utilities for plate solver evaluation.

pub mod synthetic;
pub mod harness;
pub mod comparison;

pub use synthetic::SyntheticField;
pub use harness::{BenchmarkResult, BenchmarkSuite};
