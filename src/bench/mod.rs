//! Benchmarking utilities for plate solver evaluation.

pub mod comparison;
pub mod harness;
pub mod synthetic;

pub use harness::{BenchmarkResult, BenchmarkSuite};
pub use synthetic::SyntheticField;
