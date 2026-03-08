//! Benchmark harness for systematic testing.

use std::time::{Duration, Instant};

use super::synthetic::{generate_test_suite, SyntheticConfig, SyntheticField};
use crate::catalog::Index;
use crate::core::math::angular_separation_arcsec;
use crate::core::types::RaDec;
use crate::solver::{Solution, SolveError, Solver, SolverConfig};

/// Result of a single benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Whether the solve succeeded.
    pub solved: bool,
    /// Time taken to solve (or fail).
    pub solve_time: Duration,
    /// Error in RA (arcseconds), if solved.
    pub ra_error_arcsec: Option<f64>,
    /// Error in Dec (arcseconds), if solved.
    pub dec_error_arcsec: Option<f64>,
    /// Total position error (arcseconds), if solved.
    pub position_error_arcsec: Option<f64>,
    /// Rotation error (arcseconds), if solved.
    pub rotation_error_arcsec: Option<f64>,
    /// Scale error (parts per million), if solved.
    pub scale_error_ppm: Option<f64>,
    /// Number of matched stars.
    pub num_matched: usize,
    /// RMS residual (arcseconds).
    pub rms_arcsec: Option<f64>,
    /// Log-odds confidence.
    pub log_odds: Option<f64>,
    /// Error message if failed.
    pub error: Option<String>,
}

impl BenchmarkResult {
    fn success(solution: &Solution, true_wcs: &crate::wcs::Wcs, solve_time: Duration) -> Self {
        let true_center = true_wcs.crval();
        let solved_center = solution.center;

        let position_error = angular_separation_arcsec(&true_center, &solved_center);

        // Approximate RA/Dec errors
        let ra_error =
            (solved_center.ra - true_center.ra).to_degrees() * 3600.0 * true_center.dec.cos();
        let dec_error = (solved_center.dec - true_center.dec).to_degrees() * 3600.0;

        // Rotation error
        let true_rotation = true_wcs.rotation_deg();
        let solved_rotation = solution.rotation_deg;
        let rotation_error = (solved_rotation - true_rotation).abs() * 3600.0; // deg to arcsec

        // Scale error
        let true_scale = true_wcs.pixel_scale_arcsec();
        let solved_scale = solution.pixel_scale_arcsec;
        let scale_error_ppm = ((solved_scale - true_scale) / true_scale).abs() * 1_000_000.0;

        Self {
            solved: true,
            solve_time,
            ra_error_arcsec: Some(ra_error.abs()),
            dec_error_arcsec: Some(dec_error.abs()),
            position_error_arcsec: Some(position_error),
            rotation_error_arcsec: Some(rotation_error),
            scale_error_ppm: Some(scale_error_ppm),
            num_matched: solution.num_matched_stars,
            rms_arcsec: Some(solution.rms_arcsec),
            log_odds: Some(solution.log_odds),
            error: None,
        }
    }

    fn failure(error: SolveError, solve_time: Duration) -> Self {
        Self {
            solved: false,
            solve_time,
            ra_error_arcsec: None,
            dec_error_arcsec: None,
            position_error_arcsec: None,
            rotation_error_arcsec: None,
            scale_error_ppm: None,
            num_matched: 0,
            rms_arcsec: None,
            log_odds: None,
            error: Some(error.to_string()),
        }
    }
}

/// Summary statistics for a benchmark suite.
#[derive(Debug, Clone)]
pub struct BenchmarkSummary {
    /// Total number of tests.
    pub total: usize,
    /// Number of successful solves.
    pub solved: usize,
    /// Solve rate (0.0 - 1.0).
    pub solve_rate: f64,
    /// Mean solve time for successful solves.
    pub mean_solve_time: Duration,
    /// Median solve time for successful solves.
    pub median_solve_time: Duration,
    /// Mean position error (arcsec) for successful solves.
    pub mean_position_error_arcsec: f64,
    /// Max position error (arcsec) for successful solves.
    pub max_position_error_arcsec: f64,
    /// Mean RMS residual (arcsec).
    pub mean_rms_arcsec: f64,
}

/// Benchmark suite runner.
pub struct BenchmarkSuite {
    results: Vec<BenchmarkResult>,
}

impl BenchmarkSuite {
    /// Create a new empty benchmark suite.
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Run benchmarks on synthetic fields.
    pub fn run_synthetic(
        &mut self,
        index: &Index,
        num_fields: usize,
        fov_deg: f64,
        width: u32,
        height: u32,
        synthetic_config: &SyntheticConfig,
        solver_config: SolverConfig,
    ) {
        println!("Generating {} synthetic fields...", num_fields);
        let fields =
            generate_test_suite(index, num_fields, fov_deg, width, height, synthetic_config);

        println!("Running benchmarks...");
        for (i, field) in fields.iter().enumerate() {
            print!("  Field {}/{}: ", i + 1, num_fields);

            let result = self.run_single(index, field, &solver_config);

            if result.solved {
                println!(
                    "SOLVED in {:?}, error={:.2}\", {} stars",
                    result.solve_time,
                    result.position_error_arcsec.unwrap_or(0.0),
                    result.num_matched
                );
            } else {
                println!("FAILED: {}", result.error.as_deref().unwrap_or("unknown"));
            }

            self.results.push(result);
        }
    }

    /// Run a single benchmark.
    fn run_single(
        &self,
        index: &Index,
        field: &SyntheticField,
        solver_config: &SolverConfig,
    ) -> BenchmarkResult {
        let solver = Solver::new(index, solver_config.clone());

        let start = Instant::now();
        let result = solver.solve(&field.detected_stars, field.width, field.height);
        let elapsed = start.elapsed();

        match result {
            Ok(solution) => BenchmarkResult::success(&solution, &field.true_wcs, elapsed),
            Err(e) => BenchmarkResult::failure(e, elapsed),
        }
    }

    /// Get all results.
    pub fn results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    /// Compute summary statistics.
    pub fn summary(&self) -> BenchmarkSummary {
        let total = self.results.len();
        let solved: Vec<_> = self.results.iter().filter(|r| r.solved).collect();
        let num_solved = solved.len();

        let solve_rate = if total > 0 {
            num_solved as f64 / total as f64
        } else {
            0.0
        };

        let mut solve_times: Vec<Duration> = solved.iter().map(|r| r.solve_time).collect();
        solve_times.sort();

        let mean_solve_time = if !solve_times.is_empty() {
            Duration::from_nanos(
                solve_times.iter().map(|d| d.as_nanos() as u64).sum::<u64>()
                    / solve_times.len() as u64,
            )
        } else {
            Duration::ZERO
        };

        let median_solve_time = if !solve_times.is_empty() {
            solve_times[solve_times.len() / 2]
        } else {
            Duration::ZERO
        };

        let position_errors: Vec<f64> = solved
            .iter()
            .filter_map(|r| r.position_error_arcsec)
            .collect();

        let mean_position_error = if !position_errors.is_empty() {
            position_errors.iter().sum::<f64>() / position_errors.len() as f64
        } else {
            0.0
        };

        let max_position_error = position_errors.iter().cloned().fold(0.0f64, f64::max);

        let rms_values: Vec<f64> = solved.iter().filter_map(|r| r.rms_arcsec).collect();
        let mean_rms = if !rms_values.is_empty() {
            rms_values.iter().sum::<f64>() / rms_values.len() as f64
        } else {
            0.0
        };

        BenchmarkSummary {
            total,
            solved: num_solved,
            solve_rate,
            mean_solve_time,
            median_solve_time,
            mean_position_error_arcsec: mean_position_error,
            max_position_error_arcsec: max_position_error,
            mean_rms_arcsec: mean_rms,
        }
    }

    /// Print summary report.
    pub fn print_report(&self) {
        let summary = self.summary();

        println!("\n========================================");
        println!("         BENCHMARK SUMMARY");
        println!("========================================");
        println!();
        println!("Total tests:       {}", summary.total);
        println!("Solved:            {}", summary.solved);
        println!("Solve rate:        {:.1}%", summary.solve_rate * 100.0);
        println!();
        println!("Timing:");
        println!("  Mean solve time:   {:?}", summary.mean_solve_time);
        println!("  Median solve time: {:?}", summary.median_solve_time);
        println!();
        println!("Accuracy:");
        println!(
            "  Mean position error:  {:.2}\"",
            summary.mean_position_error_arcsec
        );
        println!(
            "  Max position error:   {:.2}\"",
            summary.max_position_error_arcsec
        );
        println!("  Mean RMS residual:    {:.2}\"", summary.mean_rms_arcsec);
        println!("========================================\n");
    }
}

impl Default for BenchmarkSuite {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_suite_empty() {
        let suite = BenchmarkSuite::new();
        let summary = suite.summary();
        assert_eq!(summary.total, 0);
        assert_eq!(summary.solved, 0);
    }
}
