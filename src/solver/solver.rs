//! Main solver orchestration.

use std::time::Instant;

use crate::core::types::DetectedStar;
use crate::catalog::Index;
use crate::pattern::{generate_quads, PatternMatcher};
use super::error::SolveError;
use super::hypothesis::generate_hypotheses;
use super::verify::{verify_hypothesis, VerifyConfig, VerifyResult};
use super::refine::{refine_solution, RefineConfig};
use super::solution::Solution;

/// Configuration for the solver.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Maximum number of stars to use for pattern generation.
    pub max_stars: usize,
    /// Maximum number of quads to generate.
    pub max_quads: usize,
    /// Maximum pattern matches to consider.
    pub max_matches: usize,
    /// Hash bin tolerance for pattern matching.
    pub bin_tolerance: f64,
    /// Ratio tolerance for pattern matching.
    pub ratio_tolerance: f64,
    /// Verification configuration.
    pub verify_config: VerifyConfig,
    /// Refinement configuration.
    pub refine_config: RefineConfig,
    /// Timeout in milliseconds (0 = no timeout).
    pub timeout_ms: u32,
    /// Minimum stars required to attempt solving.
    pub min_stars: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_stars: 50,
            max_quads: 100,
            max_matches: 50,
            bin_tolerance: 0.02,
            ratio_tolerance: 0.01,
            verify_config: VerifyConfig::default(),
            refine_config: RefineConfig::default(),
            timeout_ms: 30000, // 30 seconds
            min_stars: 4,
        }
    }
}

impl SolverConfig {
    /// Create a "fast" configuration that trades accuracy for speed.
    pub fn fast() -> Self {
        Self {
            max_stars: 20,
            max_quads: 30,
            max_matches: 20,
            verify_config: VerifyConfig {
                log_odds_threshold: 15.0,
                ..Default::default()
            },
            timeout_ms: 5000,
            ..Default::default()
        }
    }

    /// Create a "thorough" configuration for difficult fields.
    pub fn thorough() -> Self {
        Self {
            max_stars: 100,
            max_quads: 500,
            max_matches: 200,
            bin_tolerance: 0.03,
            ratio_tolerance: 0.015,
            verify_config: VerifyConfig {
                log_odds_threshold: 25.0,
                ..Default::default()
            },
            timeout_ms: 120000, // 2 minutes
            ..Default::default()
        }
    }
}

/// The main plate solver.
pub struct Solver<'a> {
    index: &'a Index,
    config: SolverConfig,
}

impl<'a> Solver<'a> {
    /// Create a new solver with the given index and configuration.
    pub fn new(index: &'a Index, config: SolverConfig) -> Self {
        Self { index, config }
    }

    /// Solve an image given detected star positions.
    ///
    /// Stars should be sorted by brightness (brightest first) for best results.
    pub fn solve(
        &self,
        stars: &[DetectedStar],
        image_width: u32,
        image_height: u32,
    ) -> Result<Solution, SolveError> {
        let start_time = Instant::now();

        // Check minimum stars
        if stars.len() < self.config.min_stars {
            return Err(SolveError::NotEnoughStars(self.config.min_stars, stars.len()));
        }

        // Sort stars by brightness (create sorted copy)
        let mut sorted_stars = stars.to_vec();
        sorted_stars.sort_by(|a, b| {
            b.flux
                .partial_cmp(&a.flux)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Generate quads from detected stars
        let quads = generate_quads(
            &sorted_stars,
            self.config.max_stars,
            self.config.max_quads,
        );

        if quads.is_empty() {
            return Err(SolveError::NotEnoughStars(4, sorted_stars.len()));
        }

        // Check timeout
        if self.config.timeout_ms > 0
            && start_time.elapsed().as_millis() > self.config.timeout_ms as u128
        {
            return Err(SolveError::Timeout(self.config.timeout_ms));
        }

        // Match patterns against the index
        let matcher = PatternMatcher::new(self.index)
            .with_bin_tolerance(self.config.bin_tolerance)
            .with_ratio_tolerance(self.config.ratio_tolerance);

        let matches = matcher.find_matches_batch(&quads, 10);

        if matches.is_empty() {
            return Err(SolveError::NoMatches);
        }

        // Take top matches
        let top_matches: Vec<_> = matches
            .into_iter()
            .take(self.config.max_matches)
            .collect();

        // Generate hypotheses
        let hypotheses = generate_hypotheses(
            &sorted_stars,
            &quads,
            &top_matches,
            self.index,
            image_width,
            image_height,
        );

        if hypotheses.is_empty() {
            return Err(SolveError::NoMatches);
        }

        // Verify hypotheses and find best
        let mut best_result: Option<VerifyResult> = None;

        for hypothesis in hypotheses {
            // Check timeout
            if self.config.timeout_ms > 0
                && start_time.elapsed().as_millis() > self.config.timeout_ms as u128
            {
                break;
            }

            let result = verify_hypothesis(
                &hypothesis,
                &sorted_stars,
                self.index,
                image_width,
                image_height,
                &self.config.verify_config,
            );

            // Early termination if we find a very confident match
            if result.hypothesis.log_odds > self.config.verify_config.log_odds_threshold * 1.5 {
                best_result = Some(result);
                break;
            }

            // Track best result
            if best_result.is_none()
                || result.hypothesis.log_odds > best_result.as_ref().unwrap().hypothesis.log_odds
            {
                best_result = Some(result);
            }
        }

        let best_result = best_result.ok_or(SolveError::NoMatches)?;

        // Check if best result passes threshold
        if best_result.hypothesis.log_odds < self.config.verify_config.log_odds_threshold {
            return Err(SolveError::VerificationFailed(best_result.hypothesis.log_odds));
        }

        // Refine the solution
        let refine_result = refine_solution(
            &sorted_stars,
            &best_result.hypothesis.star_matches,
            &best_result.hypothesis.wcs,
            &self.config.refine_config,
        );

        // Build final solution
        let solution = Solution::new(
            refine_result.wcs,
            image_width,
            image_height,
            refine_result.rms_arcsec,
            best_result.hypothesis.log_odds,
            best_result.hypothesis.star_matches,
        );

        Ok(solution)
    }

    /// Solve with optional progress callback.
    pub fn solve_with_progress<F>(
        &self,
        stars: &[DetectedStar],
        image_width: u32,
        image_height: u32,
        mut progress: F,
    ) -> Result<Solution, SolveError>
    where
        F: FnMut(&str, f32),
    {
        progress("Generating patterns", 0.1);

        // Sort stars
        let mut sorted_stars = stars.to_vec();
        sorted_stars.sort_by(|a, b| {
            b.flux
                .partial_cmp(&a.flux)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if sorted_stars.len() < self.config.min_stars {
            return Err(SolveError::NotEnoughStars(self.config.min_stars, sorted_stars.len()));
        }

        let quads = generate_quads(
            &sorted_stars,
            self.config.max_stars,
            self.config.max_quads,
        );

        progress("Matching patterns", 0.3);

        let matcher = PatternMatcher::new(self.index)
            .with_bin_tolerance(self.config.bin_tolerance)
            .with_ratio_tolerance(self.config.ratio_tolerance);

        let matches = matcher.find_matches_batch(&quads, 10);

        if matches.is_empty() {
            return Err(SolveError::NoMatches);
        }

        progress("Generating hypotheses", 0.5);

        let hypotheses = generate_hypotheses(
            &sorted_stars,
            &quads,
            &matches.into_iter().take(self.config.max_matches).collect::<Vec<_>>(),
            self.index,
            image_width,
            image_height,
        );

        progress("Verifying solutions", 0.7);

        let mut best_result: Option<VerifyResult> = None;
        for hypothesis in hypotheses {
            let result = verify_hypothesis(
                &hypothesis,
                &sorted_stars,
                self.index,
                image_width,
                image_height,
                &self.config.verify_config,
            );

            if best_result.is_none()
                || result.hypothesis.log_odds > best_result.as_ref().unwrap().hypothesis.log_odds
            {
                best_result = Some(result);
            }
        }

        let best_result = best_result.ok_or(SolveError::NoMatches)?;

        if best_result.hypothesis.log_odds < self.config.verify_config.log_odds_threshold {
            return Err(SolveError::VerificationFailed(best_result.hypothesis.log_odds));
        }

        progress("Refining solution", 0.9);

        let refine_result = refine_solution(
            &sorted_stars,
            &best_result.hypothesis.star_matches,
            &best_result.hypothesis.wcs,
            &self.config.refine_config,
        );

        progress("Complete", 1.0);

        Ok(Solution::new(
            refine_result.wcs,
            image_width,
            image_height,
            refine_result.rms_arcsec,
            best_result.hypothesis.log_odds,
            best_result.hypothesis.star_matches,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_config_default() {
        let config = SolverConfig::default();
        assert!(config.max_stars > 0);
        assert!(config.timeout_ms > 0);
    }

    #[test]
    fn test_solver_config_fast() {
        let fast = SolverConfig::fast();
        let default = SolverConfig::default();
        assert!(fast.max_stars < default.max_stars);
        assert!(fast.timeout_ms < default.timeout_ms);
    }

    #[test]
    fn test_solver_config_thorough() {
        let thorough = SolverConfig::thorough();
        let default = SolverConfig::default();
        assert!(thorough.max_stars > default.max_stars);
        assert!(thorough.timeout_ms > default.timeout_ms);
    }
}
