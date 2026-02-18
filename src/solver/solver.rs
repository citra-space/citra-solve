//! Main solver orchestration.

use std::time::Instant;

use crate::core::types::DetectedStar;
use crate::catalog::Index;
use crate::pattern::{generate_quads, PatternMatcher};
use super::error::SolveError;
use super::hypothesis::{generate_hypotheses, Hypothesis};
use super::verify::{verify_hypothesis, VerifyConfig, VerifyResult};
use super::refine::{refine_solution, refine_linear_wcs, RefineConfig};
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
            max_quads: 10000,
            max_matches: 1000,
            bin_tolerance: 0.05,
            ratio_tolerance: 0.05,
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
            max_quads: 50,
            max_matches: 30,
            verify_config: VerifyConfig {
                log_odds_threshold: 12.0,
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

        // Phase 1: Screen all hypotheses with raw wide verification (no refinement).
        //
        // Screening without refinement gives a fair ranking because correct
        // hypotheses naturally produce lower residuals than wrong ones.
        let mut candidates: Vec<VerifyResult> = Vec::new();

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

            if result.num_matched >= 4 {
                candidates.push(result);
            }
        }

        if candidates.is_empty() {
            return Err(SolveError::NoMatches);
        }

        // Sort candidates by log_odds (descending) for phase 2
        candidates.sort_by(|a, b| {
            b.hypothesis.log_odds
                .partial_cmp(&a.hypothesis.log_odds)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(50);

        // Phase 2: Refine and tight-verify top candidates.
        //
        // For each candidate: iteratively refine the WCS with wide tolerance to
        // bootstrap more matches, then tight-verify to confirm. This ensures only
        // genuinely correct hypotheses pass the tight tolerance check.
        let tight_config = VerifyConfig {
            position_sigma_pixels: 5.0,
            max_match_distance_pixels: 20.0,
            ..self.config.verify_config.clone()
        };

        let mut best_tight: Option<VerifyResult> = None;

        for candidate in &candidates {
            // Iterative refine-verify with wide tolerance to improve WCS
            let mut refined = candidate.clone();
            for _ in 0..3 {
                if let Some(linear_refined) = refine_linear_wcs(
                    &sorted_stars,
                    &refined.hypothesis.star_matches,
                    &refined.hypothesis.wcs,
                    image_width,
                    image_height,
                ) {
                    let refined_hyp = Hypothesis::new(
                        refined.hypothesis.star_matches.clone(),
                        linear_refined.wcs,
                        refined.hypothesis.pattern_distance,
                    );
                    let refined_result = verify_hypothesis(
                        &refined_hyp,
                        &sorted_stars,
                        self.index,
                        image_width,
                        image_height,
                        &self.config.verify_config,
                    );
                    if refined_result.num_matched > refined.num_matched
                        || refined_result.hypothesis.log_odds > refined.hypothesis.log_odds
                    {
                        refined = refined_result;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }

            // Final tight verification
            let wcs_for_tight = if let Some(linear_re) = refine_linear_wcs(
                &sorted_stars,
                &refined.hypothesis.star_matches,
                &refined.hypothesis.wcs,
                image_width,
                image_height,
            ) {
                linear_re.wcs
            } else {
                refined.hypothesis.wcs.clone()
            };

            let tight_hyp = Hypothesis::new(
                refined.hypothesis.star_matches.clone(),
                wcs_for_tight,
                refined.hypothesis.pattern_distance,
            );
            let tight_result = verify_hypothesis(
                &tight_hyp,
                &sorted_stars,
                self.index,
                image_width,
                image_height,
                &tight_config,
            );

            if tight_result.hypothesis.log_odds >= tight_config.log_odds_threshold {
                // Select by match count first (most robust discriminator),
                // then by log_odds as tiebreaker.
                let dominated = best_tight.as_ref().map_or(true, |best| {
                    tight_result.num_matched > best.num_matched
                        || (tight_result.num_matched == best.num_matched
                            && tight_result.hypothesis.log_odds > best.hypothesis.log_odds)
                });
                if dominated {
                    best_tight = Some(tight_result);
                }
            }
        }

        let best_result = best_tight.ok_or_else(|| {
            // Report the best candidate's tight log_odds for debugging
            let best_candidate_odds = candidates
                .first()
                .map(|c| c.hypothesis.log_odds)
                .unwrap_or(f64::NEG_INFINITY);
            SolveError::VerificationFailed(best_candidate_odds)
        })?;

        // Final refinement and solution building
        let final_wcs;
        let final_rms;
        let final_matches = best_result.hypothesis.star_matches.clone();

        if let Some(linear_refined) = refine_linear_wcs(
            &sorted_stars,
            &final_matches,
            &best_result.hypothesis.wcs,
            image_width,
            image_height,
        ) {
            final_wcs = linear_refined.wcs;
            final_rms = linear_refined.rms_after_pixels * final_wcs.pixel_scale_arcsec();
        } else {
            let refine_result = refine_solution(
                &sorted_stars,
                &final_matches,
                &best_result.hypothesis.wcs,
                &self.config.refine_config,
            );
            final_wcs = refine_result.wcs;
            final_rms = refine_result.rms_arcsec;
        }

        let solution = Solution::new(
            final_wcs,
            image_width,
            image_height,
            final_rms,
            best_result.hypothesis.log_odds,
            final_matches,
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
