//! Main solver orchestration.

use std::sync::OnceLock;
use std::time::Instant;

use super::error::SolveError;
use super::hypothesis::{generate_hypotheses, Hypothesis};
use super::refine::{refine_linear_wcs, refine_solution, RefineConfig};
use super::solution::Solution;
use super::verify::{
    verify_hypothesis_with_index, CatalogStarIndex, DetectedStarIndex, VerifyConfig, VerifyResult,
};
use crate::catalog::Index;
use crate::core::types::DetectedStar;
use crate::pattern::{generate_quads, PatternMatcher};

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
    /// Hard cap on hash bins queried per quad.
    pub max_hash_bins: usize,
    /// Hard cap on patterns scanned in a single hash bin.
    pub max_patterns_per_bin: usize,
    /// Enable two-stage solving (fast pass followed by full pass on low confidence).
    pub enable_staged_solve: bool,
    /// Fast-stage acceptance threshold for log-odds.
    pub stage_accept_log_odds: f64,
    /// Fast-stage acceptance threshold for matched stars.
    pub stage_accept_matches: usize,
    /// Optional maximum mapped index size (bytes).
    pub max_index_bytes: Option<u64>,
    /// Optional maximum number of patterns allowed in loaded index.
    pub max_index_patterns: Option<u32>,
    /// Enable declination-band catalog indexing during verification.
    pub use_catalog_spatial_index: bool,
    /// Declination-band size for catalog spatial index.
    pub catalog_index_bin_deg: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_stars: 60,
            max_quads: 2200,
            max_matches: 220,
            bin_tolerance: 0.02,
            ratio_tolerance: 0.02,
            verify_config: VerifyConfig::default(),
            refine_config: RefineConfig::default(),
            timeout_ms: 30000, // 30 seconds
            min_stars: 4,
            max_hash_bins: 20_000,
            max_patterns_per_bin: 4_096,
            enable_staged_solve: true,
            stage_accept_log_odds: 38.0,
            stage_accept_matches: 14,
            max_index_bytes: Some(150 * 1024 * 1024),
            max_index_patterns: Some(8_000_000),
            use_catalog_spatial_index: true,
            catalog_index_bin_deg: 5.0,
        }
    }
}

impl SolverConfig {
    /// Create a "fast" configuration that trades accuracy for speed.
    pub fn fast() -> Self {
        let mut config = Self {
            max_stars: 28,
            max_quads: 700,
            max_matches: 110,
            bin_tolerance: 0.02,
            ratio_tolerance: 0.02,
            verify_config: VerifyConfig {
                log_odds_threshold: 18.0,
                ..Default::default()
            },
            timeout_ms: 5000,
            ..Default::default()
        };
        config.enable_staged_solve = false;
        config
    }

    /// Create a memory-constrained profile for embedded devices.
    pub fn constrained() -> Self {
        Self {
            max_stars: 60,
            max_quads: 2200,
            max_matches: 220,
            bin_tolerance: 0.02,
            ratio_tolerance: 0.02,
            max_hash_bins: 12_000,
            max_patterns_per_bin: 2_048,
            max_index_bytes: Some(200 * 1024 * 1024),
            max_index_patterns: Some(10_000_000),
            ..Default::default()
        }
    }

    /// Create a "thorough" configuration for difficult fields.
    pub fn thorough() -> Self {
        Self {
            max_stars: 100,
            max_quads: 4000,
            max_matches: 500,
            bin_tolerance: 0.03,
            ratio_tolerance: 0.015,
            verify_config: VerifyConfig {
                log_odds_threshold: 25.0,
                ..Default::default()
            },
            timeout_ms: 120000, // 2 minutes
            max_hash_bins: 60_000,
            max_patterns_per_bin: 32_768,
            max_index_bytes: None,
            max_index_patterns: None,
            ..Default::default()
        }
    }
}

/// The main plate solver.
pub struct Solver<'a> {
    index: &'a Index,
    config: SolverConfig,
    catalog_star_index: OnceLock<CatalogStarIndex>,
}

impl<'a> Solver<'a> {
    /// Create a new solver with the given index and configuration.
    pub fn new(index: &'a Index, config: SolverConfig) -> Self {
        Self {
            index,
            config,
            catalog_star_index: OnceLock::new(),
        }
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
        if stars.len() < self.config.min_stars {
            return Err(SolveError::NotEnoughStars(
                self.config.min_stars,
                stars.len(),
            ));
        }
        self.enforce_resource_limits()?;

        if self.should_run_fast_stage() {
            let stage_config = self.fast_stage_config();
            let stage_solver = Solver::new(self.index, stage_config);
            if let Ok(stage_solution) =
                stage_solver.solve_single_window(stars, image_width, image_height)
            {
                if self.fast_stage_accepted(&stage_solution) {
                    return Ok(stage_solution);
                }
            }
        }

        self.solve_single_window(stars, image_width, image_height)
    }

    fn enforce_resource_limits(&self) -> Result<(), SolveError> {
        if let Some(max_bytes) = self.config.max_index_bytes {
            let mapped = self.index.mapped_len_bytes() as u64;
            if mapped > max_bytes {
                return Err(SolveError::ResourceLimitExceeded(format!(
                    "index size {} bytes exceeds configured limit {} bytes",
                    mapped, max_bytes
                )));
            }
        }

        if let Some(max_patterns) = self.config.max_index_patterns {
            let patterns = self.index.num_patterns();
            if patterns > max_patterns {
                return Err(SolveError::ResourceLimitExceeded(format!(
                    "index patterns {} exceed configured limit {}",
                    patterns, max_patterns
                )));
            }
        }

        Ok(())
    }

    fn should_run_fast_stage(&self) -> bool {
        self.config.enable_staged_solve
            && (self.config.max_stars > 40
                || self.config.max_quads > 900
                || self.config.max_matches > 110
                || self.config.bin_tolerance > 0.02
                || self.config.ratio_tolerance > 0.02)
    }

    fn fast_stage_config(&self) -> SolverConfig {
        let mut config = self.config.clone();
        config.enable_staged_solve = false;
        config.max_stars = config.max_stars.min(40).max(config.min_stars.max(8));
        config.max_quads = config.max_quads.min(900).max(300);
        config.max_matches = config.max_matches.min(110).max(48);
        config.bin_tolerance = config.bin_tolerance.min(0.02);
        config.ratio_tolerance = config.ratio_tolerance.min(0.02);
        config.max_hash_bins = config.max_hash_bins.min(8_000).max(2_000);
        config.max_patterns_per_bin = config.max_patterns_per_bin.min(2_048).max(256);
        if config.timeout_ms > 0 {
            config.timeout_ms = config.timeout_ms.min(1500);
        }
        config
    }

    fn fast_stage_accepted(&self, solution: &Solution) -> bool {
        solution.log_odds >= self.config.stage_accept_log_odds
            && solution.num_matched_stars >= self.config.stage_accept_matches
            && solution.rms_arcsec <= 400.0
    }

    fn solve_single_window(
        &self,
        stars: &[DetectedStar],
        image_width: u32,
        image_height: u32,
    ) -> Result<Solution, SolveError> {
        let start_time = Instant::now();

        // Check minimum stars
        if stars.len() < self.config.min_stars {
            return Err(SolveError::NotEnoughStars(
                self.config.min_stars,
                stars.len(),
            ));
        }

        // Sort stars by brightness (create sorted copy)
        let mut sorted_stars = stars.to_vec();
        sorted_stars.sort_by(|a, b| {
            b.flux
                .partial_cmp(&a.flux)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // Keep only the brightest stars for solving and verification.
        // This suppresses cloud/noise detections that dominate lower-SNR tails.
        if sorted_stars.len() > self.config.max_stars {
            sorted_stars.truncate(self.config.max_stars);
        }

        // Generate quads from detected stars
        let image_diagonal = ((image_width as f64).powi(2) + (image_height as f64).powi(2)).sqrt();
        let min_edge_pixels = (image_diagonal * 0.05).max(8.0);
        let max_edge_pixels = image_diagonal * 0.98;

        let mut quads = crate::pattern::quad::generate_quads_brightness_priority(
            &sorted_stars,
            self.config.max_stars,
            self.config.max_quads,
            min_edge_pixels,
            max_edge_pixels,
        );
        if quads.len() < (self.config.max_quads / 4).max(64) {
            quads = generate_quads(&sorted_stars, self.config.max_stars, self.config.max_quads);
        }

        if quads.is_empty() {
            return Err(SolveError::NotEnoughStars(4, sorted_stars.len()));
        }

        // Check timeout
        if self.config.timeout_ms > 0
            && start_time.elapsed().as_millis() > self.config.timeout_ms as u128
        {
            return Err(SolveError::Timeout(self.config.timeout_ms));
        }

        let catalog_index = if self.config.use_catalog_spatial_index {
            Some(self.catalog_star_index.get_or_init(|| {
                CatalogStarIndex::from_index(self.index, self.config.catalog_index_bin_deg)
            }))
        } else {
            None
        };

        // Match patterns against the index
        let matcher = PatternMatcher::new(self.index)
            .with_bin_tolerance(self.config.bin_tolerance)
            .with_ratio_tolerance(self.config.ratio_tolerance)
            .with_max_hash_bins(self.config.max_hash_bins)
            .with_max_patterns_per_bin(self.config.max_patterns_per_bin);

        let matches = matcher.find_matches_batch(&quads, 6);

        if matches.is_empty() {
            return Err(SolveError::NoMatches);
        }

        // Take top matches
        let top_matches: Vec<_> = matches.into_iter().take(self.config.max_matches).collect();

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
        let verify_index = DetectedStarIndex::new(
            &sorted_stars,
            image_width,
            image_height,
            self.config.verify_config.max_match_distance_pixels,
        );

        let phase1_limit = (self.config.max_matches * 4).clamp(80, 900);
        let mut best_phase1_odds = f64::NEG_INFINITY;
        let mut best_phase1_matches = 0usize;

        for hypothesis in hypotheses.into_iter().take(phase1_limit) {
            // Check timeout
            if self.config.timeout_ms > 0
                && start_time.elapsed().as_millis() > self.config.timeout_ms as u128
            {
                break;
            }

            let result = verify_hypothesis_with_index(
                &hypothesis,
                &sorted_stars,
                self.index,
                image_width,
                image_height,
                &self.config.verify_config,
                Some(&verify_index),
                catalog_index,
            );

            if result.num_matched >= 4 {
                best_phase1_odds = best_phase1_odds.max(result.hypothesis.log_odds);
                best_phase1_matches = best_phase1_matches.max(result.num_matched);
                candidates.push(result);
            }

            // Strong early-hit heuristic: if we already have a high-confidence
            // high-match candidate, stop spending time on tail hypotheses.
            if candidates.len() >= 50
                && best_phase1_odds >= self.config.verify_config.log_odds_threshold + 22.0
                && best_phase1_matches >= 20
            {
                break;
            }
        }

        if candidates.is_empty() {
            return Err(SolveError::NoMatches);
        }

        // Sort candidates by match count first, then confidence.
        candidates.sort_by(|a, b| {
            b.num_matched.cmp(&a.num_matched).then_with(|| {
                b.hypothesis
                    .log_odds
                    .partial_cmp(&a.hypothesis.log_odds)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });
        candidates.truncate(100);

        // Phase 2: Refine and tight-verify top candidates.
        //
        // For each candidate: iteratively refine the WCS with wide tolerance to
        // bootstrap more matches, then tight-verify to confirm. This ensures only
        // genuinely correct hypotheses pass the tight tolerance check.
        let tight_config = VerifyConfig {
            position_sigma_pixels: 2.5,
            max_match_distance_pixels: 8.0,
            ..self.config.verify_config.clone()
        };
        let (index_fov_min_deg, index_fov_max_deg) = self.index.fov_range_deg();
        let min_allowed_fov = index_fov_min_deg as f64 * 0.7;
        let max_allowed_fov = index_fov_max_deg as f64 * 1.08;
        let tight_index = DetectedStarIndex::new(
            &sorted_stars,
            image_width,
            image_height,
            tight_config.max_match_distance_pixels,
        );

        let mut best_tight: Option<VerifyResult> = None;
        let mut best_relaxed: Option<VerifyResult> = None;
        let strict_min_matches = 7usize;
        let strict_min_detected_frac = 0.07;
        let strict_min_expected_frac = 0.10;
        let strict_max_rms_px = 6.5;
        let relaxed_min_matches = 6usize;
        let relaxed_min_detected_frac = 0.05;
        let relaxed_min_expected_frac = 0.07;
        let relaxed_max_rms_px = 8.5;

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
                    let refined_result = verify_hypothesis_with_index(
                        &refined_hyp,
                        &sorted_stars,
                        self.index,
                        image_width,
                        image_height,
                        &self.config.verify_config,
                        Some(&verify_index),
                        catalog_index,
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
            let tight_result = verify_hypothesis_with_index(
                &tight_hyp,
                &sorted_stars,
                self.index,
                image_width,
                image_height,
                &tight_config,
                Some(&tight_index),
                catalog_index,
            );

            let match_fraction_detected =
                tight_result.num_matched as f64 / sorted_stars.len().max(1) as f64;
            let match_fraction_expected =
                tight_result.num_matched as f64 / tight_result.num_expected.max(1) as f64;

            // Reject hypotheses whose implied field-of-view is far outside the
            // index design range (common in false-positive pattern matches).
            let tight_scale = tight_result.hypothesis.wcs.pixel_scale_arcsec();
            let tight_fov_w = image_width as f64 * tight_scale / 3600.0;
            let tight_fov_h = image_height as f64 * tight_scale / 3600.0;
            if tight_fov_w < min_allowed_fov
                || tight_fov_h < min_allowed_fov * 0.6
                || tight_fov_w > max_allowed_fov
                || tight_fov_h > max_allowed_fov
            {
                continue;
            }

            let strict_ok = tight_result.num_matched >= strict_min_matches
                && match_fraction_detected >= strict_min_detected_frac
                && match_fraction_expected >= strict_min_expected_frac
                && tight_result.rms_residual_pixels <= strict_max_rms_px;
            let relaxed_ok = tight_result.num_matched >= relaxed_min_matches
                && match_fraction_detected >= relaxed_min_detected_frac
                && match_fraction_expected >= relaxed_min_expected_frac
                && tight_result.rms_residual_pixels <= relaxed_max_rms_px;

            if strict_ok && tight_result.hypothesis.log_odds >= tight_config.log_odds_threshold {
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
            } else if relaxed_ok
                && tight_result.hypothesis.log_odds >= tight_config.log_odds_threshold - 4.0
            {
                let dominated = best_relaxed.as_ref().map_or(true, |best| {
                    tight_result.num_matched > best.num_matched
                        || (tight_result.num_matched == best.num_matched
                            && tight_result.hypothesis.log_odds > best.hypothesis.log_odds)
                });
                if dominated {
                    best_relaxed = Some(tight_result);
                }
            }
        }

        let best_result = best_tight.or(best_relaxed).ok_or_else(|| {
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

        // Final sanity guard against low-quality false locks.
        if solution.rms_arcsec > 600.0 {
            return Err(SolveError::VerificationFailed(
                best_result.hypothesis.log_odds,
            ));
        }

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
        self.enforce_resource_limits()?;
        progress("Generating patterns", 0.1);

        // Sort stars
        let mut sorted_stars = stars.to_vec();
        sorted_stars.sort_by(|a, b| {
            b.flux
                .partial_cmp(&a.flux)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if sorted_stars.len() > self.config.max_stars {
            sorted_stars.truncate(self.config.max_stars);
        }

        if sorted_stars.len() < self.config.min_stars {
            return Err(SolveError::NotEnoughStars(
                self.config.min_stars,
                sorted_stars.len(),
            ));
        }

        let quads = generate_quads(&sorted_stars, self.config.max_stars, self.config.max_quads);

        progress("Matching patterns", 0.3);

        let catalog_index = if self.config.use_catalog_spatial_index {
            Some(self.catalog_star_index.get_or_init(|| {
                CatalogStarIndex::from_index(self.index, self.config.catalog_index_bin_deg)
            }))
        } else {
            None
        };

        let matcher = PatternMatcher::new(self.index)
            .with_bin_tolerance(self.config.bin_tolerance)
            .with_ratio_tolerance(self.config.ratio_tolerance)
            .with_max_hash_bins(self.config.max_hash_bins)
            .with_max_patterns_per_bin(self.config.max_patterns_per_bin);

        let matches = matcher.find_matches_batch(&quads, 6);

        if matches.is_empty() {
            return Err(SolveError::NoMatches);
        }

        progress("Generating hypotheses", 0.5);

        let hypotheses = generate_hypotheses(
            &sorted_stars,
            &quads,
            &matches
                .into_iter()
                .take(self.config.max_matches)
                .collect::<Vec<_>>(),
            self.index,
            image_width,
            image_height,
        );

        progress("Verifying solutions", 0.7);

        let verify_index = DetectedStarIndex::new(
            &sorted_stars,
            image_width,
            image_height,
            self.config.verify_config.max_match_distance_pixels,
        );

        let mut best_result: Option<VerifyResult> = None;
        for hypothesis in hypotheses {
            let result = verify_hypothesis_with_index(
                &hypothesis,
                &sorted_stars,
                self.index,
                image_width,
                image_height,
                &self.config.verify_config,
                Some(&verify_index),
                catalog_index,
            );

            if best_result.is_none()
                || result.hypothesis.log_odds > best_result.as_ref().unwrap().hypothesis.log_odds
            {
                best_result = Some(result);
            }
        }

        let best_result = best_result.ok_or(SolveError::NoMatches)?;

        if best_result.hypothesis.log_odds < self.config.verify_config.log_odds_threshold {
            return Err(SolveError::VerificationFailed(
                best_result.hypothesis.log_odds,
            ));
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
        assert!(config.max_index_bytes.is_some());
        assert!(config.max_index_patterns.is_some());
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
        assert!(thorough.max_index_bytes.is_none());
        assert!(thorough.max_index_patterns.is_none());
    }

    #[test]
    fn test_solver_config_constrained() {
        let constrained = SolverConfig::constrained();
        assert!(constrained.max_index_bytes.is_some());
        assert!(constrained.max_index_patterns.is_some());
        assert!(constrained.bin_tolerance <= 0.02);
        assert!(constrained.max_hash_bins <= 12_000);
    }
}
