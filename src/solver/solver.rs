//! Main solver orchestration.

use std::collections::{BTreeMap, HashMap, HashSet};
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
use crate::core::math;
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
    /// Optional lower bound for expected field width (degrees).
    pub expected_fov_min_deg: Option<f64>,
    /// Optional upper bound for expected field width (degrees).
    pub expected_fov_max_deg: Option<f64>,
    /// Minimum score margin required over the runner-up candidate.
    pub min_solution_margin: f64,
    /// Maximum hypotheses to verify in phase 1.
    pub phase1_hypothesis_cap: usize,
    /// Maximum candidates to carry into phase 2.
    pub phase2_candidate_cap: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_stars: 60,
            max_quads: 1200,
            max_matches: 320,
            bin_tolerance: 0.02,
            ratio_tolerance: 0.02,
            verify_config: VerifyConfig::default(),
            refine_config: RefineConfig::default(),
            timeout_ms: 30000, // 30 seconds
            min_stars: 4,
            max_hash_bins: 8_000,
            max_patterns_per_bin: 1_024,
            enable_staged_solve: true,
            stage_accept_log_odds: 38.0,
            stage_accept_matches: 14,
            max_index_bytes: Some(150 * 1024 * 1024),
            max_index_patterns: Some(8_000_000),
            use_catalog_spatial_index: true,
            catalog_index_bin_deg: 5.0,
            expected_fov_min_deg: None,
            expected_fov_max_deg: None,
            min_solution_margin: 7.5,
            phase1_hypothesis_cap: 220,
            phase2_candidate_cap: 30,
        }
    }
}

impl SolverConfig {
    /// Create a "fast" configuration that trades accuracy for speed.
    pub fn fast() -> Self {
        let mut config = Self {
            max_stars: 28,
            max_quads: 500,
            max_matches: 140,
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
            max_quads: 1000,
            max_matches: 260,
            bin_tolerance: 0.02,
            ratio_tolerance: 0.02,
            max_hash_bins: 6_000,
            max_patterns_per_bin: 768,
            max_index_bytes: Some(200 * 1024 * 1024),
            max_index_patterns: Some(10_000_000),
            expected_fov_min_deg: None,
            expected_fov_max_deg: None,
            phase1_hypothesis_cap: 180,
            phase2_candidate_cap: 24,
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
            expected_fov_min_deg: None,
            expected_fov_max_deg: None,
            phase1_hypothesis_cap: 480,
            phase2_candidate_cap: 64,
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

#[derive(Debug, Clone)]
struct RankedHypothesis {
    hypothesis: Hypothesis,
    votes: u16,
    quad_support: u16,
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
        config.phase1_hypothesis_cap = config.phase1_hypothesis_cap.min(140).max(80);
        config.phase2_candidate_cap = config.phase2_candidate_cap.min(20).max(12);
        config.min_solution_margin = config.min_solution_margin.min(6.0);
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

    fn verification_score(result: &VerifyResult) -> f64 {
        let matched = result.num_matched as f64;
        let coverage_geom = (result.expected_coverage * result.detected_coverage).sqrt();
        let residual_penalty =
            1.3 * result.rms_residual_pixels + 1.8 * result.median_residual_pixels;
        let low_match_penalty = if matched < 6.0 {
            (6.0 - matched) * 8.0
        } else {
            0.0
        };

        matched * 8.5
            + 26.0 * coverage_geom
            + 12.0 * result.brightness_consistency
            + 0.75 * result.hypothesis.log_odds
            - residual_penalty
            - low_match_penalty
    }

    fn nearly_duplicate(a: &VerifyResult, b: &VerifyResult) -> bool {
        let center_sep_deg =
            math::angular_separation(&a.hypothesis.wcs.crval(), &b.hypothesis.wcs.crval())
                .to_degrees();
        let scale_a = a.hypothesis.wcs.pixel_scale_arcsec().max(1e-6);
        let scale_b = b.hypothesis.wcs.pixel_scale_arcsec().max(1e-6);
        let scale_delta = (scale_a / scale_b).ln().abs();
        center_sep_deg < 0.40 && scale_delta < 0.050
    }

    fn hypothesis_consensus_key(hyp: &Hypothesis) -> (i16, i16, i16, i16) {
        let center = hyp.wcs.crval();
        let dec_deg = center.dec.to_degrees();
        let cos_dec = center.dec.cos().abs().max(0.15);
        let mut ra_deg = center.ra.to_degrees();
        if ra_deg < 0.0 {
            ra_deg += 360.0;
        }
        let ra_eq = ra_deg * cos_dec;
        let scale = hyp.wcs.pixel_scale_arcsec().max(1e-6);
        let rot = hyp.wcs.rotation_deg();

        // Coarse bins intentionally mirror astrometry/tetra3 style hypothesis
        // voting: broad geometric consensus before expensive verification.
        let ra_bin = (ra_eq / 0.65).round() as i16;
        let dec_bin = (dec_deg / 0.65).round() as i16;
        let scale_bin = ((scale.ln()) / 0.05).round() as i16;
        let rot_bin = (rot / 11.0).round() as i16;
        (ra_bin, dec_bin, scale_bin, rot_bin)
    }

    fn rank_hypotheses_for_verification(hypotheses: Vec<Hypothesis>) -> Vec<RankedHypothesis> {
        if hypotheses.is_empty() {
            return Vec::new();
        }

        let mut buckets: BTreeMap<(i16, i16, i16, i16), Vec<Hypothesis>> = BTreeMap::new();
        for hyp in hypotheses {
            buckets
                .entry(Self::hypothesis_consensus_key(&hyp))
                .or_default()
                .push(hyp);
        }

        let mut ranked: Vec<RankedHypothesis> = Vec::new();
        for mut group in buckets.into_values() {
            let votes = group.len().min(u16::MAX as usize) as u16;
            let quad_support = {
                let mut uniq: HashSet<usize> = HashSet::new();
                for h in &group {
                    uniq.insert(h.source_quad_idx);
                }
                uniq.len().min(u16::MAX as usize) as u16
            };
            group.sort_by(|a, b| {
                a.pattern_distance
                    .partial_cmp(&b.pattern_distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        let ac = a.wcs.crval();
                        let bc = b.wcs.crval();
                        ac.ra
                            .partial_cmp(&bc.ra)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| {
                                ac.dec
                                    .partial_cmp(&bc.dec)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                    })
            });
            for hyp in group.into_iter().take(6) {
                ranked.push(RankedHypothesis {
                    hypothesis: hyp,
                    votes,
                    quad_support,
                });
            }
        }

        ranked.sort_by(|a, b| {
            let quad_cmp = b.quad_support.cmp(&a.quad_support);
            if quad_cmp != std::cmp::Ordering::Equal {
                return quad_cmp;
            }
            let vote_cmp = b.votes.cmp(&a.votes);
            if vote_cmp != std::cmp::Ordering::Equal {
                return vote_cmp;
            }
            a.hypothesis
                .pattern_distance
                .partial_cmp(&b.hypothesis.pattern_distance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    let ac = a.hypothesis.wcs.crval();
                    let bc = b.hypothesis.wcs.crval();
                    ac.ra
                        .partial_cmp(&bc.ra)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| {
                            ac.dec
                                .partial_cmp(&bc.dec)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                })
        });
        ranked
    }

    fn solve_single_window(
        &self,
        stars: &[DetectedStar],
        image_width: u32,
        image_height: u32,
    ) -> Result<Solution, SolveError> {
        let start_time = Instant::now();
        let debug = std::env::var_os("CITRA_DEBUG").is_some();
        let debug_truth = if debug {
            std::env::var("CITRA_DEBUG_TRUTH")
                .ok()
                .and_then(|v| parse_debug_truth(&v))
        } else {
            None
        };

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
        let min_edge_pixels = (image_diagonal * 0.025).max(6.0);
        // Keep quads mostly local/intermediate scale for distortion resilience.
        // Very large quads are disproportionately affected by lens/TAN mismatch
        // and tend to produce false hypotheses on real wide-field frames.
        let max_edge_pixels = image_diagonal
            * if sorted_stars.len() >= 36 {
                0.48
            } else if sorted_stars.len() >= 24 {
                0.56
            } else {
                0.64
            };

        // Tetra3-style high-confidence seed set: exhaustively enumerate quads
        // from a small bright core so at least some pure-star patterns are tried.
        let mut quads: Vec<crate::pattern::Quad> = Vec::with_capacity(self.config.max_quads);
        let core_stars = sorted_stars.len().min(30).max(8);
        let core_quads = generate_quads(&sorted_stars[..core_stars], core_stars, 20_000);
        let core_quota = (self.config.max_quads * 3 / 5).max(220);
        let mut seen: HashSet<[usize; 4]> = HashSet::new();
        if !core_quads.is_empty() {
            let mut core_candidates: Vec<crate::pattern::Quad> = core_quads
                .into_iter()
                .filter(|q| {
                    q.max_edge_pixels >= min_edge_pixels && q.max_edge_pixels <= max_edge_pixels
                })
                .collect();
            if core_candidates.len() > core_quota {
                // Deterministic pseudo-random sampling across combination space.
                // This avoids lexicographic bias that can miss true quads.
                core_candidates.sort_by_key(|q| {
                    let [a, b, c, d] = q.star_indices;
                    let mut x = (a as u64).wrapping_mul(0x9E37_79B1_85EB_CA87);
                    x ^= (b as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
                    x ^= (c as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
                    x ^= (d as u64).wrapping_mul(0x27D4_EB2F_1656_67C5);
                    x
                });
                core_candidates.truncate(core_quota);
            }
            for q in core_candidates {
                if quads.len() >= core_quota {
                    break;
                }
                if seen.insert(q.star_indices) {
                    quads.push(q);
                }
            }
        }

        let bright_quota = (self.config.max_quads / 3).max(140);
        let bright_quads = crate::pattern::quad::generate_quads_brightness_priority(
            &sorted_stars,
            self.config.max_stars,
            bright_quota.min(self.config.max_quads),
            min_edge_pixels,
            max_edge_pixels,
        );
        for q in bright_quads {
            if quads.len() >= self.config.max_quads {
                break;
            }
            if seen.insert(q.star_indices) {
                quads.push(q);
            }
        }

        // Blend in uniformly sampled quads for missing-star resilience.
        let uniform_quads = generate_quads(
            &sorted_stars,
            self.config.max_stars,
            self.config.max_quads.saturating_mul(2),
        );
        if quads.len() < self.config.max_quads {
            for q in uniform_quads {
                if quads.len() >= self.config.max_quads {
                    break;
                }
                if q.max_edge_pixels < min_edge_pixels || q.max_edge_pixels > max_edge_pixels {
                    continue;
                }
                if seen.insert(q.star_indices) {
                    quads.push(q);
                }
            }
        }

        // If edge gating was too strict (very sparse fields), relax to ensure
        // we still have enough candidate quads for a solve attempt.
        if quads.len() < self.config.max_quads / 6 {
            let relaxed_max_edge = image_diagonal * 0.92;
            let relaxed = generate_quads(
                &sorted_stars,
                self.config.max_stars,
                self.config.max_quads.saturating_mul(2),
            );
            for q in relaxed {
                if quads.len() >= self.config.max_quads {
                    break;
                }
                if q.max_edge_pixels < min_edge_pixels || q.max_edge_pixels > relaxed_max_edge {
                    continue;
                }
                if seen.insert(q.star_indices) {
                    quads.push(q);
                }
            }
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

        // Match patterns against the index using scale-aware probing.
        let (index_fov_min_deg, index_fov_max_deg) = self.index.fov_range_deg();
        let mut fov_hint_min = self
            .config
            .expected_fov_min_deg
            .unwrap_or(index_fov_min_deg as f64);
        let mut fov_hint_max = self
            .config
            .expected_fov_max_deg
            .unwrap_or(index_fov_max_deg as f64);
        fov_hint_min = fov_hint_min.max(index_fov_min_deg as f64 * 0.8);
        fov_hint_max = fov_hint_max.min(index_fov_max_deg as f64 * 1.2);
        if fov_hint_max <= fov_hint_min {
            fov_hint_min = index_fov_min_deg as f64;
            fov_hint_max = index_fov_max_deg as f64;
        }

        let matcher = PatternMatcher::new(self.index)
            .with_bin_tolerance(self.config.bin_tolerance)
            .with_ratio_tolerance(self.config.ratio_tolerance)
            // Slightly wider tetra tolerance improves resilience to
            // centroid noise while ratio checks keep false matches bounded.
            .with_tetra_tolerance((self.config.ratio_tolerance * 1.8).clamp(0.020, 0.055))
            .with_max_hash_bins(self.config.max_hash_bins)
            .with_max_patterns_per_bin(self.config.max_patterns_per_bin)
            .with_scale_hint(image_diagonal, fov_hint_min, fov_hint_max)
            .with_max_scale_band_delta(4);

        let matches_per_quad = if quads.len() > 1_000 {
            3
        } else if quads.len() > 700 {
            4
        } else {
            6
        };
        let matches = matcher.find_matches_batch(&quads, matches_per_quad);
        if debug {
            eprintln!(
                "citra-debug: stars={} quads={} matches={}",
                sorted_stars.len(),
                quads.len(),
                matches.len()
            );
        }
        if matches.is_empty() {
            return Err(SolveError::NoMatches);
        }

        let mut per_quad_counts: HashMap<usize, u8> = HashMap::new();
        let mut top_matches = Vec::with_capacity(self.config.max_matches);
        for m in matches {
            let count = per_quad_counts.entry(m.detected_quad_idx).or_insert(0);
            if *count >= 2 {
                continue;
            }
            *count += 1;
            top_matches.push(m);
            if top_matches.len() >= self.config.max_matches {
                break;
            }
        }
        let hypotheses = generate_hypotheses(
            &sorted_stars,
            &quads,
            &top_matches,
            self.index,
            image_width,
            image_height,
        );
        let raw_hypothesis_count = hypotheses.len();
        let ranked_hypotheses = Self::rank_hypotheses_for_verification(hypotheses);
        if debug {
            let mut best_seed_rms = f64::MAX;
            let mut sample = 0usize;
            for ranked in ranked_hypotheses.iter().take(200) {
                let hyp = &ranked.hypothesis;
                if hyp.star_matches.is_empty() {
                    continue;
                }
                let mut sum_sq = 0.0;
                for (det_idx, cat_star) in &hyp.star_matches {
                    let det = &sorted_stars[*det_idx];
                    let (px, py) = hyp.wcs.sky_to_pixel(&cat_star.position);
                    let dx = det.x - px;
                    let dy = det.y - py;
                    sum_sq += dx * dx + dy * dy;
                }
                let rms = (sum_sq / hyp.star_matches.len() as f64).sqrt();
                if rms < best_seed_rms {
                    best_seed_rms = rms;
                }
                sample += 1;
            }
            eprintln!(
                "citra-debug: top_matches={} hypotheses_raw={} hypotheses_ranked={} best_seed_rms_px={:.2} sampled={}",
                top_matches.len(),
                raw_hypothesis_count,
                ranked_hypotheses.len(),
                best_seed_rms,
                sample
            );
            for (i, ranked) in ranked_hypotheses.iter().take(8).enumerate() {
                eprintln!(
                    "citra-debug: hypothesis_votes idx={} votes={} quad_support={}",
                    i, ranked.votes, ranked.quad_support
                );
            }
        }
        if ranked_hypotheses.is_empty() {
            return Err(SolveError::NoMatches);
        }

        // Phase 1: wide verification and coarse ranking.
        let verify_index = DetectedStarIndex::new(
            &sorted_stars,
            image_width,
            image_height,
            self.config.verify_config.max_match_distance_pixels,
        );
        let phase1_limit = self
            .config
            .phase1_hypothesis_cap
            .clamp(80, 900)
            .min(ranked_hypotheses.len());
        let mut phase1_candidates: Vec<(VerifyResult, u16, u16)> = Vec::new();
        let mut best_phase1_odds = f64::NEG_INFINITY;
        let mut best_phase1_matches = 0usize;
        let mut debug_hyp_samples = 0usize;
        let mut truth_best: Option<(f64, f64, usize, f64)> = None;
        let mut phase1_checked = 0usize;
        let min_phase1_checks = 24usize;

        for ranked in ranked_hypotheses.into_iter().take(phase1_limit) {
            let hypothesis = ranked.hypothesis;
            let votes = ranked.votes;
            let quad_support = ranked.quad_support;
            if self.config.timeout_ms > 0
                && start_time.elapsed().as_millis() > self.config.timeout_ms as u128
                && phase1_checked >= min_phase1_checks
            {
                break;
            }
            phase1_checked += 1;

            let mut result = verify_hypothesis_with_index(
                &hypothesis,
                &sorted_stars,
                self.index,
                image_width,
                image_height,
                &self.config.verify_config,
                Some(&verify_index),
                catalog_index,
            );
            result.hypothesis.log_odds += 2.1 * (votes as f64).ln_1p();
            result.hypothesis.log_odds += 1.3 * (quad_support as f64).ln_1p();
            best_phase1_odds = best_phase1_odds.max(result.hypothesis.log_odds);
            best_phase1_matches = best_phase1_matches.max(result.num_matched);
            if let Some((truth_ra_deg, truth_dec_deg, truth_scale_arcsec)) = debug_truth {
                let c = result.hypothesis.wcs.crval();
                let c_sep_arcsec = math::angular_separation(
                    &c,
                    &crate::core::types::RaDec::new(
                        truth_ra_deg.to_radians(),
                        truth_dec_deg.to_radians(),
                    ),
                )
                .to_degrees()
                    * 3600.0;
                let scale = result.hypothesis.wcs.pixel_scale_arcsec();
                let scale_pct = ((scale / truth_scale_arcsec) - 1.0).abs() * 100.0;
                let metric = c_sep_arcsec + 25.0 * scale_pct;
                let replace = match truth_best {
                    Some((best_metric, _, _, _)) => metric < best_metric,
                    None => true,
                };
                if replace {
                    truth_best = Some((metric, c_sep_arcsec, result.num_matched, scale_pct));
                }
            }
            if debug && debug_hyp_samples < 6 {
                eprintln!(
                    "citra-debug: phase1_sample idx={} matched={} log_odds={:.2} votes={} quad_support={} seed_pairs={}",
                    debug_hyp_samples,
                    result.num_matched,
                    result.hypothesis.log_odds,
                    votes,
                    quad_support,
                    hypothesis.star_matches.len()
                );
                debug_hyp_samples += 1;
            }

            if result.num_matched < 4 {
                continue;
            }
            if result.expected_coverage < 0.003 {
                continue;
            }
            if result.median_residual_pixels
                > self.config.verify_config.max_match_distance_pixels * 0.75
            {
                continue;
            }
            if result.rms_residual_pixels
                > self.config.verify_config.max_match_distance_pixels * 0.95
            {
                continue;
            }
            if result.hypothesis.log_odds < -45.0 {
                continue;
            }

            let tetra3_resilient_accept = votes >= 3
                && quad_support >= 1
                && result.num_matched >= 5
                && result.median_residual_pixels
                    <= self.config.verify_config.max_match_distance_pixels * 0.62;
            let baseline_accept = result.num_matched >= 6
                && quad_support >= 1
                && result.median_residual_pixels
                    <= self.config.verify_config.max_match_distance_pixels * 0.75;
            let sparse_accept = votes >= 8
                && quad_support >= 2
                && result.num_matched >= 4
                && result.detected_coverage >= 0.08
                && result.expected_coverage >= 0.06
                && result.median_residual_pixels
                    <= self.config.verify_config.max_match_distance_pixels * 0.45;
            let consensus_four_star_accept = votes >= 4
                && quad_support >= 2
                && result.num_matched == 4
                && result.hypothesis.log_odds >= 30.0
                && result.detected_coverage >= 0.09
                && result.median_residual_pixels
                    <= self.config.verify_config.max_match_distance_pixels * 0.48;
            if tetra3_resilient_accept
                || baseline_accept
                || sparse_accept
                || consensus_four_star_accept
            {
                phase1_candidates.push((result, votes, quad_support));
                if phase1_candidates.len() >= self.config.phase2_candidate_cap.saturating_mul(3)
                    && best_phase1_matches >= 8
                    && best_phase1_odds >= self.config.verify_config.log_odds_threshold + 8.0
                {
                    break;
                }
            }
        }
        if debug {
            eprintln!(
                "citra-debug: phase1_candidates={} best_phase1_matches={} best_phase1_odds={:.2}",
                phase1_candidates.len(),
                best_phase1_matches,
                best_phase1_odds
            );
            if let Some((metric, sep, matched, scale_pct)) = truth_best {
                eprintln!(
                    "citra-debug: phase1_truth_best metric={:.1} center_err_arcsec={:.1} scale_err_pct={:.2} matched={}",
                    metric, sep, scale_pct, matched
                );
            }
        }

        if phase1_candidates.is_empty() {
            return Err(SolveError::VerificationFailed(best_phase1_odds));
        }

        phase1_candidates.sort_by(|a, b| {
            let score_b = Self::verification_score(&b.0)
                + 2.0 * (b.1 as f64).ln_1p()
                + 1.6 * (b.2 as f64).ln_1p();
            let score_a = Self::verification_score(&a.0)
                + 2.0 * (a.1 as f64).ln_1p()
                + 1.6 * (a.2 as f64).ln_1p();
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if debug {
            for (idx, (cand, votes, quad_support)) in phase1_candidates.iter().take(12).enumerate()
            {
                eprintln!(
                    "citra-debug: phase1_rank idx={} matched={} exp_cov={:.3} det_cov={:.3} rms_px={:.2} med_px={:.2} odds={:.2} votes={} quad_support={} score={:.2}",
                    idx,
                    cand.num_matched,
                    cand.expected_coverage,
                    cand.detected_coverage,
                    cand.rms_residual_pixels,
                    cand.median_residual_pixels,
                    cand.hypothesis.log_odds,
                    votes,
                    quad_support,
                    Self::verification_score(cand)
                        + 2.0 * (*votes as f64).ln_1p()
                        + 1.6 * (*quad_support as f64).ln_1p()
                );
            }
        }
        phase1_candidates.truncate(self.config.phase2_candidate_cap.clamp(12, 96));

        // Phase 2: one-pass linear refinement + tight verification.
        let tight_config = VerifyConfig {
            position_sigma_pixels: self.config.verify_config.position_sigma_pixels.min(2.8),
            max_match_distance_pixels: (self.config.verify_config.max_match_distance_pixels * 0.82)
                .clamp(6.0, 11.0),
            false_positive_rate: self
                .config
                .verify_config
                .false_positive_rate
                .clamp(0.05, 0.35),
            log_odds_threshold: self.config.verify_config.log_odds_threshold + 2.0,
        };
        let tight_index = DetectedStarIndex::new(
            &sorted_stars,
            image_width,
            image_height,
            tight_config.max_match_distance_pixels,
        );

        let mut finalists: Vec<(VerifyResult, f64, u16, u16)> = Vec::new();
        let mut reject_fov = 0usize;
        let mut reject_matches = 0usize;
        let mut reject_cov = 0usize;
        let mut reject_rms = 0usize;
        let mut reject_med = 0usize;
        let mut phase2_checked = 0usize;
        let min_phase2_checks = 6usize;
        for (candidate, votes, quad_support) in phase1_candidates.into_iter() {
            if self.config.timeout_ms > 0
                && start_time.elapsed().as_millis() > self.config.timeout_ms as u128
                && phase2_checked >= min_phase2_checks
            {
                break;
            }
            phase2_checked += 1;

            let refined_wcs = if let Some(linear_refined) = refine_linear_wcs(
                &sorted_stars,
                &candidate.hypothesis.star_matches,
                &candidate.hypothesis.wcs,
                image_width,
                image_height,
            ) {
                linear_refined.wcs
            } else {
                candidate.hypothesis.wcs.clone()
            };

            let tight_hyp = Hypothesis::new(
                candidate.hypothesis.star_matches.clone(),
                refined_wcs,
                candidate.hypothesis.pattern_distance,
                candidate.hypothesis.source_quad_idx,
            );
            let tight = verify_hypothesis_with_index(
                &tight_hyp,
                &sorted_stars,
                self.index,
                image_width,
                image_height,
                &tight_config,
                Some(&tight_index),
                catalog_index,
            );

            let scale = tight.hypothesis.wcs.pixel_scale_arcsec();
            let fov_w = image_width as f64 * scale / 3600.0;
            let fov_h = image_height as f64 * scale / 3600.0;
            let min_allowed_fov = fov_hint_min * 0.8;
            let max_allowed_fov = fov_hint_max * 1.2;
            if fov_w < min_allowed_fov
                || fov_h < min_allowed_fov * 0.6
                || fov_w > max_allowed_fov
                || fov_h > max_allowed_fov
            {
                reject_fov += 1;
                continue;
            }

            let wide_field = fov_w > 24.0 || fov_h > 18.0;
            let vote_slack = if votes >= 8 || quad_support >= 3 {
                1usize
            } else {
                0usize
            };
            let min_matches = if wide_field { 5usize } else { 6usize }.saturating_sub(vote_slack);
            let max_rms = if wide_field { 12.0 } else { 10.0 };
            let max_med = if wide_field { 14.0 } else { 12.0 };
            let min_detected_cov =
                (if wide_field { 0.055 } else { 0.075 } - 0.006 * vote_slack as f64).max(0.04);
            let min_expected_cov =
                (if wide_field { 0.060 } else { 0.080 } - 0.006 * vote_slack as f64).max(0.04);
            let min_brightness_consistency = if tight.num_matched <= 6 { 0.44 } else { 0.30 };

            if tight.num_matched < min_matches {
                reject_matches += 1;
                continue;
            }
            if tight.detected_coverage < min_detected_cov {
                reject_cov += 1;
                continue;
            }
            if tight.expected_coverage < min_expected_cov {
                reject_cov += 1;
                continue;
            }
            if tight.brightness_consistency < min_brightness_consistency {
                reject_cov += 1;
                continue;
            }
            if tight.rms_residual_pixels > max_rms {
                reject_rms += 1;
                continue;
            }
            if tight.median_residual_pixels > max_med {
                reject_med += 1;
                continue;
            }
            if tight.hypothesis.log_odds < -40.0 {
                continue;
            }
            if tight.num_matched <= 5 && quad_support < 2 {
                continue;
            }
            let score = Self::verification_score(&tight)
                + 2.2 * (votes as f64).ln_1p()
                + 1.8 * (quad_support as f64).ln_1p();
            if debug && finalists.len() < 8 {
                eprintln!(
                    "citra-debug: finalist cand matched={} exp_cov={:.3} det_cov={:.3} rms_px={:.2} med_px={:.2} odds={:.2} votes={} quad_support={} score={:.2}",
                    tight.num_matched,
                    tight.expected_coverage,
                    tight.detected_coverage,
                    tight.rms_residual_pixels,
                    tight.median_residual_pixels,
                    tight.hypothesis.log_odds,
                    votes,
                    quad_support,
                    score
                );
            }
            finalists.push((tight, score, votes, quad_support));

            if finalists.len() >= 2 {
                finalists
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                if finalists[0].0.num_matched >= 14
                    && finalists[0].0.hypothesis.log_odds >= tight_config.log_odds_threshold + 10.0
                {
                    break;
                }
            }
        }
        if debug {
            eprintln!(
                "citra-debug: finalists={} rejects fov={} match={} cov={} rms={} med={}",
                finalists.len(),
                reject_fov,
                reject_matches,
                reject_cov,
                reject_rms,
                reject_med
            );
        }

        if finalists.is_empty() {
            return Err(SolveError::VerificationFailed(best_phase1_odds));
        }

        finalists.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        finalists.dedup_by(|a, b| Self::nearly_duplicate(&a.0, &b.0));
        if debug {
            if let Some((best, score, votes, quad_support)) = finalists.first() {
                eprintln!(
                    "citra-debug: best_final matched={} exp_cov={:.3} det_cov={:.3} rms_px={:.2} med_px={:.2} odds={:.2} votes={} quad_support={} score={:.2}",
                    best.num_matched,
                    best.expected_coverage,
                    best.detected_coverage,
                    best.rms_residual_pixels,
                    best.median_residual_pixels,
                    best.hypothesis.log_odds,
                    votes,
                    quad_support,
                    score
                );
            }
            if finalists.len() > 1 {
                let (runner, score, votes, quad_support) = &finalists[1];
                eprintln!(
                    "citra-debug: runner_up matched={} exp_cov={:.3} det_cov={:.3} rms_px={:.2} med_px={:.2} odds={:.2} votes={} quad_support={} score={:.2}",
                    runner.num_matched,
                    runner.expected_coverage,
                    runner.detected_coverage,
                    runner.rms_residual_pixels,
                    runner.median_residual_pixels,
                    runner.hypothesis.log_odds,
                    votes,
                    quad_support,
                    score
                );
            }
        }
        if finalists.len() > 1 {
            let margin = finalists[0].1 - finalists[1].1;
            let vote_dominant = finalists[0].2 >= finalists[1].2.saturating_add(2)
                || finalists[0].3 >= finalists[1].3.saturating_add(1);
            let dominant = vote_dominant
                || finalists[0].0.num_matched >= finalists[1].0.num_matched + 2
                || finalists[0].0.hypothesis.log_odds >= finalists[1].0.hypothesis.log_odds + 10.0;
            let runner_center_sep_deg = math::angular_separation(
                &finalists[0].0.hypothesis.wcs.crval(),
                &finalists[1].0.hypothesis.wcs.crval(),
            )
            .to_degrees();
            let best_scale = finalists[0].0.hypothesis.wcs.pixel_scale_arcsec().max(1e-6);
            let runner_scale = finalists[1].0.hypothesis.wcs.pixel_scale_arcsec().max(1e-6);
            let runner_scale_pct = ((best_scale / runner_scale) - 1.0).abs() * 100.0;
            let family_consensus = runner_center_sep_deg < 1.0 && runner_scale_pct < 2.5;

            let adaptive_margin = if finalists[0].0.num_matched < 8 {
                self.config.min_solution_margin.max(8.0)
            } else {
                self.config.min_solution_margin
            };
            let residual_advantage = finalists[0].0.median_residual_pixels + 0.7
                < finalists[1].0.median_residual_pixels
                || finalists[0].0.rms_residual_pixels + 1.0 < finalists[1].0.rms_residual_pixels;
            let coverage_advantage = finalists[0].0.detected_coverage + 0.025
                > finalists[1].0.detected_coverage
                || finalists[0].0.expected_coverage + 0.020 > finalists[1].0.expected_coverage;
            let high_confidence_best = finalists[0].0.num_matched >= 6
                && finalists[0].0.hypothesis.log_odds >= 32.0
                && finalists[0].0.detected_coverage >= 0.09;

            if margin < adaptive_margin
                && !dominant
                && !family_consensus
                && !(high_confidence_best && (residual_advantage || coverage_advantage))
            {
                return Err(SolveError::VerificationFailed(
                    finalists[0].0.hypothesis.log_odds,
                ));
            }
        }

        let best_result = finalists.remove(0).0;

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
        let rms_cap = (solution.pixel_scale_arcsec * 10.0).clamp(120.0, 900.0);
        if solution.rms_arcsec > rms_cap {
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
            .with_tetra_tolerance((self.config.ratio_tolerance * 1.8).clamp(0.020, 0.055))
            .with_max_hash_bins(self.config.max_hash_bins)
            .with_max_patterns_per_bin(self.config.max_patterns_per_bin)
            .with_scale_hint(
                ((image_width as f64).powi(2) + (image_height as f64).powi(2)).sqrt(),
                self.config
                    .expected_fov_min_deg
                    .unwrap_or(self.index.fov_range_deg().0 as f64),
                self.config
                    .expected_fov_max_deg
                    .unwrap_or(self.index.fov_range_deg().1 as f64),
            )
            .with_max_scale_band_delta(4);

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

fn parse_debug_truth(raw: &str) -> Option<(f64, f64, f64)> {
    let mut it = raw.split(',');
    let ra = it.next()?.trim().parse::<f64>().ok()?;
    let dec = it.next()?.trim().parse::<f64>().ok()?;
    let scale = it.next()?.trim().parse::<f64>().ok()?;
    if !ra.is_finite() || !dec.is_finite() || !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    Some((ra, dec, scale))
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
