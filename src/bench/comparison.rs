//! Comparison with astrometry.net solutions.

use std::fs;
use std::path::Path;
use std::process::Command;

use crate::core::math::angular_separation_arcsec;
use crate::core::types::RaDec;
use crate::solver::Solution;
use crate::wcs::Wcs;

/// Result of comparing Chameleon solution with astrometry.net.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Chameleon solved successfully.
    pub chameleon_solved: bool,
    /// Astrometry.net solved successfully.
    pub astrometry_solved: bool,
    /// Position difference between solutions (arcsec).
    pub position_diff_arcsec: Option<f64>,
    /// Rotation difference (arcsec).
    pub rotation_diff_arcsec: Option<f64>,
    /// Scale difference (ppm).
    pub scale_diff_ppm: Option<f64>,
    /// Chameleon solve time.
    pub chameleon_time_ms: u64,
    /// Astrometry.net solve time.
    pub astrometry_time_ms: u64,
}

/// Parse a WCS from an astrometry.net .wcs file.
pub fn parse_astrometry_wcs<P: AsRef<Path>>(path: P) -> Option<Wcs> {
    let content = fs::read_to_string(path).ok()?;

    let mut crpix1: Option<f64> = None;
    let mut crpix2: Option<f64> = None;
    let mut crval1: Option<f64> = None;
    let mut crval2: Option<f64> = None;
    let mut cd1_1: Option<f64> = None;
    let mut cd1_2: Option<f64> = None;
    let mut cd2_1: Option<f64> = None;
    let mut cd2_2: Option<f64> = None;

    for line in content.lines() {
        if line.starts_with("CRPIX1") {
            crpix1 = parse_fits_value(line);
        } else if line.starts_with("CRPIX2") {
            crpix2 = parse_fits_value(line);
        } else if line.starts_with("CRVAL1") {
            crval1 = parse_fits_value(line);
        } else if line.starts_with("CRVAL2") {
            crval2 = parse_fits_value(line);
        } else if line.starts_with("CD1_1") {
            cd1_1 = parse_fits_value(line);
        } else if line.starts_with("CD1_2") {
            cd1_2 = parse_fits_value(line);
        } else if line.starts_with("CD2_1") {
            cd2_1 = parse_fits_value(line);
        } else if line.starts_with("CD2_2") {
            cd2_2 = parse_fits_value(line);
        }
    }

    // Check if we have all required values
    let crpix = (crpix1?, crpix2?);
    let crval = RaDec::from_degrees(crval1?, crval2?);
    let cd = [
        [cd1_1.unwrap_or(0.0), cd1_2.unwrap_or(0.0)],
        [cd2_1.unwrap_or(0.0), cd2_2.unwrap_or(0.0)],
    ];

    // Convert from 1-indexed FITS to 0-indexed
    Some(Wcs::new((crpix.0 - 1.0, crpix.1 - 1.0), crval, cd))
}

fn parse_fits_value(line: &str) -> Option<f64> {
    let parts: Vec<&str> = line.split('=').collect();
    if parts.len() < 2 {
        return None;
    }
    let value_part = parts[1].split('/').next()?;
    value_part.trim().parse().ok()
}

/// Compare a Chameleon solution with an astrometry.net solution.
pub fn compare_solutions(chameleon: &Solution, astrometry_wcs: &Wcs) -> ComparisonResult {
    let cham_center = chameleon.center;
    let astr_center = astrometry_wcs.crval();

    let position_diff = angular_separation_arcsec(&cham_center, &astr_center);

    let rotation_diff = (chameleon.rotation_deg - astrometry_wcs.rotation_deg()).abs() * 3600.0;

    let cham_scale = chameleon.pixel_scale_arcsec;
    let astr_scale = astrometry_wcs.pixel_scale_arcsec();
    let scale_diff = ((cham_scale - astr_scale) / astr_scale).abs() * 1_000_000.0;

    ComparisonResult {
        chameleon_solved: true,
        astrometry_solved: true,
        position_diff_arcsec: Some(position_diff),
        rotation_diff_arcsec: Some(rotation_diff),
        scale_diff_ppm: Some(scale_diff),
        chameleon_time_ms: 0, // Filled in by caller
        astrometry_time_ms: 0,
    }
}

/// Run astrometry.net's solve-field command.
///
/// Returns the path to the generated .wcs file if successful.
pub fn run_astrometry_net<P: AsRef<Path>>(
    image_path: P,
    scale_low: f64,
    scale_high: f64,
    timeout_seconds: u32,
) -> Result<String, String> {
    let image_path = image_path.as_ref();

    // Check if solve-field is available
    let check = Command::new("which").arg("solve-field").output();
    if check.is_err() || !check.unwrap().status.success() {
        return Err("astrometry.net solve-field not found in PATH".to_string());
    }

    // Run solve-field
    let output = Command::new("solve-field")
        .arg("--scale-units")
        .arg("degwidth")
        .arg("--scale-low")
        .arg(scale_low.to_string())
        .arg("--scale-high")
        .arg(scale_high.to_string())
        .arg("--no-plots")
        .arg("--no-verify")
        .arg("--crpix-center")
        .arg("--cpulimit")
        .arg(timeout_seconds.to_string())
        .arg("--overwrite")
        .arg(image_path)
        .output()
        .map_err(|e| format!("Failed to run solve-field: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("solve-field failed: {}", stderr));
    }

    // Look for the .wcs file
    let wcs_path = image_path.with_extension("wcs");
    if wcs_path.exists() {
        Ok(wcs_path.to_string_lossy().to_string())
    } else {
        Err("No solution found (no .wcs file generated)".to_string())
    }
}

/// Generate a star list file for astrometry.net from detected stars.
pub fn write_xylist<P: AsRef<Path>>(
    path: P,
    stars: &[(f64, f64, f64)], // (x, y, flux)
) -> Result<(), std::io::Error> {
    use std::io::Write;

    let mut file = fs::File::create(path)?;

    // Write simple text format (can also write FITS, but text is easier)
    writeln!(file, "# X Y FLUX")?;
    for (x, y, flux) in stars {
        writeln!(file, "{:.6} {:.6} {:.6}", x, y, flux)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_fits_value() {
        let line = "CRPIX1  =           512.00000 / Reference pixel";
        let value = parse_fits_value(line);
        assert!(value.is_some());
        assert!((value.unwrap() - 512.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_fits_value_scientific() {
        let line = "CD1_1   =  -2.77778E-04 / CD matrix element";
        let value = parse_fits_value(line);
        assert!(value.is_some());
    }
}
