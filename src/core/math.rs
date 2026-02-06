//! Mathematical utilities for spherical geometry and transformations.

use super::types::{RaDec, Vec3};
use std::f64::consts::PI;

/// Angular separation between two sky positions in radians.
#[inline]
pub fn angular_separation(a: &RaDec, b: &RaDec) -> f64 {
    Vec3::from_radec(a).angle_to(&Vec3::from_radec(b))
}

/// Angular separation in arcseconds.
#[inline]
pub fn angular_separation_arcsec(a: &RaDec, b: &RaDec) -> f64 {
    angular_separation(a, b).to_degrees() * 3600.0
}

/// Great circle distance using Haversine formula (alternative implementation).
/// More numerically stable for small distances.
pub fn haversine_distance(a: &RaDec, b: &RaDec) -> f64 {
    let d_ra = b.ra - a.ra;
    let d_dec = b.dec - a.dec;

    let h = (d_dec / 2.0).sin().powi(2)
        + a.dec.cos() * b.dec.cos() * (d_ra / 2.0).sin().powi(2);

    2.0 * h.sqrt().asin()
}

/// Convert degrees to radians.
#[inline]
pub fn deg_to_rad(deg: f64) -> f64 {
    deg * PI / 180.0
}

/// Convert radians to degrees.
#[inline]
pub fn rad_to_deg(rad: f64) -> f64 {
    rad * 180.0 / PI
}

/// Convert degrees to arcseconds.
#[inline]
pub fn deg_to_arcsec(deg: f64) -> f64 {
    deg * 3600.0
}

/// Convert arcseconds to degrees.
#[inline]
pub fn arcsec_to_deg(arcsec: f64) -> f64 {
    arcsec / 3600.0
}

/// Convert arcseconds to radians.
#[inline]
pub fn arcsec_to_rad(arcsec: f64) -> f64 {
    deg_to_rad(arcsec_to_deg(arcsec))
}

/// Convert radians to arcseconds.
#[inline]
pub fn rad_to_arcsec(rad: f64) -> f64 {
    deg_to_arcsec(rad_to_deg(rad))
}

/// Compute the centroid of a set of unit vectors (on the sphere).
pub fn spherical_centroid(points: &[Vec3]) -> Vec3 {
    if points.is_empty() {
        return Vec3::default();
    }
    let sum = points.iter().fold(Vec3::new(0.0, 0.0, 0.0), |acc, p| acc + *p);
    sum.normalize()
}

/// Compute the position angle from point A to point B.
/// Returns the angle measured east from north, in radians [0, 2π).
pub fn position_angle(from: &RaDec, to: &RaDec) -> f64 {
    let d_ra = to.ra - from.ra;
    let y = d_ra.sin() * to.dec.cos();
    let x = from.dec.cos() * to.dec.sin() - from.dec.sin() * to.dec.cos() * d_ra.cos();
    let mut pa = y.atan2(x);
    if pa < 0.0 {
        pa += 2.0 * PI;
    }
    pa
}

/// Rotate a point around an axis by an angle (Rodrigues' rotation formula).
pub fn rotate_around_axis(point: &Vec3, axis: &Vec3, angle: f64) -> Vec3 {
    let k = axis.normalize();
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    // v' = v*cos(a) + (k x v)*sin(a) + k*(k·v)*(1-cos(a))
    let k_cross_v = k.cross(point);
    let k_dot_v = k.dot(point);

    Vec3 {
        x: point.x * cos_a + k_cross_v.x * sin_a + k.x * k_dot_v * (1.0 - cos_a),
        y: point.y * cos_a + k_cross_v.y * sin_a + k.y * k_dot_v * (1.0 - cos_a),
        z: point.z * cos_a + k_cross_v.z * sin_a + k.z * k_dot_v * (1.0 - cos_a),
    }
}

/// Build a rotation matrix from Euler angles (ZYZ convention).
/// Returns a 3x3 matrix as [[f64; 3]; 3].
pub fn euler_to_matrix(phi: f64, theta: f64, psi: f64) -> [[f64; 3]; 3] {
    let (s1, c1) = phi.sin_cos();
    let (s2, c2) = theta.sin_cos();
    let (s3, c3) = psi.sin_cos();

    [
        [
            c1 * c2 * c3 - s1 * s3,
            -c1 * c2 * s3 - s1 * c3,
            c1 * s2,
        ],
        [
            s1 * c2 * c3 + c1 * s3,
            -s1 * c2 * s3 + c1 * c3,
            s1 * s2,
        ],
        [-s2 * c3, s2 * s3, c2],
    ]
}

/// Apply a 3x3 rotation matrix to a Vec3.
pub fn apply_matrix(m: &[[f64; 3]; 3], v: &Vec3) -> Vec3 {
    Vec3 {
        x: m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        y: m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        z: m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    }
}

/// Transpose a 3x3 matrix (for inverse rotation).
pub fn transpose_matrix(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// Compute the median of a slice (modifies the input).
pub fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

/// Compute mean and standard deviation.
pub fn mean_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    (mean, variance.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn test_angular_separation() {
        let a = RaDec::new(0.0, 0.0);
        let b = RaDec::new(FRAC_PI_2, 0.0);
        let sep = angular_separation(&a, &b);
        assert!((sep - FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn test_haversine_vs_vec3() {
        let a = RaDec::from_degrees(10.0, 20.0);
        let b = RaDec::from_degrees(10.5, 20.5);
        let sep1 = angular_separation(&a, &b);
        let sep2 = haversine_distance(&a, &b);
        assert!((sep1 - sep2).abs() < 1e-10);
    }

    #[test]
    fn test_rotate_around_axis() {
        let v = Vec3::new(1.0, 0.0, 0.0);
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let rotated = rotate_around_axis(&v, &axis, FRAC_PI_2);
        assert!((rotated.x).abs() < 1e-10);
        assert!((rotated.y - 1.0).abs() < 1e-10);
        assert!((rotated.z).abs() < 1e-10);
    }

    #[test]
    fn test_median() {
        let mut v = vec![3.0, 1.0, 2.0];
        assert!((median(&mut v) - 2.0).abs() < 1e-10);

        let mut v = vec![4.0, 1.0, 3.0, 2.0];
        assert!((median(&mut v) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_mean_std() {
        let v = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let (mean, std) = mean_std(&v);
        assert!((mean - 5.0).abs() < 1e-10);
        assert!((std - 2.0).abs() < 1e-10);
    }
}
