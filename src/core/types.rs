//! Core type definitions for coordinates and stars.

use std::f64::consts::PI;

/// Right Ascension and Declination in radians.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaDec {
    /// Right ascension in radians [0, 2π)
    pub ra: f64,
    /// Declination in radians [-π/2, π/2]
    pub dec: f64,
}

impl RaDec {
    /// Create a new RaDec from radians.
    #[inline]
    pub fn new(ra: f64, dec: f64) -> Self {
        Self { ra, dec }
    }

    /// Create a new RaDec from degrees.
    #[inline]
    pub fn from_degrees(ra_deg: f64, dec_deg: f64) -> Self {
        Self {
            ra: ra_deg.to_radians(),
            dec: dec_deg.to_radians(),
        }
    }

    /// Get right ascension in degrees.
    #[inline]
    pub fn ra_deg(&self) -> f64 {
        self.ra.to_degrees()
    }

    /// Get declination in degrees.
    #[inline]
    pub fn dec_deg(&self) -> f64 {
        self.dec.to_degrees()
    }

    /// Get right ascension in hours.
    #[inline]
    pub fn ra_hours(&self) -> f64 {
        self.ra * 12.0 / PI
    }

    /// Convert to a unit 3D vector.
    #[inline]
    pub fn to_vec3(&self) -> Vec3 {
        Vec3::from_radec(self)
    }

    /// Normalize RA to [0, 2π) and clamp Dec to [-π/2, π/2].
    pub fn normalize(&self) -> Self {
        let mut ra = self.ra % (2.0 * PI);
        if ra < 0.0 {
            ra += 2.0 * PI;
        }
        let dec = self.dec.clamp(-PI / 2.0, PI / 2.0);
        Self { ra, dec }
    }
}

impl Default for RaDec {
    fn default() -> Self {
        Self { ra: 0.0, dec: 0.0 }
    }
}

/// 3D unit vector for spherical calculations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    /// Create a new Vec3.
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Create a unit vector from RaDec.
    #[inline]
    pub fn from_radec(radec: &RaDec) -> Self {
        let cos_dec = radec.dec.cos();
        Self {
            x: cos_dec * radec.ra.cos(),
            y: cos_dec * radec.ra.sin(),
            z: radec.dec.sin(),
        }
    }

    /// Convert to RaDec.
    #[inline]
    pub fn to_radec(&self) -> RaDec {
        let dec = self.z.asin();
        let ra = self.y.atan2(self.x);
        RaDec::new(ra, dec).normalize()
    }

    /// Dot product.
    #[inline]
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product.
    #[inline]
    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Vector magnitude (length).
    #[inline]
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Normalize to unit length.
    #[inline]
    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        if mag > 0.0 {
            Self {
                x: self.x / mag,
                y: self.y / mag,
                z: self.z / mag,
            }
        } else {
            *self
        }
    }

    /// Angular separation between two unit vectors in radians.
    #[inline]
    pub fn angle_to(&self, other: &Self) -> f64 {
        // Use atan2 for numerical stability at small and large angles
        let cross_mag = self.cross(other).magnitude();
        let dot = self.dot(other);
        cross_mag.atan2(dot)
    }

    /// Scale the vector.
    #[inline]
    pub fn scale(&self, s: f64) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    /// Add two vectors.
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    /// Subtract two vectors.
    #[inline]
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    /// Project this unit vector onto the tangent plane at `center` (gnomonic projection).
    ///
    /// Returns (xi, eta) standard coordinates in radians.
    /// This matches what a camera with TAN projection produces, ensuring
    /// that distance ratios in the tangent plane match pixel distance ratios.
    pub fn gnomonic_project(&self, center: &Vec3) -> Option<(f64, f64)> {
        let dot = self.dot(center);
        if dot <= 0.0 {
            return None; // Behind the tangent point
        }

        // Build orthonormal basis at the tangent point
        // center is the normal, we need two tangent vectors
        let radec = center.to_radec();
        let sin_ra = radec.ra.sin();
        let cos_ra = radec.ra.cos();
        let sin_dec = radec.dec.sin();
        let cos_dec = radec.dec.cos();

        // e_xi points East (direction of increasing RA)
        let e_xi = Vec3::new(-sin_ra, cos_ra, 0.0);
        // e_eta points North (direction of increasing Dec)
        let e_eta = Vec3::new(-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec);

        // Project: standard coordinates = (star · e_xi, star · e_eta) / (star · center)
        let xi = self.dot(&e_xi) / dot;
        let eta = self.dot(&e_eta) / dot;

        Some((xi, eta))
    }
}

impl Default for Vec3 {
    fn default() -> Self {
        Self::new(1.0, 0.0, 0.0)
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self::add(&self, &other)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self::sub(&self, &other)
    }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        self.scale(s)
    }
}

/// A star detected in an image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectedStar {
    /// X pixel coordinate (column).
    pub x: f64,
    /// Y pixel coordinate (row).
    pub y: f64,
    /// Relative brightness/flux (for sorting by brightness).
    pub flux: f64,
}

impl DetectedStar {
    /// Create a new detected star.
    #[inline]
    pub fn new(x: f64, y: f64, flux: f64) -> Self {
        Self { x, y, flux }
    }

    /// Euclidean distance to another star in pixel space.
    #[inline]
    pub fn distance_to(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// A star from the catalog.
#[derive(Debug, Clone, Copy)]
pub struct CatalogStar {
    /// Catalog ID (e.g., Hipparcos number).
    pub id: u32,
    /// Sky position.
    pub position: RaDec,
    /// Visual magnitude.
    pub magnitude: f32,
}

impl CatalogStar {
    /// Create a new catalog star.
    pub fn new(id: u32, ra: f64, dec: f64, magnitude: f32) -> Self {
        Self {
            id,
            position: RaDec::new(ra, dec),
            magnitude,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn test_radec_from_degrees() {
        let rd = RaDec::from_degrees(180.0, 45.0);
        assert!((rd.ra - PI).abs() < 1e-10);
        assert!((rd.dec - FRAC_PI_2 / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec3_from_radec() {
        // North pole
        let pole = RaDec::new(0.0, FRAC_PI_2);
        let v = Vec3::from_radec(&pole);
        assert!((v.x).abs() < 1e-10);
        assert!((v.y).abs() < 1e-10);
        assert!((v.z - 1.0).abs() < 1e-10);

        // On equator at RA=0
        let eq = RaDec::new(0.0, 0.0);
        let v = Vec3::from_radec(&eq);
        assert!((v.x - 1.0).abs() < 1e-10);
        assert!((v.y).abs() < 1e-10);
        assert!((v.z).abs() < 1e-10);
    }

    #[test]
    fn test_vec3_roundtrip() {
        let original = RaDec::from_degrees(45.0, 30.0);
        let v = Vec3::from_radec(&original);
        let back = v.to_radec();
        assert!((original.ra - back.ra).abs() < 1e-10);
        assert!((original.dec - back.dec).abs() < 1e-10);
    }

    #[test]
    fn test_angle_between() {
        let v1 = Vec3::new(1.0, 0.0, 0.0);
        let v2 = Vec3::new(0.0, 1.0, 0.0);
        let angle = v1.angle_to(&v2);
        assert!((angle - FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn test_detected_star_distance() {
        let s1 = DetectedStar::new(0.0, 0.0, 100.0);
        let s2 = DetectedStar::new(3.0, 4.0, 100.0);
        assert!((s1.distance_to(&s2) - 5.0).abs() < 1e-10);
    }
}
