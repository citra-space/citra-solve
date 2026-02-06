//! World Coordinate System (WCS) transformations.
//!
//! This module implements the FITS WCS standard for transforming
//! between pixel coordinates and sky coordinates.

mod projection;
mod distortion;
mod transform;

pub use projection::Wcs;
pub use distortion::SipDistortion;
