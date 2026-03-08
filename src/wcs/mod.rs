//! World Coordinate System (WCS) transformations.
//!
//! This module implements the FITS WCS standard for transforming
//! between pixel coordinates and sky coordinates.

mod distortion;
mod projection;
mod transform;

pub use distortion::{fit_sip_distortion, SipDistortion};
pub use projection::Wcs;
pub use transform::WcsWithDistortion;
