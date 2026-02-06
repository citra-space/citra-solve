//! Debug hypothesis generation for the correct pattern match.

use citra_solve::catalog::Index;
use citra_solve::core::types::RaDec;
use citra_solve::core::math::angular_separation;
use citra_solve::extract::{extract_stars, ExtractionConfig};
use citra_solve::pattern::generate_quads;
use citra_solve::wcs::Wcs;

const PERMUTATIONS_4: [[usize; 4]; 24] = [
    [0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1], [0, 3, 1, 2], [0, 3, 2, 1],
    [1, 0, 2, 3], [1, 0, 3, 2], [1, 2, 0, 3], [1, 2, 3, 0], [1, 3, 0, 2], [1, 3, 2, 0],
    [2, 0, 1, 3], [2, 0, 3, 1], [2, 1, 0, 3], [2, 1, 3, 0], [2, 3, 0, 1], [2, 3, 1, 0],
    [3, 0, 1, 2], [3, 0, 2, 1], [3, 1, 0, 2], [3, 1, 2, 0], [3, 2, 0, 1], [3, 2, 1, 0],
];

fn main() {
    let image_path = "exposure_5.jpg";
    let index_path = "hipparcos_deep.idx";

    // Known good WCS from astrometry.net
    let true_crval = RaDec::from_degrees(14.3012483353, -9.12424543834);
    let true_crpix = (903.227925618, 575.047943115);
    let true_cd = [
        [-0.0159317827922, 0.00630093735133],
        [-0.00629294246889, -0.0159290388907],
    ];
    let true_wcs = Wcs::new(true_crpix, true_crval, true_cd);

    let index = Index::open(index_path).expect("Failed to open index");

    // Extract stars
    let config = ExtractionConfig {
        max_stars: 50,
        sigma_threshold: 5.0,
        ..Default::default()
    };
    let stars = extract_stars(image_path, &config).expect("Failed to extract");

    // Generate quads
    let quads = generate_quads(&stars, 30, 100);
    let quad0 = &quads[0];

    println!("Quad 0 detected star indices: {:?}", quad0.star_indices);
    println!("Quad 0 detected star positions:");
    for &idx in &quad0.star_indices {
        println!("  Det {}: ({:.1}, {:.1})", idx, stars[idx].x, stars[idx].y);
    }
    println!();

    // Catalog stars: [51, 272, 306, 376]
    let cat_indices = [51u32, 376, 306, 272]; // Order from image quad matching
    let cat_stars: Vec<_> = cat_indices
        .iter()
        .map(|&idx| index.get_catalog_star(idx).unwrap())
        .collect();

    println!("Catalog stars [51, 376, 306, 272]:");
    for (idx, cat) in cat_indices.iter().zip(&cat_stars) {
        let (px, py) = true_wcs.sky_to_pixel(&cat.position);
        println!("  Cat {}: sky=({:.4}°, {:.4}°) -> pixel=({:.1}, {:.1})",
            idx, cat.position.ra_deg(), cat.position.dec_deg(), px, py);
    }
    println!();

    // Try all 24 permutations
    println!("Trying all 24 permutations:");
    let image_width = 1456u32;
    let image_height = 1088u32;

    for (perm_idx, perm) in PERMUTATIONS_4.iter().enumerate() {
        // Create star matches with this permutation
        // perm[i] maps detected star position i to catalog star position perm[i]
        let matches: Vec<(usize, _)> = quad0
            .star_indices
            .iter()
            .enumerate()
            .map(|(i, &det_idx)| (det_idx, cat_stars[perm[i]].clone()))
            .collect();

        // Estimate WCS
        if let Some(wcs) = estimate_wcs(&stars, &matches, image_width, image_height) {
            let center = wcs.crval();
            let scale = wcs.pixel_scale_arcsec();

            // Compute residual
            let mut sum_sq = 0.0;
            for (det_idx, cat) in &matches {
                let (pred_x, pred_y) = wcs.sky_to_pixel(&cat.position);
                let dx = stars[*det_idx].x - pred_x;
                let dy = stars[*det_idx].y - pred_y;
                sum_sq += dx*dx + dy*dy;
            }
            let rms = (sum_sq / matches.len() as f64).sqrt();

            // Check if close to true solution
            let sep = angular_separation(&center, &true_crval).to_degrees();
            let close = sep < 5.0;

            let marker = if close { " <<<< CLOSE" } else { "" };
            println!("  Perm {:2} {:?}: center=({:7.3}°,{:7.3}°) scale={:.1}\"/px rms={:.1}px sep={:.1}°{}",
                perm_idx, perm, center.ra_deg(), center.dec_deg(), scale, rms, sep, marker);

            if close {
                println!("    Matches:");
                for (det_idx, cat) in &matches {
                    let (pred_x, pred_y) = wcs.sky_to_pixel(&cat.position);
                    println!("      Det {} ({:.1},{:.1}) -> Cat {} pred ({:.1},{:.1})",
                        det_idx, stars[*det_idx].x, stars[*det_idx].y, cat.id, pred_x, pred_y);
                }
            }
        }
    }
}

fn estimate_wcs(
    detected_stars: &[citra_solve::core::types::DetectedStar],
    matches: &[(usize, citra_solve::core::types::CatalogStar)],
    _image_width: u32,
    _image_height: u32,
) -> Option<Wcs> {
    if matches.len() < 3 {
        return None;
    }

    // Use first matched star as reference
    let (ref_det_idx, ref_cat) = &matches[0];
    let ref_det = &detected_stars[*ref_det_idx];
    let crpix = (ref_det.x, ref_det.y);
    let crval = ref_cat.position;

    // Precompute trig values for crval
    let crval_ra = crval.ra;
    let crval_dec = crval.dec;
    let sin_crval_dec = crval_dec.sin();
    let cos_crval_dec = crval_dec.cos();

    // Build matrices for least squares
    let mut a_mat: Vec<[f64; 2]> = Vec::new();
    let mut b_x: Vec<f64> = Vec::new();
    let mut b_y: Vec<f64> = Vec::new();

    for (det_idx, cat_star) in matches.iter().skip(1) {
        let det = &detected_stars[*det_idx];

        let dx = det.x - crpix.0;
        let dy = det.y - crpix.1;

        let ra = cat_star.position.ra;
        let dec = cat_star.position.dec;

        let sin_dec = dec.sin();
        let cos_dec = dec.cos();
        let delta_ra = ra - crval_ra;
        let sin_delta_ra = delta_ra.sin();
        let cos_delta_ra = delta_ra.cos();

        let denom = sin_crval_dec * sin_dec + cos_crval_dec * cos_dec * cos_delta_ra;

        if denom < 0.1 {
            continue;
        }

        let xi = (cos_dec * sin_delta_ra) / denom;
        let eta = (cos_crval_dec * sin_dec - sin_crval_dec * cos_dec * cos_delta_ra) / denom;

        a_mat.push([xi, eta]);
        b_x.push(dx);
        b_y.push(dy);
    }

    if a_mat.len() < 2 {
        return None;
    }

    // Solve least squares using normal equations
    let mut ata = [[0.0; 2]; 2];
    for row in &a_mat {
        ata[0][0] += row[0] * row[0];
        ata[0][1] += row[0] * row[1];
        ata[1][0] += row[1] * row[0];
        ata[1][1] += row[1] * row[1];
    }

    let mut atb_x = [0.0; 2];
    let mut atb_y = [0.0; 2];
    for (i, row) in a_mat.iter().enumerate() {
        atb_x[0] += row[0] * b_x[i];
        atb_x[1] += row[1] * b_x[i];
        atb_y[0] += row[0] * b_y[i];
        atb_y[1] += row[1] * b_y[i];
    }

    let det = ata[0][0] * ata[1][1] - ata[0][1] * ata[1][0];
    if det.abs() < 1e-20 {
        return None;
    }

    let inv_det = 1.0 / det;
    let ata_inv = [
        [ata[1][1] * inv_det, -ata[0][1] * inv_det],
        [-ata[1][0] * inv_det, ata[0][0] * inv_det],
    ];

    let cd11 = ata_inv[0][0] * atb_x[0] + ata_inv[0][1] * atb_x[1];
    let cd12 = ata_inv[1][0] * atb_x[0] + ata_inv[1][1] * atb_x[1];
    let cd21 = ata_inv[0][0] * atb_y[0] + ata_inv[0][1] * atb_y[1];
    let cd22 = ata_inv[1][0] * atb_y[0] + ata_inv[1][1] * atb_y[1];

    let cd_det = cd11 * cd22 - cd12 * cd21;
    if cd_det.abs() < 1e-20 {
        return None;
    }

    let rad_to_deg = 180.0 / std::f64::consts::PI;
    let cd_inv_scale = rad_to_deg / cd_det;

    let cd = [
        [cd22 * cd_inv_scale, -cd12 * cd_inv_scale],
        [-cd21 * cd_inv_scale, cd11 * cd_inv_scale],
    ];

    Some(Wcs::new(crpix, crval, cd))
}
