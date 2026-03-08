//! Debug index to see if patterns exist for our matched catalog stars.

use citra_solve::catalog::Index;
use citra_solve::core::math::angular_separation;
use citra_solve::core::types::RaDec;

fn main() {
    let index_path = "hipparcos_deep.idx";
    let index = Index::open(index_path).expect("Failed to open index");

    let (fov_min, fov_max) = index.fov_range_deg();
    println!(
        "Index: {} stars, {} patterns",
        index.num_stars(),
        index.num_patterns()
    );
    println!("FOV range: {:.1}° - {:.1}°\n", fov_min, fov_max);

    // The catalog stars we matched in our image
    let matched_stars = [
        51u32, 376, 306, 272, 321, 1075, 1427, 1231, 3550, 2851, 3784, 2444, 1215, 5733, 1886,
    ];

    // Get positions of these stars
    let mut star_positions: Vec<(u32, RaDec)> = Vec::new();
    for &idx in &matched_stars {
        if let Ok(star) = index.get_star(idx) {
            star_positions.push((idx, star.to_radec()));
        }
    }

    // Compute angular separations between matched stars
    println!("Angular separations between matched catalog stars:");
    for i in 0..star_positions.len() {
        for j in (i + 1)..star_positions.len() {
            let sep = angular_separation(&star_positions[i].1, &star_positions[j].1).to_degrees();
            if sep < 30.0 {
                println!(
                    "  Stars {} - {}: {:.2}°",
                    star_positions[i].0, star_positions[j].0, sep
                );
            }
        }
    }
    println!();

    // The first quad we found: stars [51, 376, 306, 272]
    // Compute what the quad ratios should be in angular terms
    let quad_stars = [51u32, 376, 306, 272];
    println!("Computing angular quad ratios for stars {:?}:", quad_stars);

    let mut positions: Vec<RaDec> = Vec::new();
    for &idx in &quad_stars {
        let star = index.get_star(idx).unwrap();
        positions.push(star.to_radec());
    }

    // Compute all 6 pairwise angular distances
    let mut angular_dists: Vec<f64> = Vec::new();
    for i in 0..4 {
        for j in (i + 1)..4 {
            let sep = angular_separation(&positions[i], &positions[j]).to_degrees();
            angular_dists.push(sep);
            println!("  {} - {}: {:.4}°", quad_stars[i], quad_stars[j], sep);
        }
    }
    println!();

    // Find max and compute ratios
    let max_dist = angular_dists.iter().cloned().fold(0.0f64, f64::max);
    println!("Max angular distance: {:.4}°", max_dist);

    let mut ratios: Vec<f64> = angular_dists.iter().map(|d| d / max_dist).collect();
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "Angular ratios (sorted): [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        ratios[0], ratios[1], ratios[2], ratios[3], ratios[4]
    );
    println!();

    // Check if this is within the FOV range
    println!(
        "Is max distance ({:.2}°) within FOV range ({:.1}° - {:.1}°)? {}",
        max_dist,
        fov_min,
        fov_max,
        if max_dist >= fov_min as f64 && max_dist <= fov_max as f64 {
            "YES"
        } else {
            "NO"
        }
    );
    println!();

    // Scan index for patterns containing our matched stars
    println!("Scanning index for patterns containing matched stars...");
    let matched_set: std::collections::HashSet<u32> = matched_stars.iter().cloned().collect();

    // We need to iterate through bins to find patterns
    let mut patterns_with_matched = 0;
    let mut patterns_with_4_matched = 0;

    for bin in 0..index.num_bins() {
        if let Ok(patterns) = index.get_patterns_in_bin(bin) {
            for pattern in patterns {
                let stars: Vec<u32> = pattern.star_indices.iter().map(|&s| s as u32).collect();
                let in_matched: usize = stars.iter().filter(|s| matched_set.contains(s)).count();

                if in_matched >= 3 {
                    patterns_with_matched += 1;
                    if in_matched == 4 {
                        patterns_with_4_matched += 1;
                        if patterns_with_4_matched <= 5 {
                            println!("  Pattern with all 4 in matched set: {:?}", stars);
                            let r = pattern.ratios();
                            println!(
                                "    Ratios: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                                r[0], r[1], r[2], r[3], r[4]
                            );
                        }
                    }
                }
            }
        }
    }

    println!(
        "\nFound {} patterns with 3+ matched stars",
        patterns_with_matched
    );
    println!(
        "Found {} patterns with all 4 stars in matched set",
        patterns_with_4_matched
    );

    // Also compute what the ratio would be for our specific quad
    println!("\nOur image quad [51, 376, 306, 272] ratios comparison:");
    println!("  Image ratios (from pixels): [0.4353, 0.4635, 0.4822, 0.5371, 0.7993]");
    println!(
        "  Angular ratios (computed):  [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        ratios[0], ratios[1], ratios[2], ratios[3], ratios[4]
    );
}
