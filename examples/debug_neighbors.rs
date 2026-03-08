//! Debug neighbor relationships between our target stars.

use citra_solve::catalog::Index;
use citra_solve::core::math::angular_separation;
use citra_solve::core::types::RaDec;

fn main() {
    let index_path = "hipparcos_deep.idx";
    let index = Index::open(index_path).expect("Failed to open index");

    let (fov_min, fov_max) = index.fov_range_deg();
    println!("Index FOV range: {:.1}° - {:.1}°", fov_min, fov_max);
    println!(
        "Neighbor threshold used in build: {:.1}° (fov_max * 1.2)\n",
        fov_max as f64 * 1.2
    );

    // Our target stars
    let targets = [51u32, 272, 306, 376];

    // Get positions
    let mut positions: Vec<(u32, RaDec)> = Vec::new();
    for &idx in &targets {
        let star = index.get_star(idx).unwrap();
        positions.push((idx, star.to_radec()));
    }

    // Compute all pairwise distances
    println!("Angular distances between target stars:");
    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            let sep = angular_separation(&positions[i].1, &positions[j].1).to_degrees();
            let within = if sep <= fov_max as f64 * 1.2 {
                "YES"
            } else {
                "NO"
            };
            println!(
                "  {} - {}: {:.2}° (within neighbor threshold: {})",
                positions[i].0, positions[j].0, sep, within
            );
        }
    }
    println!();

    // For star 51, count how many stars are within the neighbor threshold
    let star51 = index.get_star(51).unwrap();
    let pos51 = star51.to_radec();
    let threshold_rad = (fov_max as f64 * 1.2).to_radians();

    let mut neighbors_of_51: Vec<u32> = Vec::new();
    for (idx, star) in index.stars() {
        if idx == 51 {
            continue;
        }
        let pos = star.to_radec();
        let sep = angular_separation(&pos51, &pos);
        if sep <= threshold_rad {
            neighbors_of_51.push(idx);
        }
    }

    println!(
        "Star 51 has {} neighbors within {:.1}°",
        neighbors_of_51.len(),
        fov_max as f64 * 1.2
    );
    println!(
        "  First 20: {:?}",
        &neighbors_of_51[..20.min(neighbors_of_51.len())]
    );

    // Check if our target stars are in the neighbors
    println!("\n  Are target stars in neighbor list?");
    for &target in &[272u32, 306, 376] {
        let found = neighbors_of_51.contains(&target);
        println!("    Star {}: {}", target, if found { "YES" } else { "NO" });
    }

    // Check patterns containing star 51
    println!("\nPatterns containing star 51:");
    let mut count = 0;
    let mut with_targets = 0;
    for bin in 0..index.num_bins() {
        if let Ok(patterns) = index.get_patterns_in_bin(bin) {
            for pattern in patterns {
                let stars: Vec<u32> = pattern.star_indices.iter().map(|&s| s as u32).collect();
                if stars.contains(&51) {
                    count += 1;
                    let has_target = stars.iter().any(|s| [272u32, 306, 376].contains(s));
                    if has_target {
                        with_targets += 1;
                        println!("  {:?}", stars);
                    }
                }
            }
        }
    }
    println!("\nTotal patterns with star 51: {}", count);
    println!(
        "Patterns with 51 AND one of [272, 306, 376]: {}",
        with_targets
    );
}
