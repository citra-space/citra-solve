//! Find our specific target quad in the index.

use citra_solve::catalog::Index;

fn main() {
    let index_path = "hipparcos_deep.idx";
    let index = Index::open(index_path).expect("Failed to open index");

    println!("Index: {} patterns\n", index.num_patterns());

    // Target quad: stars 51, 272, 306, 376
    let target = [51u16, 272, 306, 376];

    // Search for exact match
    println!("Searching for quad [51, 272, 306, 376]...");
    let mut found = false;

    for bin in 0..index.num_bins() {
        if let Ok(patterns) = index.get_patterns_in_bin(bin) {
            for pattern in patterns {
                let mut stars = pattern.star_indices;
                stars.sort();
                if stars == target {
                    let r = pattern.ratios();
                    println!("FOUND! Bin {}, ratios: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                        bin, r[0], r[1], r[2], r[3], r[4]);
                    found = true;
                }
            }
        }
    }

    if !found {
        println!("Quad [51, 272, 306, 376] NOT FOUND in index!");
    }

    // Also search for quads containing all 4
    println!("\nSearching for any quad containing all of 51, 272, 306, 376...");
    let target_set: std::collections::HashSet<u16> = target.iter().cloned().collect();
    let mut count = 0;

    for bin in 0..index.num_bins() {
        if let Ok(patterns) = index.get_patterns_in_bin(bin) {
            for pattern in patterns {
                let stars_set: std::collections::HashSet<u16> =
                    pattern.star_indices.iter().cloned().collect();
                let overlap = stars_set.intersection(&target_set).count();
                if overlap == 4 {
                    count += 1;
                    if count <= 5 {
                        let r = pattern.ratios();
                        println!("  {:?} ratios: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                            pattern.star_indices, r[0], r[1], r[2], r[3], r[4]);
                    }
                }
            }
        }
    }

    println!("\nTotal quads with all 4 target stars: {}", count);

    // Also show expected ratios
    println!("\nExpected ratios:");
    println!("  Image ratios:   [0.4353, 0.4635, 0.4822, 0.5371, 0.7993]");
    println!("  Angular ratios: [0.4346, 0.4640, 0.4786, 0.5365, 0.7975]");
}
