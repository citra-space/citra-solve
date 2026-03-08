//! Check star positions in the index to understand pattern generation.

use citra_solve::catalog::Index;

fn main() {
    let index_path = "hipparcos_deep.idx";
    let index = Index::open(index_path).expect("Failed to open index");

    println!("Index: {} stars", index.num_stars());

    // Our matched catalog star indices
    let matched_stars = [51u32, 272, 306, 321, 376, 1075, 1215, 1231, 1427];

    println!("\nStars from our matched set (catalog index -> star data):");
    for &idx in &matched_stars {
        if let Ok(star) = index.get_star(idx) {
            let pos = star.to_radec();
            println!(
                "  Catalog idx {}: mag={:.1}, RA={:.2}°, Dec={:.2}°",
                idx,
                star.magnitude(),
                pos.ra_deg(),
                pos.dec_deg()
            );
        }
    }

    // The index stores stars sorted by magnitude
    // Let's find out the actual position of these stars in the brightness-sorted order
    println!("\nFirst 20 stars in index (sorted by brightness):");
    for (sorted_idx, (orig_idx, star)) in index.stars().enumerate().take(20) {
        println!(
            "  Sorted pos {}: catalog idx {}, mag={:.1}",
            sorted_idx,
            orig_idx,
            star.magnitude()
        );
    }

    println!("\nLooking for our matched stars' positions in brightness order:");
    for (sorted_idx, (orig_idx, star)) in index.stars().enumerate() {
        if matched_stars.contains(&orig_idx) {
            println!(
                "  Catalog idx {} is at sorted position {}, mag={:.1}",
                orig_idx,
                sorted_idx,
                star.magnitude()
            );
        }
    }

    // Check patterns that include star 51
    println!("\nScanning for patterns containing catalog idx 51...");
    let mut count_with_51 = 0;
    for bin in 0..index.num_bins() {
        if let Ok(patterns) = index.get_patterns_in_bin(bin) {
            for pattern in patterns {
                // The star_indices in patterns are u16, which are the sorted positions
                // We need to find which original catalog idx they correspond to
                // Actually, looking at builder.rs, it stores [i, j, k, l] where these
                // are the iteration indices in the sorted list.
                // But when we load the index, index.get_star(idx) uses the sorted position.
                // So pattern.star_indices are the "catalog indices" as stored.

                // Check if any of the pattern's star indices is 51
                let stars: Vec<u16> = pattern.star_indices.iter().cloned().collect();
                if stars.contains(&51) {
                    count_with_51 += 1;
                    if count_with_51 <= 5 {
                        println!("  Pattern with star 51: {:?}", stars);
                    }
                }
            }
        }
    }
    println!(
        "Total patterns containing catalog idx 51: {}",
        count_with_51
    );
}
