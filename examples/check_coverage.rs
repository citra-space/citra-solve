//! Check if the index covers a specific sky region.

use chameleon::catalog::Index;
use chameleon::core::types::RaDec;
use chameleon::core::math::angular_separation;

fn main() {
    let index_path = "hipparcos.idx";

    // Astrometry.net solution center
    let target_ra = 16.912_f64.to_radians();
    let target_dec = (-7.531_f64).to_radians();
    let target = RaDec::new(target_ra, target_dec);
    let fov_radius = 15.6_f64.to_radians(); // degrees to radians

    println!("Checking index coverage for astrometry.net solution:");
    println!("  Center: RA={:.2}°, Dec={:.2}°", 16.912, -7.531);
    println!("  FOV radius: {:.1}°\n", 15.6);

    let index = Index::open(index_path).expect("Failed to open index");
    println!("Index has {} stars\n", index.num_stars());

    // Count stars within FOV
    let mut stars_in_fov = Vec::new();
    for (idx, star) in index.stars() {
        let pos = star.to_radec();
        let sep = angular_separation(&target, &pos);

        if sep < fov_radius {
            stars_in_fov.push((idx, star.magnitude(), sep.to_degrees()));
        }
    }

    // Sort by magnitude (brightest first)
    stars_in_fov.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    println!("Stars in FOV: {}", stars_in_fov.len());

    if stars_in_fov.is_empty() {
        println!("\n*** NO STARS IN INDEX COVER THIS REGION! ***");
        println!("The index doesn't contain the sky region of your image.");
    } else {
        println!("\nBrightest 20 stars in FOV:");
        for (i, (idx, mag, sep)) in stars_in_fov.iter().take(20).enumerate() {
            println!("  {}: star {} mag={:.1} sep={:.1}°", i+1, idx, mag, sep);
        }
    }

    // Also check where the Chameleon solution pointed
    let cham_ra = 89.47_f64.to_radians();
    let cham_dec = (-15.44_f64).to_radians();
    let cham_target = RaDec::new(cham_ra, cham_dec);

    println!("\n\nChecking Chameleon's (wrong) solution location:");
    println!("  Center: RA={:.2}°, Dec={:.2}°", 89.47, -15.44);

    let mut cham_stars = Vec::new();
    for (idx, star) in index.stars() {
        let pos = star.to_radec();
        let sep = angular_separation(&cham_target, &pos);

        if sep < fov_radius {
            cham_stars.push((idx, star.magnitude(), sep.to_degrees()));
        }
    }

    println!("Stars at Chameleon's solution: {}", cham_stars.len());
}
