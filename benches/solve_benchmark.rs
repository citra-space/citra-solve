//! Criterion benchmarks for the plate solver.

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

use chameleon::core::types::{RaDec, DetectedStar};
use chameleon::pattern::{generate_quads, compute_hash};
use chameleon::wcs::Wcs;

fn bench_quad_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("quad_generation");

    // Generate test stars
    for num_stars in [10, 20, 30, 50].iter() {
        let stars: Vec<DetectedStar> = (0..*num_stars)
            .map(|i| {
                let angle = i as f64 * 0.1;
                DetectedStar::new(
                    512.0 + 200.0 * angle.cos(),
                    512.0 + 200.0 * angle.sin(),
                    1000.0 - i as f64 * 10.0,
                )
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(num_stars),
            &stars,
            |b, stars| {
                b.iter(|| generate_quads(stars, 30, 100));
            },
        );
    }

    group.finish();
}

fn bench_hash_computation(c: &mut Criterion) {
    let ratios = [0.1, 0.25, 0.5, 0.75, 0.9];

    c.bench_function("compute_hash", |b| {
        b.iter(|| compute_hash(&ratios, 1_000_000));
    });
}

fn bench_wcs_transform(c: &mut Criterion) {
    let wcs = Wcs::new(
        (512.0, 512.0),
        RaDec::from_degrees(180.0, 45.0),
        [[-0.001, 0.0], [0.0, 0.001]],
    );

    let mut group = c.benchmark_group("wcs_transform");

    group.bench_function("pixel_to_sky", |b| {
        b.iter(|| wcs.pixel_to_sky(100.0, 200.0));
    });

    group.bench_function("sky_to_pixel", |b| {
        let radec = RaDec::from_degrees(179.5, 45.2);
        b.iter(|| wcs.sky_to_pixel(&radec));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_quad_generation,
    bench_hash_computation,
    bench_wcs_transform,
);

criterion_main!(benches);
