//! Tetra3-style canonical quad signature utilities.

/// Compute a permutation- and reflection-invariant 4D quad signature.
///
/// Signature layout: `[x1, y1, x2, y2]` where points are coordinates of the
/// two non-baseline stars in a normalized frame whose baseline is the longest
/// edge and has length 1.0.
pub fn canonical_tetra_signature(points: &[(f64, f64); 4]) -> Option<[f64; 4]> {
    let mut max_dist2 = 0.0f64;
    let mut baseline_pairs: Vec<(usize, usize)> = Vec::new();
    const EPS: f64 = 1e-10;

    for i in 0..4 {
        for j in (i + 1)..4 {
            let dx = points[j].0 - points[i].0;
            let dy = points[j].1 - points[i].1;
            let d2 = dx * dx + dy * dy;
            if d2 > max_dist2 + EPS {
                max_dist2 = d2;
                baseline_pairs.clear();
                baseline_pairs.push((i, j));
            } else if (d2 - max_dist2).abs() <= EPS {
                baseline_pairs.push((i, j));
            }
        }
    }

    if max_dist2 <= 1e-12 || baseline_pairs.is_empty() {
        return None;
    }

    let mut best: Option<[f64; 4]> = None;
    for (a, b) in baseline_pairs {
        let mut remain = [0usize; 2];
        let mut r = 0usize;
        for idx in 0..4 {
            if idx != a && idx != b {
                remain[r] = idx;
                r += 1;
            }
        }

        if let Some(sig) = signature_for_order(points, a, b, remain[0], remain[1]) {
            maybe_update_best(&mut best, sig);
        }
        if let Some(sig) = signature_for_order(points, b, a, remain[0], remain[1]) {
            maybe_update_best(&mut best, sig);
        }
    }

    best
}

fn maybe_update_best(best: &mut Option<[f64; 4]>, candidate: [f64; 4]) {
    match best {
        Some(cur) => {
            if lex_lt(&candidate, cur) {
                *cur = candidate;
            }
        }
        None => *best = Some(candidate),
    }
}

fn lex_lt(a: &[f64; 4], b: &[f64; 4]) -> bool {
    for i in 0..4 {
        let da = a[i];
        let db = b[i];
        if (da - db).abs() <= 1e-12 {
            continue;
        }
        return da < db;
    }
    false
}

fn signature_for_order(
    points: &[(f64, f64); 4],
    a: usize,
    b: usize,
    c: usize,
    d: usize,
) -> Option<[f64; 4]> {
    let ax = points[a].0;
    let ay = points[a].1;
    let vx = points[b].0 - ax;
    let vy = points[b].1 - ay;
    let len2 = vx * vx + vy * vy;
    if len2 <= 1e-12 {
        return None;
    }

    let mut p1 = project(points[c], (ax, ay), (vx, vy), len2);
    let mut p2 = project(points[d], (ax, ay), (vx, vy), len2);

    // Deterministic order for the two interior points.
    if p2.x < p1.x || ((p2.x - p1.x).abs() < 1e-12 && p2.y < p1.y) {
        std::mem::swap(&mut p1, &mut p2);
    }

    // Canonical mirror handling.
    if p1.y < 0.0 || (p1.y.abs() < 1e-12 && p2.y < 0.0) {
        p1.y = -p1.y;
        p2.y = -p2.y;
    }

    let sig = [p1.x, p1.y, p2.x, p2.y];
    if sig.iter().all(|v| v.is_finite() && v.abs() <= 4.0) {
        Some(sig)
    } else {
        None
    }
}

#[derive(Clone, Copy)]
struct XY {
    x: f64,
    y: f64,
}

fn project(p: (f64, f64), a: (f64, f64), v: (f64, f64), len2: f64) -> XY {
    let px = p.0 - a.0;
    let py = p.1 - a.1;
    let x = (px * v.0 + py * v.1) / len2;
    let y = (px * v.1 - py * v.0) / len2;
    XY { x, y }
}
