//! How the density methods' time grows with the rows (run with
//! `--release -- --ignored --nocapture`; `ROWS=12500,25000` picks the sizes).

use u_insight::clustering::{hdbscan, HdbscanConfig};
use u_insight::lof::{lof, LofConfig};

fn rows(default: &[usize]) -> Vec<usize> {
    std::env::var("ROWS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| default.to_vec())
}

/// `n` rows of 4 independent standard normals, the same every run.
fn normals(n: usize) -> Vec<Vec<f64>> {
    let mut s = 5_u64;
    let mut u = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
    };
    (0..n)
        .map(|_| {
            (0..4)
                .map(|_| {
                    let (a, b) = (u(), u());
                    (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
                })
                .collect()
        })
        .collect()
}

#[test]
#[ignore = "timing, not a check"]
fn lof_time_by_rows() {
    for n in rows(&[12_500, 25_000, 50_000, 100_000]) {
        let data = normals(n);
        let t = std::time::Instant::now();
        let r = lof(&data, &LofConfig::default()).expect("valid");
        println!(
            "lof n {n}: {:.2} s ({} anomalies)",
            t.elapsed().as_secs_f64(),
            r.anomaly_count
        );
    }
}

#[test]
#[ignore = "timing, not a check"]
fn hdbscan_time_by_rows() {
    for n in rows(&[5_000, 10_000, 20_000]) {
        let data = normals(n);
        let t = std::time::Instant::now();
        let r = hdbscan(&data, &HdbscanConfig::new(5)).expect("valid");
        println!(
            "hdbscan n {n}: {:.2} s ({} clusters)",
            t.elapsed().as_secs_f64(),
            r.n_clusters
        );
    }
}
