//! How LOF's time grows with the rows (run with `--ignored --nocapture`, in release).

use u_insight::lof::{lof, LofConfig};

#[test]
#[ignore = "timing, not a check"]
fn lof_time_by_rows() {
    let mut s = 5_u64;
    let mut gauss = || {
        let mut u = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
        };
        let (a, b) = (u(), u());
        (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
    };
    // LOF_ROWS=12500,25000 to time other sizes.
    let rows: Vec<usize> = std::env::var("LOF_ROWS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![12_500, 25_000, 50_000, 100_000]);
    for n in rows {
        let data: Vec<Vec<f64>> = (0..n).map(|_| (0..4).map(|_| gauss()).collect()).collect();
        let t = std::time::Instant::now();
        let r = lof(&data, &LofConfig::default()).expect("valid");
        println!(
            "n {n}: {:.2} s ({} anomalies)",
            t.elapsed().as_secs_f64(),
            r.anomaly_count
        );
    }
}
