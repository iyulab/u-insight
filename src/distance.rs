//! The squared Euclidean distance, the inner loop of every distance-based
//! method here (k-means, DBSCAN, HDBSCAN, hierarchical, silhouette, LOF).
//!
//! A single running sum cannot be vectorised: floating-point addition is not
//! associative, so the compiler must add the squares one after the other, each
//! waiting for the one before (about 1.3 ns a coordinate -- 100 s for the
//! `n²·d` of 20,000 rows by 200 columns). Four independent partial sums let it
//! use the vector unit and keep the adder busy, several times faster, and the
//! order of the additions is fixed, so the result is the same on every run.

/// `Σ (aᵢ − bᵢ)²` over the shorter of the two slices.
#[inline]
pub(crate) fn squared_euclidean(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    let (a, b) = (&a[..len], &b[..len]);
    let (a4, b4) = (a.chunks_exact(4), b.chunks_exact(4));
    let (a_tail, b_tail) = (a4.remainder(), b4.remainder());
    let mut lanes = [0.0_f64; 4];
    for (x, y) in a4.zip(b4) {
        for ((lane, &p), &q) in lanes.iter_mut().zip(x).zip(y) {
            let diff = p - q;
            *lane += diff * diff;
        }
    }
    let mut tail = 0.0;
    for (&p, &q) in a_tail.iter().zip(b_tail) {
        let diff = p - q;
        tail += diff * diff;
    }
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]) + tail
}

/// [`squared_euclidean`] that gives up once the running sum exceeds `limit`:
/// `None` then, and otherwise `Some` of exactly the bits `squared_euclidean`
/// returns -- the same four partial sums added in the same order, the check
/// between blocks of four coordinates only reads them. Every term is
/// non-negative, so the sum so far is a lower bound of the sum, and a point
/// already farther than the bound it is being compared with need not be
/// finished (about half of the coordinates of a typical far point in 16
/// dimensions).
#[inline]
pub(crate) fn squared_euclidean_within(a: &[f64], b: &[f64], limit: f64) -> Option<f64> {
    let len = a.len().min(b.len());
    let (a, b) = (&a[..len], &b[..len]);
    let (a4, b4) = (a.chunks_exact(4), b.chunks_exact(4));
    let (a_tail, b_tail) = (a4.remainder(), b4.remainder());
    let mut lanes = [0.0_f64; 4];
    for (block, (x, y)) in a4.zip(b4).enumerate() {
        for ((lane, &p), &q) in lanes.iter_mut().zip(x).zip(y) {
            let diff = p - q;
            *lane += diff * diff;
        }
        // The check costs a branch: every eighth coordinate is often enough.
        if block % 2 == 1 && (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]) > limit {
            return None;
        }
    }
    let mut tail = 0.0;
    for (&p, &q) in a_tail.iter().zip(b_tail) {
        let diff = p - q;
        tail += diff * diff;
    }
    Some((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]) + tail)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    /// The sum matches the one-by-one sum to rounding for every length around
    /// the lane width (the tail) and a long one, and is exact on integers.
    #[test]
    fn matches_the_running_sum_at_every_length() {
        let mut state = 7_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1_u64 << 53) as f64 * 20.0 - 10.0
        };
        for len in (0..=40).chain([199, 200, 201, 1000]) {
            let a: Vec<f64> = (0..len).map(|_| next()).collect();
            let b: Vec<f64> = (0..len).map(|_| next()).collect();
            let (got, want) = (squared_euclidean(&a, &b), naive(&a, &b));
            assert!(
                (got - want).abs() <= 1e-13 * want.max(1.0),
                "len {len}: {got} vs {want}"
            );
        }
        let a: Vec<f64> = (0..37).map(f64::from).collect();
        let b: Vec<f64> = (0..37).map(|i| f64::from(i) + 2.0).collect();
        assert_eq!(squared_euclidean(&a, &b), 4.0 * 37.0);
        assert_eq!(squared_euclidean(&[], &[]), 0.0);
    }

    /// A distance that is not given up is the bits of the full one, at every length
    /// around the block width, and one past the limit is given up -- from a limit
    /// just below the true sum to zero.
    #[test]
    fn the_bounded_sum_is_the_full_sum_or_nothing() {
        let mut state = 11_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1_u64 << 53) as f64 * 8.0 - 4.0
        };
        for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 12, 16, 17, 33, 64] {
            let a: Vec<f64> = (0..len).map(|_| next()).collect();
            let b: Vec<f64> = (0..len).map(|_| next()).collect();
            let full = squared_euclidean(&a, &b);
            for limit in [f64::INFINITY, full * (1.0 + 1e-9), full] {
                let got = squared_euclidean_within(&a, &b, limit);
                assert_eq!(got.map(f64::to_bits), Some(full.to_bits()), "len {len}");
            }
            if full > 0.0 && len >= 8 {
                // A sum that has passed the limit by the first block is given up; one
                // that passes only at the very end is not (it finishes and is `Some`).
                assert_eq!(squared_euclidean_within(&a, &b, 0.0), None, "len {len}");
            }
        }
    }
}
