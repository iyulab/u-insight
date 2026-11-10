//! Exact nearest neighbours of the rows of a data set, through a k-d tree.
//!
//! The neighbour searches of the density methods (LOF's k-distance and
//! k-neighbourhood) compared every point with every other one and sorted the
//! lot: `O(n² log n)`. A k-d tree (Bentley 1975; Friedman, Bentley & Finkel
//! 1977) answers the same questions by visiting only the cells that can hold a
//! closer point, about `O(log n)` cells a query in few dimensions — it degrades
//! towards a full scan as the dimension grows, never below one.
//!
//! The answers are *exactly* the brute-force ones: the same distances
//! (`sqrt` of [`crate::distance::squared_euclidean`] of the same pair, bit for
//! bit), the same neighbours, ordered by distance and then by index. A cell is
//! skipped only when the splits that bound it put it farther than the current
//! bound by more than rounding, so no point a full scan would keep is lost.
//!
//! The rows are also copied once into one array in the order of the leaves, so
//! a leaf is read from consecutive memory, and a row's distance is given up as
//! soon as the coordinates seen so far put it past the bound it is compared
//! with ([`crate::distance::squared_euclidean_within`]): in many dimensions,
//! where the tree visits most rows, that is where the time goes.

use crate::distance::squared_euclidean_within;

/// Rows at or below this count stay in one leaf, scanned directly.
const LEAF: usize = 16;

/// Margin on a pruning bound: the distance to a splitting plane is one
/// coordinate's difference, the distance to a point is a rounded square root
/// of a rounded sum — the latter may fall below the former by an ulp or two.
const SLACK: f64 = 1e-9;

enum Node {
    Leaf {
        lo: usize,
        hi: usize,
    },
    Split {
        dim: usize,
        at: f64,
        left: usize,
        right: usize,
    },
}

/// A k-d tree over the rows of `data` (every row the same length, finite).
pub(crate) struct KdTree<'a> {
    data: &'a [Vec<f64>],
    order: Vec<usize>,
    nodes: Vec<Node>,
    dims: usize,
    /// The rows in the order of `order`, `dims` numbers each.
    rows: Vec<f64>,
}

impl<'a> KdTree<'a> {
    pub(crate) fn new(data: &'a [Vec<f64>]) -> Self {
        let mut tree = KdTree {
            data,
            order: (0..data.len()).collect(),
            nodes: Vec::new(),
            dims: data.first().map_or(0, Vec::len),
            rows: Vec::new(),
        };
        if !data.is_empty() {
            tree.build(0, data.len());
            tree.rows = tree
                .order
                .iter()
                .flat_map(|&i| data[i].iter().copied())
                .collect();
        }
        tree
    }

    /// Builds the subtree over `order[lo..hi]`, returning its node index.
    fn build(&mut self, lo: usize, hi: usize) -> usize {
        let dims = self.data[self.order[lo]].len();
        if hi - lo <= LEAF || dims == 0 {
            self.nodes.push(Node::Leaf { lo, hi });
            return self.nodes.len() - 1;
        }
        // Split the widest coordinate at its median.
        let data = self.data;
        let spread = |d: usize, order: &[usize]| {
            let (mn, mx) = order
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &i| {
                    (mn.min(data[i][d]), mx.max(data[i][d]))
                });
            mx - mn
        };
        let slice = &self.order[lo..hi];
        let dim = (0..dims)
            .max_by(|&a, &b| spread(a, slice).total_cmp(&spread(b, slice)))
            .expect("a row has at least one coordinate");
        if spread(dim, slice) == 0.0 {
            // Every row in this cell is the same point.
            self.nodes.push(Node::Leaf { lo, hi });
            return self.nodes.len() - 1;
        }
        let mid = lo + (hi - lo) / 2;
        self.order[lo..hi]
            .select_nth_unstable_by(mid - lo, |&a, &b| data[a][dim].total_cmp(&data[b][dim]));
        let at = data[self.order[mid]][dim];
        let slot = self.nodes.len();
        self.nodes.push(Node::Leaf { lo, hi }); // placeholder
        let left = self.build(lo, mid);
        let right = self.build(mid, hi);
        self.nodes[slot] = Node::Split {
            dim,
            at,
            left,
            right,
        };
        slot
    }

    /// The distance from row `i` to its `k`-th nearest other row
    /// (`1 <= k < n`) — what sorting every other row by distance and taking the
    /// `k`-th would give.
    pub(crate) fn kth_distance(&self, i: usize, k: usize) -> f64 {
        self.search(i, k, None).0
    }

    /// The `k`-th nearest distance of row `i` (`1 <= k < n`) and every other
    /// row within `tie` (a relative margin) of it, as `(index, distance)`
    /// ordered by distance and then by index — the prefix a full sort keeps.
    ///
    /// One descent answers both: a row is kept when it is within the margin of
    /// the `k`-th distance *as it stood when the row was met*, which is never
    /// smaller than the final one, so every row of the final neighbourhood is
    /// among those kept and the rest are dropped at the end.
    pub(crate) fn neighbourhood(&self, i: usize, k: usize, tie: f64) -> (f64, Vec<(usize, f64)>) {
        self.search(i, k, Some(tie))
    }

    /// The descent behind [`Self::kth_distance`] (`tie` `None`: the neighbours
    /// are not collected) and [`Self::neighbourhood`].
    fn search(&self, i: usize, k: usize, tie: Option<f64>) -> (f64, Vec<(usize, f64)>) {
        // The k smallest (distance, index) pairs seen so far, largest first.
        let mut best: std::collections::BinaryHeap<Pair> =
            std::collections::BinaryHeap::with_capacity(k + 1);
        let mut kept: Vec<(usize, f64)> = Vec::new();
        let mut offsets = vec![0.0_f64; self.data[i].len()];
        self.knn(
            &mut Descent {
                i,
                k,
                tie,
                query: &self.data[i],
                offsets: &mut offsets,
                best: &mut best,
                kept: &mut kept,
            },
            0,
            0.0,
        );
        let kth = best.peek().expect("k >= 1 rows besides i").0;
        let reach = kth * (1.0 + tie.unwrap_or(0.0));
        kept.retain(|&(_, d)| d <= reach);
        kept.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        (kth, kept)
    }

    /// `rd` is the squared distance from row `i` to the cell of `node`, as far
    /// as the splits above it show (Arya & Mount 1993): one offset per
    /// coordinate, so a cell is bounded by every split made on the way down,
    /// not only the last — in many dimensions the difference decides how much
    /// of the tree is skipped.
    fn knn(&self, st: &mut Descent<'_>, node: usize, rd: f64) {
        match self.nodes[node] {
            Node::Leaf { lo, hi } => {
                // Past `reach` a row is neither among the k nearest nor within the
                // margin of the k-th, so its distance need not be finished; `SLACK`
                // keeps a row at the boundary on the side of finishing it. Only a row
                // that enters the k nearest moves it.
                let (mut reach, mut limit_squared) = st.limits();
                for p in lo..hi {
                    let Some(squared) = squared_euclidean_within(
                        st.query,
                        &self.rows[p * self.dims..(p + 1) * self.dims],
                        limit_squared,
                    ) else {
                        continue;
                    };
                    let j = self.order[p];
                    if j == st.i {
                        continue;
                    }
                    let pair = Pair(squared.sqrt(), j);
                    if st.tie.is_some() && pair.0 <= reach {
                        st.kept.push((j, pair.0));
                    }
                    if st.best.len() < st.k {
                        st.best.push(pair);
                    } else if pair < *st.best.peek().expect("k entries") {
                        st.best.pop();
                        st.best.push(pair);
                    } else {
                        continue;
                    }
                    (reach, limit_squared) = st.limits();
                }
            }
            Node::Split {
                dim,
                at,
                left,
                right,
            } => {
                let gap = self.data[st.i][dim] - at;
                let (near, far) = if gap < 0.0 {
                    (left, right)
                } else {
                    (right, left)
                };
                self.knn(st, near, rd);
                let bound = if st.best.len() < st.k {
                    f64::INFINITY
                } else {
                    st.best.peek().expect("k entries").0
                };
                let before = st.offsets[dim];
                let after = gap.abs().max(before);
                let far_rd = rd - before * before + after * after;
                let limit = bound * (1.0 + SLACK);
                if far_rd <= limit * limit {
                    st.offsets[dim] = after;
                    self.knn(st, far, far_rd);
                    st.offsets[dim] = before;
                }
            }
        }
    }

    /// Every other row within `radius` of row `i`, as `(index, distance)`,
    /// ordered by distance and then by index — the prefix a full sort keeps.
    pub(crate) fn within(&self, i: usize, radius: f64) -> Vec<(usize, f64)> {
        let mut out = Vec::new();
        let mut offsets = vec![0.0_f64; self.data[i].len()];
        self.range(i, radius, 0, 0.0, &mut offsets, &mut out);
        out.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        out
    }

    fn range(
        &self,
        i: usize,
        radius: f64,
        node: usize,
        rd: f64,
        offsets: &mut [f64],
        out: &mut Vec<(usize, f64)>,
    ) {
        match self.nodes[node] {
            Node::Leaf { lo, hi } => {
                let limit = radius * (1.0 + SLACK);
                for p in lo..hi {
                    let j = self.order[p];
                    if j == i {
                        continue;
                    }
                    if let Some(squared) = squared_euclidean_within(
                        &self.data[i],
                        &self.rows[p * self.dims..(p + 1) * self.dims],
                        limit * limit,
                    ) {
                        let d = squared.sqrt();
                        if d <= radius {
                            out.push((j, d));
                        }
                    }
                }
            }
            Node::Split {
                dim,
                at,
                left,
                right,
            } => {
                let gap = self.data[i][dim] - at;
                let (near, far) = if gap < 0.0 {
                    (left, right)
                } else {
                    (right, left)
                };
                self.range(i, radius, near, rd, offsets, out);
                let before = offsets[dim];
                let after = gap.abs().max(before);
                let far_rd = rd - before * before + after * after;
                let limit = radius * (1.0 + SLACK);
                if far_rd <= limit * limit {
                    offsets[dim] = after;
                    self.range(i, radius, far, far_rd, offsets, out);
                    offsets[dim] = before;
                }
            }
        }
    }
}

/// The state of one nearest-neighbour descent.
struct Descent<'s> {
    i: usize,
    k: usize,
    /// The relative margin of the neighbours to keep, `None` to keep none.
    tie: Option<f64>,
    /// The coordinates of row `i`.
    query: &'s [f64],
    /// Per coordinate, how far row `i` lies outside the current cell.
    offsets: &'s mut [f64],
    best: &'s mut std::collections::BinaryHeap<Pair>,
    kept: &'s mut Vec<(usize, f64)>,
}

impl Descent<'_> {
    /// The farthest a row can be and still matter -- the k-th distance so far with
    /// the tie margin -- and the square of that with the rounding slack, the bound a
    /// distance is given up at. Infinite until k rows have been met.
    fn limits(&self) -> (f64, f64) {
        if self.best.len() < self.k {
            return (f64::INFINITY, f64::INFINITY);
        }
        let reach = self.best.peek().expect("k entries").0 * (1.0 + self.tie.unwrap_or(0.0));
        let limit = reach * (1.0 + SLACK);
        (reach, limit * limit)
    }
}

/// A (distance, index) pair ordered by distance, then by index — the order a
/// stable sort by distance of indices taken in ascending order produces.
#[derive(PartialEq)]
struct Pair(f64, usize);

impl Eq for Pair {}

impl PartialOrd for Pair {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Pair {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn euclid(data: &[Vec<f64>]) -> impl Fn(usize, usize) -> f64 + '_ {
        move |i, j| crate::distance::squared_euclidean(&data[i], &data[j]).sqrt()
    }

    /// Deterministic points: a few clusters, duplicates and a grid (ties).
    fn points(n: usize, dims: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64)
        };
        (0..n)
            .map(|r| match r % 5 {
                0 => vec![(r % 7) as f64; dims], // exact duplicates and grid ties
                1 => (0..dims).map(|d| (d as f64) * 3.0 + next()).collect(),
                _ => (0..dims).map(|_| next() * 10.0).collect(),
            })
            .collect()
    }

    fn brute(data: &[Vec<f64>], i: usize) -> Vec<(usize, f64)> {
        let d = euclid(data);
        let mut all: Vec<(usize, f64)> = (0..data.len())
            .filter(|&j| j != i)
            .map(|j| (j, d(i, j)))
            .collect();
        all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        all
    }

    #[test]
    fn the_tree_answers_exactly_what_a_full_sort_does() {
        for (n, dims, seed) in [
            (1usize, 2usize, 1u64),
            (2, 1, 2),
            (40, 1, 3),
            (300, 2, 4),
            (300, 4, 5),
            (200, 9, 6),
            (400, 12, 7),
            (400, 16, 8),
        ] {
            let data = points(n, dims, seed);
            let tree = KdTree::new(&data);
            for i in 0..n {
                let sorted = brute(&data, i);
                for k in [1, 3, 20] {
                    if k >= n {
                        continue;
                    }
                    let kth = tree.kth_distance(i, k);
                    assert_eq!(
                        kth.to_bits(),
                        sorted[k - 1].1.to_bits(),
                        "n {n} dims {dims} row {i} k {k}"
                    );
                    let radius = kth * (1.0 + 1e-12);
                    let expected: Vec<(usize, f64)> = sorted
                        .iter()
                        .copied()
                        .take_while(|&(_, d)| d <= radius)
                        .collect();
                    assert_eq!(
                        tree.within(i, radius),
                        expected,
                        "n {n} dims {dims} row {i} k {k}"
                    );
                    // One descent gives the same distance and the neighbours a sort keeps.
                    let (got_kth, got) = tree.neighbourhood(i, k, 1e-12);
                    assert_eq!(got_kth.to_bits(), kth.to_bits());
                    assert_eq!(got, expected, "n {n} dims {dims} row {i} k {k}");
                }
            }
        }
    }
}
