//! Exact nearest neighbours of the rows of a data set, through a k-d tree.
//!
//! The neighbour searches of the density methods (LOF's k-distance and
//! k-neighbourhood) compared every point with every other one and sorted the
//! lot: `O(n² log n)`. A k-d tree (Bentley 1975; Friedman, Bentley & Finkel
//! 1977) answers the same questions by visiting only the cells that can hold a
//! closer point, about `O(log n)` cells a query in few dimensions — it degrades
//! towards a full scan as the dimension grows, never below one.
//!
//! The answers are *exactly* the brute-force ones: the same distances (the
//! caller's distance function, applied to the same pair), the same neighbours,
//! ordered by distance and then by index. A cell is skipped only when its
//! splitting plane is farther than the current bound by more than rounding,
//! so no point a full scan would keep is lost.

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
}

impl<'a> KdTree<'a> {
    pub(crate) fn new(data: &'a [Vec<f64>]) -> Self {
        let mut tree = KdTree {
            data,
            order: (0..data.len()).collect(),
            nodes: Vec::new(),
        };
        if !data.is_empty() {
            tree.build(0, data.len());
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
    /// (`1 <= k < n`), as `dist` measures it — what sorting every other row by
    /// distance and taking the `k`-th would give.
    pub(crate) fn kth_distance(
        &self,
        i: usize,
        k: usize,
        dist: impl Fn(usize, usize) -> f64,
    ) -> f64 {
        // The k smallest (distance, index) pairs seen so far, largest first.
        let mut best: std::collections::BinaryHeap<Pair> =
            std::collections::BinaryHeap::with_capacity(k + 1);
        self.knn(0, i, k, &dist, &mut best);
        best.peek().expect("k >= 1 rows besides i").0
    }

    fn knn(
        &self,
        node: usize,
        i: usize,
        k: usize,
        dist: &impl Fn(usize, usize) -> f64,
        best: &mut std::collections::BinaryHeap<Pair>,
    ) {
        match self.nodes[node] {
            Node::Leaf { lo, hi } => {
                for &j in &self.order[lo..hi] {
                    if j == i {
                        continue;
                    }
                    let pair = Pair(dist(i, j), j);
                    if best.len() < k {
                        best.push(pair);
                    } else if pair < *best.peek().expect("k entries") {
                        best.pop();
                        best.push(pair);
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
                self.knn(near, i, k, dist, best);
                let bound = if best.len() < k {
                    f64::INFINITY
                } else {
                    best.peek().expect("k entries").0
                };
                if gap.abs() <= bound * (1.0 + SLACK) {
                    self.knn(far, i, k, dist, best);
                }
            }
        }
    }

    /// Every other row within `radius` of row `i`, as `(index, distance)`,
    /// ordered by distance and then by index — the prefix a full sort keeps.
    pub(crate) fn within(
        &self,
        i: usize,
        radius: f64,
        dist: impl Fn(usize, usize) -> f64,
    ) -> Vec<(usize, f64)> {
        let mut out = Vec::new();
        self.range(0, i, radius, &dist, &mut out);
        out.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        out
    }

    fn range(
        &self,
        node: usize,
        i: usize,
        radius: f64,
        dist: &impl Fn(usize, usize) -> f64,
        out: &mut Vec<(usize, f64)>,
    ) {
        match self.nodes[node] {
            Node::Leaf { lo, hi } => {
                for &j in &self.order[lo..hi] {
                    if j != i {
                        let d = dist(i, j);
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
                self.range(near, i, radius, dist, out);
                if gap.abs() <= radius * (1.0 + SLACK) {
                    self.range(far, i, radius, dist, out);
                }
            }
        }
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
        move |i, j| {
            data[i]
                .iter()
                .zip(&data[j])
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt()
        }
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
        ] {
            let data = points(n, dims, seed);
            let tree = KdTree::new(&data);
            for i in 0..n {
                let sorted = brute(&data, i);
                for k in [1, 3, 20] {
                    if k >= n {
                        continue;
                    }
                    let kth = tree.kth_distance(i, k, euclid(&data));
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
                        tree.within(i, radius, euclid(&data)),
                        expected,
                        "n {n} dims {dims} row {i} k {k}"
                    );
                }
            }
        }
    }
}
