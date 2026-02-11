//! Disjoint Set Union (Union-Find) for connectivity.
//!
//! Overview
//! --------
//! This module provides a minimal **Disjoint Set Union (DSU)** data structure,
//! also known as **Union-Find**, used to maintain a partition of `n` elements
//! into disjoint sets under two operations:
//!
//! - `find(x)`: return the representative (root) of the set containing `x`,
//! - `union(a, b)`: merge the sets containing `a` and `b`.
//!
//! The implementation uses two classic optimizations:
//! - **path compression** in `find()`,
//! - **union by size** in `union()`.
//!
//! These optimizations make the amortized cost per operation effectively constant
//! for practical input sizes (more precisely: inverse Ackermann).
//!
//! Intended usage
//! --------------
//! This DSU is used to compute **connected components** in an undirected view of a graph.
//! Given a set of edges `(u, v)` (direction ignored), connectivity can be computed by:
//!
//! 1) initializing `UnionFind::new(n_nodes)`,
//! 2) calling `union(u, v)` for each edge,
//! 3) grouping nodes by their root `find(u)`.
//!
//! Indexing model
//! --------------
//! Elements are identified by `usize` indices in `0..n-1`. The DSU does not store
//! external keys; it only stores parent pointers and sizes.
//!
//! Complexity
//! ----------
//! With path compression and union by size:
//! - `find` and `union` are amortized `O(α(n))`, where `α` is inverse Ackermann,
//!   which is < 5 for any realistic `n`.
//! - memory is `O(n)`.
//!
//! Notes
//! -----
//! - This DSU is designed for **undirected connectivity**.
//! - If a directed graph is given, edges should be treated as undirected when calling
//!   `union(u, v)`.

/// Disjoint Set Union (Union-Find) with path compression and union by size.
///
/// The DSU maintains a forest of rooted trees:
/// - each element points to a parent,
/// - roots are representatives of disjoint sets,
/// - `size[root]` stores the size of the set for union-by-size.
///
/// Notes
/// -----
/// - Indices are `usize` and refer to an external dense node indexing.
/// - This structure is intended for undirected connectivity computations.
#[derive(Debug, Clone)]
pub struct UnionFind {
    /// Parent pointer for each element.
    ///
    /// Invariant
    /// ---------
    /// - If `parent[x] == x`, then `x` is a root.
    /// - Otherwise, `parent[x]` is another index closer to the root.
    parent: Vec<usize>,

    /// Size of the set for each root.
    ///
    /// Invariant
    /// ---------
    /// - Only guaranteed meaningful for roots.
    /// - After `union`, the new root size is updated as `size[root] += size[child_root]`.
    size: Vec<u32>,
}

impl UnionFind {
    /// Create a DSU over `n` elements indexed `0..n-1`.
    ///
    /// Each element starts in its own singleton set:
    /// - `parent[i] = i`
    /// - `size[i] = 1`
    ///
    /// Arguments
    /// ---------
    /// * `n` – Number of elements in the DSU.
    ///
    /// Return
    /// ------
    /// A `UnionFind` instance representing `n` singleton sets.
    pub fn new(n: usize) -> Self {
        let mut parent = Vec::with_capacity(n);
        let mut size = Vec::with_capacity(n);

        for i in 0..n {
            parent.push(i);
            size.push(1);
        }

        Self { parent, size }
    }

    /// Return the number of elements in the DSU.
    ///
    /// Return
    /// ------
    /// Number of elements currently managed by the DSU.
    ///
    /// Notes
    /// -----
    /// - This is not the number of sets/components.
    /// - Counting sets requires computing unique roots (e.g. by scanning `find(i)`).
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// Find the representative (root) of `x` with path compression.
    ///
    /// The representative is the root of the tree containing `x`.
    ///
    /// Path compression
    /// ----------------
    /// This method flattens the tree by making every visited node point directly to
    /// the root. Over many operations, this dramatically reduces future `find()` cost.
    ///
    /// Implementation
    /// --------------
    /// This is an iterative two-pass implementation:
    /// 1) walk up parent pointers to find the root,
    /// 2) walk again and redirect all nodes along the path to the root.
    ///
    /// Arguments
    /// ---------
    /// * `x` – Element index.
    ///
    /// Return
    /// ------
    /// The root representative of the set containing `x`.
    ///
    /// Notes
    /// -----
    /// - This method takes `&mut self` because path compression mutates the structure.
    /// - `x` must be a valid index in `0..self.len()`.
    #[inline]
    pub fn find(&mut self, mut x: usize) -> usize {
        // Pass 1: find root.
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }

        // Pass 2: compress path to root.
        while self.parent[x] != x {
            let p = self.parent[x];
            self.parent[x] = root;
            x = p;
        }

        root
    }

    /// Union the sets containing `a` and `b`.
    ///
    /// If `a` and `b` are already in the same set, this does nothing.
    ///
    /// Union by size
    /// -------------
    /// To keep trees shallow, the root of the smaller set is attached under
    /// the root of the larger set.
    ///
    /// Arguments
    /// ---------
    /// * `a` – First element index.
    /// * `b` – Second element index.
    ///
    /// Notes
    /// -----
    /// - This method calls `find()` and therefore performs path compression.
    /// - Indices must be valid in `0..self.len()`.
    #[inline]
    pub fn union(&mut self, a: usize, b: usize) {
        let mut ra = self.find(a);
        let mut rb = self.find(b);

        // Already in the same set.
        if ra == rb {
            return;
        }

        // Attach smaller set under larger set.
        if self.size[ra] < self.size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }

        self.parent[rb] = ra;
        self.size[ra] += self.size[rb];
    }

    /// Extend the DSU by `k` fresh singleton elements.
    ///
    /// This appends `k` new elements at the end of the DSU index space.
    /// If the DSU currently has `n` elements, the new elements will be:
    /// - `n, n+1, ..., n+k-1`
    ///
    /// Each new element starts as its own set:
    /// - `parent[idx] = idx`
    /// - `size[idx] = 1`
    ///
    /// Arguments
    /// ---------
    /// * `k` – Number of new singleton elements to append.
    ///
    /// Notes
    /// -----
    /// - This is useful when the underlying node set grows over time and the DSU
    ///   must be expanded without rebuilding from scratch.
    pub fn extend(&mut self, k: usize) {
        let start = self.parent.len();

        self.parent.reserve(k);
        self.size.reserve(k);

        for i in 0..k {
            let idx = start + i;
            self.parent.push(idx);
            self.size.push(1);
        }
    }
}
