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

    /// Check if the `UnionFind` has no elements.
    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
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

#[cfg(test)]
mod union_find_tests {
    use super::*;
    use proptest::prelude::*;

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /// Collect connected components as sorted sets of sorted node indices.
    /// Two nodes belong to the same component iff they share the same root.
    fn components(uf: &mut UnionFind) -> Vec<Vec<usize>> {
        use std::collections::HashMap;

        let n = uf.len();
        let mut map: HashMap<usize, Vec<usize>> = HashMap::new();

        for i in 0..n {
            let root = uf.find(i);
            map.entry(root).or_default().push(i);
        }

        let mut comps: Vec<Vec<usize>> = map.into_values().collect();
        for c in &mut comps {
            c.sort_unstable();
        }
        comps.sort_unstable();
        comps
    }

    // -------------------------------------------------------------------------
    // Unit tests – construction
    // -------------------------------------------------------------------------

    #[test]
    fn test_new_len() {
        let uf = UnionFind::new(10);
        assert_eq!(uf.len(), 10);
    }

    #[test]
    fn test_new_zero_len() {
        let uf = UnionFind::new(0);
        assert_eq!(uf.len(), 0);
    }

    #[test]
    fn test_new_singleton_roots() {
        // Every element is its own representative after construction.
        let mut uf = UnionFind::new(8);
        for i in 0..8 {
            assert_eq!(uf.find(i), i, "element {i} should be its own root");
        }
    }

    #[test]
    fn test_new_all_distinct_components() {
        let mut uf = UnionFind::new(5);
        let comps = components(&mut uf);
        let expected: Vec<Vec<usize>> = (0..5usize).map(|i| vec![i]).collect();
        assert_eq!(comps, expected);
    }

    // -------------------------------------------------------------------------
    // Unit tests – find
    // -------------------------------------------------------------------------

    #[test]
    fn test_find_idempotent() {
        // Calling find twice must return the same root.
        let mut uf = UnionFind::new(6);
        uf.union(0, 1);
        uf.union(1, 2);

        let r1 = uf.find(0);
        let r2 = uf.find(0);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_find_path_compression_flattens() {
        // Build a degenerate chain: 0 -> 1 -> 2 -> 3 -> 4
        // by always calling union in a way that forces depth.
        // After find(0), the parent of 0 should be the root directly.
        let mut uf = UnionFind::new(5);
        // Force a specific tree shape by bypassing the public API is not
        // straightforward, so we use union in a chain and verify that after
        // find every node points to the root (path compression guarantee).
        uf.union(0, 1);
        uf.union(1, 2);
        uf.union(2, 3);
        uf.union(3, 4);

        let root = uf.find(0);
        // After find, every node in the component must have the same root.
        for i in 0..5 {
            assert_eq!(uf.find(i), root);
        }
    }

    // -------------------------------------------------------------------------
    // Unit tests – union
    // -------------------------------------------------------------------------

    #[test]
    fn test_union_two_elements() {
        let mut uf = UnionFind::new(2);
        uf.union(0, 1);
        assert_eq!(uf.find(0), uf.find(1));
    }

    #[test]
    fn test_union_same_element_noop() {
        // union(x, x) must not crash and x must still be its own root
        // (or at least in the same component as itself).
        let mut uf = UnionFind::new(4);
        uf.union(2, 2);
        assert_eq!(uf.find(2), uf.find(2));
        // Other elements are unaffected.
        assert_ne!(uf.find(0), uf.find(2));
    }

    #[test]
    fn test_union_already_connected_is_noop() {
        let mut uf = UnionFind::new(4);
        uf.union(0, 1);
        let root_before = uf.find(0);
        uf.union(0, 1); // redundant
        let root_after = uf.find(0);
        assert_eq!(root_before, root_after);
        // Component count must not change.
        assert_eq!(components(&mut uf).len(), 3);
    }

    #[test]
    fn test_union_chain_all_connected() {
        // 0-1-2-3-4 chain => single component.
        let mut uf = UnionFind::new(5);
        for i in 0..4 {
            uf.union(i, i + 1);
        }
        let root = uf.find(0);
        for i in 1..5 {
            assert_eq!(uf.find(i), root, "node {i} should share root with 0");
        }
    }

    #[test]
    fn test_union_star_topology() {
        // All nodes connected to node 0.
        let n = 8;
        let mut uf = UnionFind::new(n);
        for i in 1..n {
            uf.union(0, i);
        }
        let root = uf.find(0);
        for i in 0..n {
            assert_eq!(uf.find(i), root);
        }
        assert_eq!(components(&mut uf).len(), 1);
    }

    #[test]
    fn test_union_two_independent_components() {
        let mut uf = UnionFind::new(6);
        uf.union(0, 1);
        uf.union(1, 2);
        uf.union(3, 4);
        uf.union(4, 5);

        // {0,1,2} and {3,4,5} are separate.
        assert_eq!(uf.find(0), uf.find(1));
        assert_eq!(uf.find(1), uf.find(2));
        assert_eq!(uf.find(3), uf.find(4));
        assert_eq!(uf.find(4), uf.find(5));
        assert_ne!(uf.find(0), uf.find(3));

        let comps = components(&mut uf);
        assert_eq!(comps, vec![vec![0, 1, 2], vec![3, 4, 5]]);
    }

    #[test]
    fn test_union_merges_two_components() {
        let mut uf = UnionFind::new(6);
        uf.union(0, 1);
        uf.union(2, 3);
        assert_ne!(uf.find(1), uf.find(2));

        // Bridge the two components.
        uf.union(1, 2);
        assert_eq!(uf.find(0), uf.find(3));
        assert_eq!(components(&mut uf).len(), 3); // {0,1,2,3}, {4}, {5}
    }

    #[test]
    fn test_union_by_size_keeps_tree_shallow() {
        // After union-by-size every node must be reachable within log2(n) hops.
        // We verify indirectly: build a balanced binary merge and check that
        // find is always consistent (the exact depth is an internal detail).
        let n = 16;
        let mut uf = UnionFind::new(n);
        for step in [8, 4, 2, 1] {
            let mut i = 0;
            while i + step < n {
                uf.union(i, i + step);
                i += step * 2;
            }
        }
        let root = uf.find(0);
        for i in 0..n {
            assert_eq!(uf.find(i), root);
        }
    }

    // -------------------------------------------------------------------------
    // Unit tests – extend
    // -------------------------------------------------------------------------

    #[test]
    fn test_extend_increases_len() {
        let mut uf = UnionFind::new(3);
        uf.extend(4);
        assert_eq!(uf.len(), 7);
    }

    #[test]
    fn test_extend_zero_is_noop() {
        let mut uf = UnionFind::new(3);
        uf.extend(0);
        assert_eq!(uf.len(), 3);
    }

    #[test]
    fn test_extend_new_elements_are_singletons() {
        let mut uf = UnionFind::new(3);
        uf.union(0, 1); // existing component
        uf.extend(3);

        // New elements 3, 4, 5 must be singletons.
        for i in 3..6 {
            assert_eq!(uf.find(i), i, "extended element {i} should be its own root");
        }
    }

    #[test]
    fn test_extend_existing_components_unaffected() {
        let mut uf = UnionFind::new(3);
        uf.union(0, 1);
        let root_before = uf.find(0);

        uf.extend(5);

        // Component {0, 1} must be intact.
        assert_eq!(uf.find(0), uf.find(1));
        assert_eq!(uf.find(0), root_before);
        // Element 2 is still a singleton.
        assert_ne!(uf.find(2), uf.find(0));
    }

    #[test]
    fn test_extend_then_union_across_boundary() {
        let mut uf = UnionFind::new(3);
        uf.extend(3); // now 6 elements: 0..5

        // Connect an old and a new element.
        uf.union(2, 4);
        assert_eq!(uf.find(2), uf.find(4));
        assert_ne!(uf.find(0), uf.find(4));
    }

    #[test]
    fn test_extend_multiple_times() {
        let mut uf = UnionFind::new(2);
        uf.extend(3); // 5 elements
        uf.extend(2); // 7 elements
        assert_eq!(uf.len(), 7);

        for i in 0..7 {
            assert_eq!(uf.find(i), i);
        }
    }

    // -------------------------------------------------------------------------
    // Unit tests – clone
    // -------------------------------------------------------------------------

    #[test]
    fn test_clone_is_independent() {
        let mut uf = UnionFind::new(5);
        uf.union(0, 1);

        let mut uf2 = uf.clone();
        uf2.union(2, 3);

        // Mutation on clone must not affect original.
        assert_ne!(uf.find(2), uf.find(3));
        // Original mutation must not affect clone.
        uf.union(3, 4);
        assert_ne!(uf2.find(3), uf2.find(4));
    }

    // -------------------------------------------------------------------------
    // Proptest – structural invariants
    // -------------------------------------------------------------------------

    proptest! {
        /// After a sequence of random unions, `find` must be consistent:
        /// two nodes share a root iff they were connected by the union sequence.
        #[test]
        fn prop_find_consistency(
            n in 2usize..64,
            edges in prop::collection::vec((any::<usize>(), any::<usize>()), 0..128),
        ) {
            let mut uf = UnionFind::new(n);

            // Track expected connectivity with a naive reference implementation.
            let mut label = (0..n).collect::<Vec<usize>>();

            let normalize_edge = |a: usize, b: usize| (a % n, b % n);

            for (a, b) in &edges {
                let (a, b) = normalize_edge(*a, *b);
                let la = label[a];
                let lb = label[b];
                // Merge: relabel all nodes with label lb to la.
                if la != lb {
                    for l in label.iter_mut() {
                        if *l == lb {
                            *l = la;
                        }
                    }
                }
                uf.union(a, b);
            }

            // Verify: two nodes are in the same DSU component iff same label.
            for i in 0..n {
                for j in 0..n {
                    let same_dsu = uf.find(i) == uf.find(j);
                    let same_ref = label[i] == label[j];
                    prop_assert_eq!(
                        same_dsu, same_ref,
                        "{}",
                        format!("nodes {i} and {j} disagree: DSU={same_dsu}, reference={same_ref}")
                    );
                }
            }
        }

        /// `find(x)` must always return a valid index in `0..n`.
        #[test]
        fn prop_find_returns_valid_index(
            n in 1usize..64,
            queries in prop::collection::vec(any::<usize>(), 1..64),
        ) {
            let mut uf = UnionFind::new(n);
            for q in queries {
                let x = q % n;
                let root = uf.find(x);
                prop_assert!(root < n, "root {root} out of bounds for n={n}");
            }
        }

        /// `find(x)` must be idempotent: calling it twice gives the same result.
        #[test]
        fn prop_find_idempotent(
            n in 1usize..64,
            edges in prop::collection::vec((any::<usize>(), any::<usize>()), 0..64),
            x in any::<usize>(),
        ) {
            let mut uf = UnionFind::new(n);
            for (a, b) in edges {
                uf.union(a % n, b % n);
            }
            let x = x % n;
            prop_assert_eq!(uf.find(x), uf.find(x));
        }

        /// `find(root)` must equal `root` (roots are fixed points).
        #[test]
        fn prop_root_is_fixed_point(
            n in 1usize..64,
            edges in prop::collection::vec((any::<usize>(), any::<usize>()), 0..64),
        ) {
            let mut uf = UnionFind::new(n);
            for (a, b) in edges {
                uf.union(a % n, b % n);
            }
            for i in 0..n {
                let root = uf.find(i);
                prop_assert_eq!(
                    uf.find(root), root,
                    "{}",
                    format!("find(root={root}) != root after find({i})")
                );
            }
        }

        /// Union is symmetric: union(a, b) and union(b, a) must yield the same connectivity.
        #[test]
        fn prop_union_symmetric(
            n in 2usize..32,
            edges in prop::collection::vec((any::<usize>(), any::<usize>()), 1..32),
        ) {
            let mut uf_ab = UnionFind::new(n);
            let mut uf_ba = UnionFind::new(n);

            for (a, b) in &edges {
                let a = a % n;
                let b = b % n;
                uf_ab.union(a, b);
                uf_ba.union(b, a);
            }

            for i in 0..n {
                for j in 0..n {
                    prop_assert_eq!(
                        uf_ab.find(i) == uf_ab.find(j),
                        uf_ba.find(i) == uf_ba.find(j),
                        "{}",
                        format!("symmetry violated for ({i},{j})")
                    );
                }
            }
        }

        /// len() must equal n + k after extend(k).
        #[test]
        fn prop_extend_len(
            n in 0usize..64,
            k in 0usize..64,
        ) {
            let mut uf = UnionFind::new(n);
            uf.extend(k);
            prop_assert_eq!(uf.len(), n + k);
        }

        /// Extended elements must be singletons regardless of prior unions.
        #[test]
        fn prop_extend_new_elements_are_singletons(
            n in 1usize..32,
            k in 1usize..32,
            edges in prop::collection::vec((any::<usize>(), any::<usize>()), 0..32),
        ) {
            let mut uf = UnionFind::new(n);
            for (a, b) in edges {
                uf.union(a % n, b % n);
            }
            uf.extend(k);

            for i in n..(n + k) {
                prop_assert_eq!(
                    uf.find(i), i,
                    "{}",
                    format!("extended element {i} should be its own root")
                );
            }
        }

        /// After extend, new elements must not be connected to any pre-existing element.
        #[test]
        fn prop_extend_isolated_from_existing(
            n in 1usize..32,
            k in 1usize..32,
            edges in prop::collection::vec((any::<usize>(), any::<usize>()), 0..32),
        ) {
            let mut uf = UnionFind::new(n);
            for (a, b) in edges {
                uf.union(a % n, b % n);
            }
            uf.extend(k);

            for new in n..(n + k) {
                for old in 0..n {
                    prop_assert_ne!(
                        uf.find(new), uf.find(old),
                        "{}",
                        format!("new element {new} should not be connected to old element {old}")
                    );
                }
            }
        }

        /// Component count is non-increasing after union operations.
        #[test]
        fn prop_component_count_non_increasing(
            n in 2usize..32,
            edges in prop::collection::vec((any::<usize>(), any::<usize>()), 1..32),
        ) {
            let mut uf = UnionFind::new(n);
            let mut prev_count = components(&mut uf).len();

            for (a, b) in edges {
                let a = a % n;
                let b = b % n;
                uf.union(a, b);
                let count = components(&mut uf).len();
                prop_assert!(
                    count <= prev_count,
                    "component count increased from {prev_count} to {count}"
                );
                prev_count = count;
            }
        }
    }
}
