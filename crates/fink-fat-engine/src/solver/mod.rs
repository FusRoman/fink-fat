pub mod components;

/// Disjoint Set Union (Union-Find) with path compression and union by size.
///
/// Notes
/// -----
/// - Indices are `usize` and refer to node indices in `InterNightGraph::nodes`.
/// - This DSU is intended for *undirected* connectivity.
#[derive(Debug, Clone)]
pub struct UnionFind {
    parent: Vec<usize>,
    size: Vec<u32>,
}

impl UnionFind {
    /// Create a DSU over `n` elements: 0..n-1.
    pub fn new(n: usize) -> Self {
        let mut parent = Vec::with_capacity(n);
        let mut size = Vec::with_capacity(n);
        for i in 0..n {
            parent.push(i);
            size.push(1);
        }
        Self { parent, size }
    }

    /// Current number of elements.
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// Find the representative (root) of `x` with path compression.
    #[inline]
    pub fn find(&mut self, mut x: usize) -> usize {
        // Iterative path compression (two-pass).
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        while self.parent[x] != x {
            let p = self.parent[x];
            self.parent[x] = root;
            x = p;
        }
        root
    }

    /// Union the sets containing `a` and `b`.
    #[inline]
    pub fn union(&mut self, a: usize, b: usize) {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return;
        }

        // Union by size: attach smaller tree under larger tree.
        if self.size[ra] < self.size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        self.size[ra] += self.size[rb];
    }

    /// Extend the DSU by `k` fresh singleton elements.
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
