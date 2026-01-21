// use std::time::Instant;

// use crate::{
//     graph::{graph::InterNightGraph, node_id::NodeId},
//     solver::{
//         Solver, SolverOutput,
//         components::{ComponentStats, ConnectedComponents},
//         min_cost_flow::MinCostFlowSolver,
//         trivial_solver::TrivialSolver,
//     },
// };

// /// Solver choice for a component.
// #[derive(Copy, Clone, Debug, PartialEq, Eq)]
// pub enum SolverChoice {
//     /// Skip or solve with a tiny direct method.
//     Trivial,
//     /// Use min-cost flow (global optimum within the component).
//     MinCostFlow,
//     /// Use blob-breaker: window + partition + local peeling.
//     BlobBreaker,
// }

// /// A single work item produced by the planner.
// #[derive(Clone, Debug)]
// pub struct WorkItem {
//     pub component_id: u32,
//     pub choice: SolverChoice,
// }

// /// A plan for one solver pass.
// #[derive(Clone, Debug, Default)]
// pub struct SolvePlan {
//     pub items: Vec<WorkItem>,
// }

// /// Policy parameters for routing.
// ///
// /// Notes
// /// -----
// /// This is intentionally simple. You can refine later (degree caps, density,
// /// night-window triggers, etc.).
// #[derive(Copy, Clone, Debug)]
// pub struct SolverPolicy {
//     pub trivial_max_nodes: u32,
//     pub trivial_max_active_edges: u32,

//     /// If estimated MCF cost exceeds this, route to blob-breaker.
//     pub mcf_budget_s: f64,

//     /// Seconds per (edge * log2(nodes+1)) for the MCF solver.
//     ///
//     /// You can calibrate this constant with a small benchmark.
//     pub k_mcf_s_per_edge_logn: f64,

//     /// If a component spans more nights than this, route to blob-breaker.
//     pub max_night_span_for_mcf: u32,
// }

// impl Default for SolverPolicy {
//     fn default() -> Self {
//         Self {
//             trivial_max_nodes: 8,
//             trivial_max_active_edges: 16,
//             mcf_budget_s: 0.05,          // 50 ms
//             k_mcf_s_per_edge_logn: 1e-8, // placeholder (calibrate later)
//             max_night_span_for_mcf: 4,   // typical blob-breaker window size
//         }
//     }
// }

// /// Manager object holding the routing policy.
// #[derive(Clone, Debug)]
// pub struct SolverManager {
//     pub policy: SolverPolicy,
// }

// impl SolverManager {
//     /// Build a solve plan from components + stats.
//     pub fn make_plan(&self, cc: &ConnectedComponents, stats: &[ComponentStats]) -> SolvePlan {
//         debug_assert_eq!(cc.components.len(), stats.len());

//         let mut items = Vec::with_capacity(cc.components.len());
//         for (cid, st) in stats.iter().enumerate() {
//             let choice = self.classify(*st);
//             items.push(WorkItem {
//                 component_id: cid as u32,
//                 choice,
//             });
//         }

//         SolvePlan { items }
//     }

//     /// Execute the plan (skeleton).
//     ///
//     /// Notes
//     /// -----
//     /// This does not implement MCF/blob-breaker yet. It is just the orchestration
//     /// skeleton and the place where you will:
//     /// - run the chosen solver,
//     /// - run IOD on returned tracks,
//     /// - deactivate the corresponding edges.
//     /// Execute the plan (trivial solver only for now).
//     ///
//     /// Notes
//     /// -----
//     /// - Returns one `SolverOutput` per planned work item (in plan order).
//     /// - MinCostFlow / BlobBreaker are not implemented yet and currently return
//     ///   empty outputs.
//     pub fn run_plan(
//         &self,
//         graph: &InterNightGraph,
//         cc: &ConnectedComponents,
//         plan: &SolvePlan,
//     ) -> Vec<SolverOutput> {
//         // Compute stats once (cheap) so each solver gets the right numbers.
//         let stats: Vec<ComponentStats> = cc.compute_stats(graph);

//         // Trivial solver instance (you can move this into SolverManager later).
//         let trivial = TrivialSolver::default();

//         let min_cost_flow = MinCostFlowSolver::default();

//         let mut outputs: Vec<SolverOutput> = Vec::with_capacity(plan.items.len());

//         for item in &plan.items {
//             let cid = item.component_id as usize;
//             let nodes = &cc.components[cid];
//             let st = stats[cid];

//             let mut out: SolverOutput = match item.choice {
//                 SolverChoice::Trivial => trivial.solve(graph, nodes, st),

//                 SolverChoice::MinCostFlow => min_cost_flow.solve(graph, nodes, st),

//                 SolverChoice::BlobBreaker => {
//                     let t0 = Instant::now();

//                     let window_span = self.policy.max_night_span_for_mcf.max(1);

//                     // Night bounds for the component.
//                     let mut min_night = u32::MAX;
//                     let mut max_night = 0u32;
//                     for &nid in nodes {
//                         let night = graph.nodes[nid.idx()].night.0;
//                         min_night = min_night.min(night);
//                         max_night = max_night.max(night);
//                     }

//                     if min_night == u32::MAX {
//                         SolverOutput {
//                             diag: crate::solver::SolverDiagnostics {
//                                 solver_name: "blob_breaker",
//                                 n_nodes: st.n_nodes,
//                                 m_active_edges: st.m_active_edges,
//                                 ..Default::default()
//                             },
//                             ..Default::default()
//                         }
//                     } else {
//                         let mut tracks = Vec::new();
//                         let mut proposed_deactivations = Vec::new();

//                         // Bitsets to keep peeling/membership O(1).
//                         let mut peeled: Vec<bool> = vec![false; graph.nodes.len()];
//                         let mut membership: Vec<bool> = vec![false; graph.nodes.len()];
//                         let mut window_nodes: Vec<NodeId> = Vec::new();

//                         let mut start_night = min_night;
//                         loop {
//                             let end_night = start_night.saturating_add(window_span);

//                             window_nodes.clear();
//                             for &nid in nodes {
//                                 if peeled[nid.idx()] {
//                                     continue;
//                                 }
//                                 let night = graph.nodes[nid.idx()].night.0;
//                                 if night >= start_night && night <= end_night {
//                                     window_nodes.push(nid);
//                                 }
//                             }

//                             if window_nodes.len() >= 2 {
//                                 let parts = ConnectedComponents::recompute_local_exact(
//                                     graph,
//                                     &window_nodes,
//                                 );
//                                 for comp in parts {
//                                     if comp.len() < 2 {
//                                         continue;
//                                     }

//                                     let st_local =
//                                         component_stats_fast(graph, &comp, &mut membership);
//                                     let mut local_out = trivial.solve(graph, &comp, st_local);

//                                     // Peel nodes used by returned tracks.
//                                     for tr in &local_out.tracks {
//                                         for &nid in &tr.nodes {
//                                             peeled[nid.idx()] = true;
//                                         }
//                                     }

//                                     tracks.append(&mut local_out.tracks);
//                                     proposed_deactivations
//                                         .append(&mut local_out.proposed_deactivations);
//                                 }
//                             }

//                             if end_night >= max_night {
//                                 break;
//                             }
//                             start_night = end_night.saturating_add(1);
//                         }

//                         proposed_deactivations.sort_unstable();
//                         proposed_deactivations.dedup();

//                         // Keep tracks best-first for downstream consumers.
//                         tracks.sort_by(|a, b| {
//                             a.cost
//                                 .partial_cmp(&b.cost)
//                                 .unwrap_or(std::cmp::Ordering::Equal)
//                         });

//                         let nb_tracks = tracks.len();

//                         SolverOutput {
//                             tracks,
//                             proposed_deactivations,
//                             diag: crate::solver::SolverDiagnostics {
//                                 solver_name: "blob_breaker",
//                                 n_nodes: st.n_nodes,
//                                 m_active_edges: st.m_active_edges,
//                                 n_candidates: nodes.len() as u32,
//                                 n_selected: nb_tracks as u32,
//                                 time_spent_s: t0.elapsed().as_secs_f64(),
//                                 ..Default::default()
//                             },
//                         }
//                     }
//                 }
//             };

//             // Component id is known here (WorkItem), not in ComponentStats.
//             out.diag.component_id = item.component_id;

//             // (Later) orchestration hooks:
//             // - run IOD on out.tracks
//             // - validate
//             // - graph.deactivate_edges(...)
//             //
//             // For now we only return the solver outputs.
//             outputs.push(out);
//         }

//         outputs
//     }

//     /// Classify a component into a solver choice using the current policy.
//     #[inline]
//     pub fn classify(&self, st: ComponentStats) -> SolverChoice {
//         // Trivial fast path.
//         if st.n_nodes <= self.policy.trivial_max_nodes
//             || st.m_active_edges <= self.policy.trivial_max_active_edges
//         {
//             return SolverChoice::Trivial;
//         }

//         // Night span guardrail.
//         if st.night_span > self.policy.max_night_span_for_mcf {
//             return SolverChoice::BlobBreaker;
//         }

//         // Budget-aware MCF vs BlobBreaker.
//         let t_est = estimate_mcf_time_s(self.policy, st.n_nodes, st.m_active_edges);
//         if t_est <= self.policy.mcf_budget_s {
//             SolverChoice::MinCostFlow
//         } else {
//             SolverChoice::BlobBreaker
//         }
//     }
// }

// #[inline]
// fn estimate_mcf_time_s(policy: SolverPolicy, n_nodes: u32, m_edges: u32) -> f64 {
//     let logn = ((n_nodes as f64) + 1.0).log2();
//     policy.k_mcf_s_per_edge_logn * (m_edges as f64) * logn
// }

// /// Cheap per-component stats without reallocating global connected components.
// fn component_stats_fast(
//     graph: &InterNightGraph,
//     nodes: &[NodeId],
//     membership: &mut Vec<bool>,
// ) -> ComponentStats {
//     if nodes.is_empty() {
//         return ComponentStats::default();
//     }

//     let mut touched: Vec<usize> = Vec::with_capacity(nodes.len());
//     let mut min_night = u32::MAX;
//     let mut max_night = 0u32;

//     for &nid in nodes {
//         let idx = nid.idx();
//         if !membership[idx] {
//             membership[idx] = true;
//             touched.push(idx);
//         }
//         let night = graph.nodes[idx].night.0;
//         min_night = min_night.min(night);
//         max_night = max_night.max(night);
//     }

//     let mut m_active_edges = 0u32;
//     for &nid in nodes {
//         for &eid in &graph.out_adj[nid.idx()] {
//             let e = &graph.edges[eid.idx()];
//             if e.active && membership[e.to.idx()] {
//                 m_active_edges += 1;
//             }
//         }
//     }

//     for idx in touched {
//         membership[idx] = false;
//     }

//     ComponentStats {
//         n_nodes: nodes.len() as u32,
//         m_active_edges,
//         night_span: max_night.saturating_sub(min_night),
//     }
// }
