#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineStage {
    LoadPersistedData,
    IngestNights,
    BuildSeeds,
    BuildEdges,
    Solve,
    FitOrbit,
    SavePersistedData,
}
