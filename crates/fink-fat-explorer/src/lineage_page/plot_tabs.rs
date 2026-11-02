use dioxus::prelude::*;

use super::hypotheses_plot::HypothesesPlot;
use super::kf_replay::{HypothesisSnapshot, KfStep};
use super::metrics_plot::MetricsPlot;
use super::rho_evolution_plot::RhoEvolutionPlot;
use super::x_axis::XAxisUnit;

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    FilterMetrics,
    Rho,
    Hypotheses,
}

impl Tab {
    fn label(self) -> &'static str {
        match self {
            Tab::FilterMetrics => "Filter metrics",
            Tab::Rho => "ρ / ρ̇",
            Tab::Hypotheses => "Hypotheses",
        }
    }
}

const ALL_TABS: [Tab; 3] = [Tab::FilterMetrics, Tab::Rho, Tab::Hypotheses];

/// One plot group visible at a time behind a tab bar — the replay produces
/// more plots (filter-consistency metrics, ρ/ρ̇ evolution, hypothesis-bank
/// diagnostics) than are useful to show all at once.
#[component]
pub fn PlotTabs(replay: Vec<KfStep>, hypotheses: Vec<HypothesisSnapshot>) -> Element {
    let mut active = use_signal(|| Tab::FilterMetrics);
    let mut x_axis_unit = use_signal(|| XAxisUnit::ObservationIndex);

    rsx! {
        div { class: "flex flex-col gap-2",
            div { class: "flex flex-wrap items-center justify-between gap-2",
                div { role: "tablist", class: "tabs tabs-boxed w-fit",
                    for tab in ALL_TABS {
                        button {
                            key: "{tab.label()}",
                            role: "tab",
                            class: if active() == tab { "tab tab-active" } else { "tab" },
                            onclick: move |_| active.set(tab),
                            "{tab.label()}"
                        }
                    }
                }

                div { class: "join",
                    for unit in XAxisUnit::ALL {
                        button {
                            key: "{unit.label()}",
                            class: if x_axis_unit() == unit { "join-item btn btn-sm btn-active" } else { "join-item btn btn-sm" },
                            onclick: move |_| x_axis_unit.set(unit),
                            "{unit.label()}"
                        }
                    }
                }
            }

            match active() {
                Tab::FilterMetrics => rsx! {
                    MetricsPlot { replay, x_axis_unit: x_axis_unit() }
                },
                Tab::Rho => rsx! {
                    RhoEvolutionPlot { replay, hypotheses, x_axis_unit: x_axis_unit() }
                },
                Tab::Hypotheses => rsx! {
                    HypothesesPlot { replay, x_axis_unit: x_axis_unit() }
                },
            }
        }
    }
}
