use dioxus::prelude::*;

use super::hypotheses_plot::HypothesesPlot;
use super::kf_replay::{HypothesisSnapshot, KfStep};
use super::metrics_plot::MetricsPlot;
use super::rho_evolution_plot::RhoEvolutionPlot;

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

    rsx! {
        div { class: "flex flex-col gap-2",
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

            match active() {
                Tab::FilterMetrics => rsx! {
                    MetricsPlot { replay }
                },
                Tab::Rho => rsx! {
                    RhoEvolutionPlot { replay, hypotheses }
                },
                Tab::Hypotheses => rsx! {
                    HypothesesPlot { replay }
                },
            }
        }
    }
}
