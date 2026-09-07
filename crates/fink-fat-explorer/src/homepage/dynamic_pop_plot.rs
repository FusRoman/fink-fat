use dioxus::prelude::*;
use std::collections::HashSet;

#[cfg(target_arch = "wasm32")]
use plotly::{
    common::{Marker, Mode, TickMode, Title, Visible},
    layout::{Axis, AxisType, Layout, Margin},
    Plot, Scatter,
};

use serde::{Deserialize, Serialize};

use crate::homepage::family::DynamicalFamily;

/// All the points of a single family, pre-split into the two coordinate
/// vectors plotly wants, so that toggling a family only rebuilds the traces
/// and never re-groups the whole population.
///
/// This is also the wire format: the server groups the population once when it
/// builds the homepage snapshot and ships these flat arrays, rather than one
/// JSON object per point carrying a repeated family label — roughly a sixfold
/// cut in payload at 214k branches.
#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub struct FamilySeries {
    pub family: DynamicalFamily,
    pub a: Vec<f32>,
    pub e: Vec<f32>,
}

/// Served from the in-RAM homepage snapshot; `None` while it is still being
/// built. See [`crate::homepage::snapshot`].
#[server]
pub async fn query_orbital_elements() -> Result<Option<Vec<FamilySeries>>, ServerFnError> {
    Ok(crate::homepage::snapshot::snapshot()
        .await
        .map(|snap| snap.series.clone()))
}

/// How often to re-check whether the homepage snapshot has finished building.
const WARMUP_POLL_MS: u64 = 1000;

#[component]
pub fn DynamicPopPlot(
    hidden_families: Signal<HashSet<DynamicalFamily>>,
    refresh_token: Signal<u64>,
) -> Element {
    let mut orbital_data = use_resource(move || async move {
        let _ = refresh_token();
        query_orbital_elements().await
    });
    let mut is_mounted = use_signal(|| false);
    // Whether plotly has drawn into the div at least once: the first draw
    // needs `new_plot`, every later one is a cheaper `react` diff. Only the
    // wasm build ever draws, and target arch is fixed at compile time, so
    // gating the hook keeps hook order consistent within a given build.
    #[cfg(target_arch = "wasm32")]
    let mut drawn = use_signal(|| false);

    // `Ok(None)` means the snapshot is still building — poll for it.
    use_effect(move || {
        let warming = matches!(&*orbital_data.read(), Some(Ok(None)));
        if warming {
            spawn(async move {
                crate::sleep_ms(WARMUP_POLL_MS).await;
                orbital_data.restart();
            });
        }
    });

    // Already grouped by family, in `DynamicalFamily`'s `Ord` (increasing
    // heliocentric distance) — which is the order the legend lists them in.
    let series = use_memo(move || match &*orbital_data.read() {
        Some(Ok(Some(series))) => series.clone(),
        _ => Vec::new(),
    });

    // Just the (family, point count) pairs — cheap enough to hand to the
    // legend as a prop, unlike the full coordinate vectors.
    let legend_entries = use_memo(move || {
        series
            .read()
            .iter()
            .map(|s| (s.family, s.a.len()))
            .collect::<Vec<_>>()
    });

    // Deliberately does *not* read `hidden_families`: doing so would re-render
    // this whole component (plot div included) on every legend toggle, on top
    // of `FamilyLegend`'s own re-render. The filtered count lives in the
    // legend, which is the only thing that needs to change.
    let status_text = match &*orbital_data.read() {
        Some(Ok(Some(series))) => {
            format!(
                "{} objects plotted",
                series.iter().map(|s| s.a.len()).sum::<usize>()
            )
        }
        Some(Ok(None)) => "Building the population index...".to_string(),
        Some(Err(e)) => format!("Error: {e}"),
        None => String::new(),
    };
    let is_loading = matches!(&*orbital_data.read(), None | Some(Ok(None)));

    use_effect(move || {
        #[cfg(target_arch = "wasm32")]
        {
            // Read both inside the effect so it re-runs on a legend toggle as
            // well as on a data load.
            let hidden = hidden_families();

            if is_mounted() {
                let height = web_sys::window()
                    .and_then(|w| w.inner_height().ok())
                    .and_then(|h| h.as_f64())
                    .map(|h| (h * 0.68) as usize)
                    .unwrap_or(700);

                // Build the figure synchronously against a borrow of the memo,
                // so only the finished `Plot` has to be moved into the task —
                // the borrow must be released before `spawn`.
                let plot = {
                    let series = series.read();
                    if series.is_empty() {
                        return;
                    }

                    let mut plot = Plot::new();
                    // Every family is always emitted as a trace; hidden ones
                    // are merely flipped to `Visible::False`. That keeps
                    // `react`'s diff down to one attribute instead of making
                    // it reconcile a different set of traces each toggle.
                    for s in series.iter() {
                        let visible = if hidden.contains(&s.family) {
                            Visible::False
                        } else {
                            Visible::True
                        };
                        let trace = Scatter::new(s.a.clone(), s.e.clone())
                            .name(s.family.label())
                            .mode(Mode::Markers)
                            .web_gl_mode(true)
                            .visible(visible)
                            .marker(Marker::new().color(s.family.color()));
                        plot.add_trace(trace);
                    }

                    let layout = Layout::new()
                        .height(height)
                        // Plotly's own legend is replaced by `FamilyLegend`
                        // below: plotly.rs 0.14 exposes no way to subscribe to
                        // `plotly_legendclick`, so the legend has to live on
                        // the Dioxus side to be able to drive the table too.
                        .show_legend(false)
                        .x_axis(
                            Axis::new()
                                .type_(AxisType::Log)
                                .title(Title::from("Semi-major axis (AU)"))
                                .tick_mode(TickMode::Array)
                                .tick_values(vec![
                                    0.5, 1.0, 2.0, 3.0, 4.6, 5.5, 10.0, 30.0, 100.0, 1000.0,
                                ])
                                .tick_text(vec![
                                    "0.5",
                                    "1",
                                    "2 (MB)",
                                    "3",
                                    "4.6 (Trojan)",
                                    "5.5 (Centaur)",
                                    "10",
                                    "30 (KBO)",
                                    "100",
                                    "1000",
                                ])
                                .tick_angle(-35.0),
                        )
                        .y_axis(Axis::new().title(Title::from("Eccentricity")))
                        .margin(Margin::new().top(20).right(20));

                    plot.set_layout(layout);
                    plot
                };

                spawn(async move {
                    if *drawn.peek() {
                        plotly::bindings::react("ae-plot-div", &plot).await;
                    } else {
                        plotly::bindings::new_plot("ae-plot-div", &plot).await;
                        drawn.set(true);
                    }
                });
            }
        }
    });

    rsx! {
        div { class: "card bg-base-100 shadow-sm",
            div { class: "card-body",
                div { class: "text-center mb-1",
                    h2 { class: "text-2xl font-bold tracking-tight", "The Solar System, Mapped" }
                    p { class: "text-sm opacity-60", "{status_text}" }
                }
                div { class: "relative", style: if is_loading { "min-height: 60vh;" },
                    div {
                        id: "ae-plot-div",
                        style: "width: 100%;",
                        onmounted: move |_| {
                            is_mounted.set(true);
                        },
                    }
                    if is_loading {
                        div { class: "absolute inset-0 flex items-center justify-center",
                            span { class: "loading loading-dots loading-lg" }
                        }
                    }
                }
                FamilyLegend { entries: legend_entries(), hidden_families }
            }
        }
    }
}

/// Stand-in for plotly's built-in legend: one clickable chip per family
/// present in the data, styled like the family badges in the lineage table.
/// Clicking a chip toggles that family in `hidden_families`, which both the
/// plot above and the table below read.
#[component]
fn FamilyLegend(
    entries: Vec<(DynamicalFamily, usize)>,
    mut hidden_families: Signal<HashSet<DynamicalFamily>>,
) -> Element {
    let all_families: Vec<DynamicalFamily> = entries.iter().map(|(f, _)| *f).collect();

    let (shown, total) = {
        let hidden = hidden_families.read();
        entries
            .iter()
            .fold((0usize, 0usize), |(shown, total), (family, count)| {
                if hidden.contains(family) {
                    (shown, total + count)
                } else {
                    (shown + count, total + count)
                }
            })
    };

    let summary = if shown == total {
        format!("{total} shown")
    } else {
        format!("{shown} / {total} shown")
    };

    rsx! {
        div { class: "flex flex-wrap items-center justify-center gap-1 mt-2",
            // Each chip is its own component rather than an inline element:
            // an attribute that varies per render does not get re-applied when
            // it sits on a bare element inside an rsx `for`, but does when it
            // sits at the root of a component's own template. The chips also
            // subscribe to `hidden_families` individually, so a toggle only
            // re-renders the two chips that actually changed.
            for (family , count) in entries {
                FamilyChip { family, count, hidden_families }
            }

            span { class: "mx-1 opacity-30", "|" }

            button {
                class: "btn btn-xs btn-ghost",
                onclick: move |_| hidden_families.write().clear(),
                "All"
            }
            button {
                class: "btn btn-xs btn-ghost",
                onclick: move |_| {
                    let mut set = hidden_families.write();
                    set.clear();
                    set.extend(all_families.iter().copied());
                },
                "None"
            }

            span { class: "text-xs opacity-60 ml-2", "{summary}" }
        }
    }
}

/// A single legend entry. Reads `hidden_families` itself so that toggling a
/// family re-renders only the chips whose state actually changed, and so that
/// its varying `style` lives at a template root where it is reliably
/// re-applied.
#[component]
fn FamilyChip(
    family: DynamicalFamily,
    count: usize,
    mut hidden_families: Signal<HashSet<DynamicalFamily>>,
) -> Element {
    let is_hidden = hidden_families.read().contains(&family);

    // `opacity` must be spelled out in *both* branches: dioxus merges the
    // style attribute property by property instead of replacing it, so a
    // property that is simply absent from the new value is left at its old
    // value — dropping `opacity` here would make a re-enabled family stay
    // transparent forever.
    let style = if is_hidden {
        format!("background-color: {}; opacity: 0.25;", family.color())
    } else {
        format!("background-color: {}; opacity: 1;", family.color())
    };

    let title = if is_hidden {
        "Click to show"
    } else {
        "Click to hide"
    };

    rsx! {
        button {
            r#type: "button",
            class: "badge badge-sm border-0 text-white cursor-pointer select-none transition-opacity",
            style: "{style}",
            title: "{title}",
            onclick: move |_| {
                let mut set = hidden_families.write();
                if !set.remove(&family) {
                    set.insert(family);
                }
            },
            "{family} {count}"
        }
    }
}
