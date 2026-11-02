use dioxus::prelude::*;
use std::collections::HashSet;

#[cfg(target_arch = "wasm32")]
use plotly::{
    common::{Mode, TickMode, Title, Visible},
    layout::{Axis, AxisType, Layout, Margin},
    Plot, Scatter,
};

use serde::{Deserialize, Serialize};

use crate::homepage::family::DynamicalFamily;
use crate::homepage::population_plot_glossary::PopulationPlotGlossary;
#[cfg(target_arch = "wasm32")]
use crate::homepage::quality_tier::marker_for;
use crate::homepage::quality_tier::QualityTier;

// plotly.rs 0.14 only wraps plotly.js's `newPlot`/`react` functions, not its
// event API, so subscribing to `plotly_click` (to jump to the clicked
// point's lineage page) goes straight through a small inline JS helper
// instead. `encodeURIComponent` runs on the JS side, so a designation with
// spaces or other reserved characters still produces a valid URL.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(inline_js = "
export function bind_plotly_click_navigation(plot_id, url_prefix) {
    var gd = document.getElementById(plot_id);
    if (!gd) { return; }
    gd.on('plotly_click', function(event_data) {
        if (!event_data || !event_data.points || event_data.points.length === 0) {
            return;
        }
        var designation = event_data.points[0].customdata;
        if (designation) {
            window.location.href = url_prefix + encodeURIComponent(designation);
        }
    });
}
")]
extern "C" {
    /// Navigates the browser to `{url_prefix}{encodeURIComponent(designation)}`
    /// whenever a point in `plot_id` is clicked, reading the designation from
    /// that point's `customdata` (set per-trace alongside the hover
    /// template). A no-op if `plot_id` isn't mounted yet.
    fn bind_plotly_click_navigation(plot_id: &str, url_prefix: &str);
}

/// All the points of a single (family, quality tier) pair, pre-split into the
/// two coordinate vectors plotly wants, so that toggling a family or a tier
/// only rebuilds the traces and never re-groups the whole population.
///
/// This is also the wire format: the server groups the population once when it
/// builds the homepage snapshot and ships these flat arrays, rather than one
/// JSON object per point carrying a repeated family/tier label — roughly a
/// sixfold cut in payload at 214k branches. `lineage_designations` is the one
/// per-point field kept alongside `a`/`e` (index-aligned with both) — it
/// labels each point on hover and lets a click jump to that lineage's page.
#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub struct PlotSeries {
    pub family: DynamicalFamily,
    pub tier: QualityTier,
    pub a: Vec<f32>,
    pub e: Vec<f32>,
    pub lineage_designations: Vec<Box<str>>,
}

/// Served from the in-RAM homepage snapshot; `None` while it is still being
/// built. See [`crate::homepage::snapshot`].
#[server]
pub async fn query_orbital_elements() -> Result<Option<Vec<PlotSeries>>, ServerFnError> {
    Ok(crate::homepage::snapshot::snapshot()
        .await
        .map(|snap| snap.series.clone()))
}

/// How often to re-check whether the homepage snapshot has finished building.
const WARMUP_POLL_MS: u64 = 1000;

/// The population's per-(family, tier) series, kept alive independently of
/// the (a, e) plot itself: [`PopulationLegendBar`] needs this data even when
/// `DynamicPopPlot` isn't the active homepage view, since the legend now
/// stays visible across all 3 views. `Home()` calls this once and hands the
/// resulting resource to both.
///
/// # Arguments
/// * `refresh_token` — bumped after a snapshot rebuild lands; restarts the
///   fetch against the new snapshot.
///
/// # Return
/// The raw query resource: `Ok(None)` while the snapshot is still building
/// (this function polls for it internally), `Ok(Some(series))` once ready,
/// `Err` on a transport failure.
pub fn use_population_series(
    refresh_token: Signal<u64>,
) -> Resource<Result<Option<Vec<PlotSeries>>, ServerFnError>> {
    let mut orbital_data = use_resource(move || async move {
        let _ = refresh_token();
        query_orbital_elements().await
    });

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

    orbital_data
}

#[component]
pub fn DynamicPopPlot(
    orbital_data: Resource<Result<Option<Vec<PlotSeries>>, ServerFnError>>,
    hidden_families: Signal<HashSet<DynamicalFamily>>,
    hidden_tiers: Signal<HashSet<QualityTier>>,
) -> Element {
    let mut is_mounted = use_signal(|| false);
    // Whether plotly has drawn into the div at least once: the first draw
    // needs `new_plot`, every later one is a cheaper `react` diff. Only the
    // wasm build ever draws, and target arch is fixed at compile time, so
    // gating the hook keeps hook order consistent within a given build.
    #[cfg(target_arch = "wasm32")]
    let mut drawn = use_signal(|| false);

    // Only the wasm build ever draws a plot from this, so — like `drawn`
    // above — this is only created there; target arch is fixed at compile
    // time, so hook order stays consistent within a given build.
    #[cfg(target_arch = "wasm32")]
    let series = use_memo(move || match &*orbital_data.read() {
        Some(Ok(Some(series))) => series.clone(),
        _ => Vec::new(),
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
            let hidden_t = hidden_tiers();

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
                    // Every (family, tier) pair is always emitted as a trace;
                    // hidden ones are merely flipped to `Visible::False`.
                    // That keeps `react`'s diff down to one attribute instead
                    // of making it reconcile a different set of traces each
                    // toggle.
                    for s in series.iter() {
                        let visible = if hidden.contains(&s.family) || hidden_t.contains(&s.tier) {
                            Visible::False
                        } else {
                            Visible::True
                        };
                        // Family/tier are constant for the whole trace, so
                        // they're baked into the template as plain text;
                        // `%{customdata}`/`%{x}`/`%{y}` are the per-point
                        // placeholders plotly.js fills in at hover time.
                        // `<extra></extra>` drops the secondary trace-name
                        // box the default template would otherwise add.
                        let hover_template = format!(
                            "<b>{family} · {tier}</b><br>Lineage: %{{customdata}}<br>\
                             Semi-major axis: %{{x:.3f}} AU<br>Eccentricity: %{{y:.3f}}\
                             <extra></extra>",
                            family = s.family.label(),
                            tier = s.tier.label(),
                        );
                        let custom_data: Vec<String> = s
                            .lineage_designations
                            .iter()
                            .map(|d| d.to_string())
                            .collect();
                        let trace = Scatter::new(s.a.clone(), s.e.clone())
                            .name(format!("{} · {}", s.family.label(), s.tier.label()))
                            .mode(Mode::Markers)
                            .web_gl_mode(true)
                            .visible(visible)
                            .marker(marker_for(s.family, s.tier))
                            .custom_data(custom_data)
                            .hover_template(hover_template);
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
                        // The graph div persists across every later `react`
                        // (only its data changes), so the click listener only
                        // needs binding once, right after the first draw.
                        bind_plotly_click_navigation("ae-plot-div", "/lineage/");
                    }
                });
            }
        }
    });

    rsx! {
        div { class: "card bg-base-100 shadow-sm",
            div { class: "card-body",
                div { class: "text-center mb-1",
                    div { class: "flex items-center justify-center gap-2",
                        h2 { class: "text-2xl font-bold tracking-tight", "The Solar System, Mapped" }
                        PopulationPlotGlossary {}
                    }
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
            }
        }
    }
}

/// The legend row shown between the navbar and the active homepage view: one
/// clickable chip per dynamical family plus one per quality tier, driving
/// `hidden_families`/`hidden_tiers` — which the (a, e) plot's traces and the
/// lineage table's query both read. Kept alive independently of which view
/// is active by taking the shared [`use_population_series`] resource rather
/// than fetching its own copy.
#[component]
pub fn PopulationLegendBar(
    orbital_data: Resource<Result<Option<Vec<PlotSeries>>, ServerFnError>>,
    hidden_families: Signal<HashSet<DynamicalFamily>>,
    hidden_tiers: Signal<HashSet<QualityTier>>,
) -> Element {
    let series = use_memo(move || match &*orbital_data.read() {
        Some(Ok(Some(series))) => series.clone(),
        _ => Vec::new(),
    });

    let family_legend_entries = use_memo(move || {
        let mut counts: Vec<(DynamicalFamily, usize)> = Vec::new();
        for s in series.read().iter() {
            match counts.iter_mut().find(|(f, _)| *f == s.family) {
                Some((_, count)) => *count += s.a.len(),
                None => counts.push((s.family, s.a.len())),
            }
        }
        counts
    });

    let tier_legend_entries = use_memo(move || {
        let mut counts: Vec<(QualityTier, usize)> = Vec::new();
        for s in series.read().iter() {
            match counts.iter_mut().find(|(t, _)| *t == s.tier) {
                Some((_, count)) => *count += s.a.len(),
                None => counts.push((s.tier, s.a.len())),
            }
        }
        counts.sort_by_key(|(tier, _)| *tier);
        counts
    });

    rsx! {
        div { class: "card bg-base-100 shadow-sm",
            div { class: "card-body py-2",
                FamilyLegend { entries: family_legend_entries(), hidden_families }
                TierLegend { entries: tier_legend_entries(), hidden_tiers }
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

/// Second legend row, same pattern as [`FamilyLegend`]/[`FamilyChip`] but for
/// [`QualityTier`]: one clickable chip per tier present in the data, toggling
/// membership in `hidden_tiers` — which both the plot's marker traces and the
/// table's "Quality" column filter read.
#[component]
fn TierLegend(
    entries: Vec<(QualityTier, usize)>,
    mut hidden_tiers: Signal<HashSet<QualityTier>>,
) -> Element {
    let all_tiers: Vec<QualityTier> = entries.iter().map(|(t, _)| *t).collect();

    let (shown, total) = {
        let hidden = hidden_tiers.read();
        entries
            .iter()
            .fold((0usize, 0usize), |(shown, total), (tier, count)| {
                if hidden.contains(tier) {
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
        div { class: "flex flex-wrap items-center justify-center gap-1 mt-1",
            for (tier , count) in entries {
                TierChip { tier, count, hidden_tiers }
            }

            span { class: "mx-1 opacity-30", "|" }

            button {
                class: "btn btn-xs btn-ghost",
                onclick: move |_| hidden_tiers.write().clear(),
                "All"
            }
            button {
                class: "btn btn-xs btn-ghost",
                onclick: move |_| {
                    let mut set = hidden_tiers.write();
                    set.clear();
                    set.extend(all_tiers.iter().copied());
                },
                "None"
            }

            span { class: "text-xs opacity-60 ml-2", "{summary}" }
        }
    }
}

/// A single tier legend entry, showing the same glyph the plot marker uses
/// for that tier so the legend visually teaches the marker mapping.
#[component]
fn TierChip(
    tier: QualityTier,
    count: usize,
    mut hidden_tiers: Signal<HashSet<QualityTier>>,
) -> Element {
    let is_hidden = hidden_tiers.read().contains(&tier);

    let class = if is_hidden {
        format!("badge badge-sm {} opacity-30", tier.badge_class())
    } else {
        format!("badge badge-sm {}", tier.badge_class())
    };

    let title = if is_hidden {
        "Click to show"
    } else {
        "Click to hide"
    };

    rsx! {
        button {
            r#type: "button",
            class: "{class} cursor-pointer select-none transition-opacity",
            title: "{title}",
            onclick: move |_| {
                let mut set = hidden_tiers.write();
                if !set.remove(&tier) {
                    set.insert(tier);
                }
            },
            "{tier.glyph()} {tier} {count}"
        }
    }
}
