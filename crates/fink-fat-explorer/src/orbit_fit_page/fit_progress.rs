use dioxus::prelude::*;

/// Loading feedback shown while an orbit fit job is running: a spinner plus
/// a read-only, auto-scrolling log of the pipeline's coarse stages
/// (`outfit` itself doesn't expose per-iteration progress hooks, so these
/// are our own pipeline's log lines, not the Newton solver's internals).
#[component]
pub fn FitProgress(
    logs: Vec<String>,
    #[props(default = false)] failed: bool,
    #[props(default = None)] error: Option<String>,
) -> Element {
    let text = match (failed, &error) {
        (true, Some(message)) => format!("{}\nFailed: {message}", logs.join("\n")),
        _ => logs.join("\n"),
    };
    let n_logs = logs.len();

    // Keep the log view scrolled to the latest line on every update.
    use_effect(use_reactive!(|(n_logs,)| {
        let _ = n_logs;
        document::eval(
            "const el = document.getElementById('orbit-fit-log');\
             if (el) { el.scrollTop = el.scrollHeight; }",
        );
    }));

    rsx! {
        div { class: "card bg-base-100 shadow-sm",
            div { class: "card-body gap-3",
                div { class: "flex items-center gap-3",
                    if failed {
                        span { class: "flex items-center justify-center w-8 h-8 rounded-full bg-error/20 text-error text-xl font-bold",
                            "✗"
                        }
                        span { class: "font-medium", "Fit failed" }
                    } else {
                        span { class: "loading loading-spinner loading-lg" }
                        span { class: "font-medium", "Fitting the orbit..." }
                    }
                }
                textarea {
                    id: "orbit-fit-log",
                    class: "textarea textarea-bordered font-mono text-xs h-64 w-full",
                    readonly: true,
                    value: "{text}",
                }
            }
        }
    }
}
