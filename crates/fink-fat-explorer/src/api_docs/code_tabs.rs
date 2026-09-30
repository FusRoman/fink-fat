//! Tabbed code viewer with a "Copy" button, used by the API documentation.

use dioxus::prelude::*;

/// One tab of [`CodeTabs`].
#[derive(Clone, Debug, PartialEq)]
pub struct CodeTab {
    /// Tab title, e.g. `"Python"`.
    pub label: &'static str,
    /// highlight.js language name (`"python"`, `"rust"`, `"bash"`, `"json"`).
    pub lang: &'static str,
    /// Source code shown in the tab.
    pub code: String,
}

impl CodeTab {
    /// Builds a tab.
    ///
    /// # Arguments
    ///
    /// * `label` - tab title.
    /// * `lang` - highlight.js language name used for syntax highlighting.
    /// * `code` - source code shown in the tab.
    ///
    /// # Return
    ///
    /// The tab.
    pub fn new(label: &'static str, lang: &'static str, code: impl Into<String>) -> Self {
        Self {
            label,
            lang,
            code: code.into(),
        }
    }
}

/// Code samples in tabs (one language per tab), with a button copying the
/// visible sample to the clipboard.
///
/// # Arguments
///
/// * `tabs` - the tabs, the first one being selected initially.
#[component]
pub fn CodeTabs(tabs: Vec<CodeTab>) -> Element {
    let mut selected = use_signal(|| 0usize);
    let mut copied = use_signal(|| false);
    let current = selected().min(tabs.len().saturating_sub(1));
    let (language, code) = tabs
        .get(current)
        .map(|t| (t.lang, t.code.clone()))
        .unwrap_or_default();

    // Highlights the code blocks not highlighted yet. Re-runs whenever the
    // selected tab changes. highlight.js replaces the element's text node by
    // `<span>`s, which Dioxus does not know about: updating the old text node
    // afterwards would silently do nothing. That is why the `code` element
    // below is rendered inside a keyed one-item list: a new key makes Dioxus
    // drop and recreate the element (a `key` on a lone element is ignored).
    use_effect(move || {
        let _ = selected();
        document::eval(
            "if (window.hljs) {
                 document.querySelectorAll('code.hljs-target:not([data-highlighted])')
                     .forEach((el) => window.hljs.highlightElement(el));
             }",
        );
    });

    let copy = {
        let code = code.clone();
        move |_| {
            let eval = document::eval("navigator.clipboard.writeText(await dioxus.recv());");
            let _ = eval.send(code.clone());
            copied.set(true);
        }
    };

    rsx! {
        div { class: "card bg-base-100 shadow-sm",
            div { class: "flex items-center justify-between px-2 pt-2",
                div { role: "tablist", class: "tabs tabs-box",
                    for (index , tab) in tabs.iter().enumerate() {
                        button {
                            key: "{tab.label}",
                            role: "tab",
                            class: if index == current { "tab tab-active" } else { "tab" },
                            onclick: move |_| {
                                selected.set(index);
                                copied.set(false);
                            },
                            "{tab.label}"
                        }
                    }
                }
                button {
                    class: "btn btn-xs btn-ghost",
                    r#type: "button",
                    onclick: copy,
                    if copied() {
                        "Copied ✓"
                    } else {
                        "Copy"
                    }
                }
            }
            pre { class: "overflow-x-auto text-sm leading-relaxed",
                for tab_key in [format!("{current}-{language}")] {
                    code {
                        key: "{tab_key}",
                        class: "hljs-target language-{language} bg-transparent!",
                        "{code}"
                    }
                }
            }
        }
    }
}
