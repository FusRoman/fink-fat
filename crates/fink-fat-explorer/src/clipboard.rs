//! Copy-to-clipboard for the explorer's "Copy" buttons, in Rust through
//! `web-sys`.
//!
//! The async Clipboard API (`navigator.clipboard`) only exists in a secure
//! context (HTTPS or `localhost`): on the explorer served over plain HTTP
//! on a LAN address it is `undefined`, which `web-sys` cannot express (its
//! `Navigator::clipboard` is not optional) — calling it there would throw.
//! [`copy_text`] therefore checks [`web_sys::Window::is_secure_context`]
//! first and falls back to the legacy `execCommand('copy')` on a temporary
//! textarea otherwise.
//!
//! Both paths need the browser to see the copy as part of the user's click,
//! so [`copy_text`] starts the copy synchronously, in the click handler, and
//! only *awaits* its outcome afterwards.

use std::future::Future;

use dioxus::prelude::*;

/// Copy `text` to the clipboard.
///
/// # Arguments
/// * `text` — the text to copy.
///
/// # Return
/// A future resolving to whether the browser accepted the copy. The copy
/// itself is started before this function returns, so it must be called
/// from the click handler itself, not from a task spawned by it.
///
/// # Notes
/// Outside the browser (server build) this does nothing and resolves to
/// `false`; event handlers never run there.
#[cfg(target_arch = "wasm32")]
pub fn copy_text(text: &str) -> impl Future<Output = bool> {
    let started = start_copy(text);
    async move {
        match started {
            Ok(promise) => wasm_bindgen_futures::JsFuture::from(promise).await.is_ok(),
            Err(copied) => copied,
        }
    }
}

/// Server-build counterpart of the wasm [`copy_text`]: never called, since
/// event handlers only run in the browser.
///
/// # Arguments
/// * `text` — unused.
///
/// # Return
/// A future resolving to `false`.
#[cfg(not(target_arch = "wasm32"))]
pub fn copy_text(text: &str) -> impl Future<Output = bool> {
    let _ = text;
    std::future::ready(false)
}

/// Start copying `text` with whichever API the page's context allows.
///
/// # Arguments
/// * `text` — the text to copy.
///
/// # Return
/// `Ok` with the pending `writeText` promise in a secure context, `Err`
/// with the already-known outcome of the `execCommand` fallback otherwise.
#[cfg(target_arch = "wasm32")]
fn start_copy(text: &str) -> Result<js_sys::Promise, bool> {
    let Some(window) = web_sys::window() else {
        return Err(false);
    };
    if window.is_secure_context() {
        return Ok(window.navigator().clipboard().write_text(text));
    }
    Err(copy_with_exec_command(&window, text).unwrap_or(false))
}

/// Copy `text` through a temporary off-screen textarea and the legacy
/// `execCommand('copy')`, the only way to copy outside a secure context.
///
/// # Arguments
/// * `window` — the browser window.
/// * `text` — the text to copy.
///
/// # Return
/// Whether the browser accepted the copy.
///
/// # Errors
/// Any DOM call failing (no document or body, element creation refused).
#[cfg(target_arch = "wasm32")]
fn copy_with_exec_command(
    window: &web_sys::Window,
    text: &str,
) -> Result<bool, wasm_bindgen::JsValue> {
    use wasm_bindgen::{JsCast, JsValue};

    let document: web_sys::HtmlDocument = window
        .document()
        .ok_or(JsValue::NULL)?
        .dyn_into()
        .map_err(JsValue::from)?;
    let body = document.body().ok_or(JsValue::NULL)?;
    let area: web_sys::HtmlTextAreaElement = document
        .create_element("textarea")?
        .dyn_into()
        .map_err(JsValue::from)?;
    area.set_value(text);
    area.set_read_only(true);
    area.set_attribute("style", "position: fixed; left: -9999px;")?;
    body.append_child(&area)?;
    area.select();
    let copied = document.exec_command("copy");
    area.remove();
    copied
}

/// Caption of a [`CopyButton`].
///
/// # Arguments
/// * `label` — the idle caption.
/// * `outcome` — the outcome of the last copy of the current text, `None`
///   if there was none.
///
/// # Return
/// `label`, `"✓ Copied"` or `"✗ Copy failed"`.
fn copy_button_caption(label: &str, outcome: Option<bool>) -> String {
    match outcome {
        None => label.to_string(),
        Some(true) => "✓ Copied".to_string(),
        Some(false) => "✗ Copy failed".to_string(),
    }
}

/// A button copying `text` to the clipboard, whose caption then reports
/// whether the copy worked.
///
/// # Arguments
/// * `text` — the text to copy.
/// * `label` — the idle caption, e.g. `"📋 Copy"`.
/// * `class` — the button's classes.
#[component]
pub fn CopyButton(
    text: String,
    label: String,
    #[props(default = "btn btn-sm btn-outline".to_string())] class: String,
) -> Element {
    // The last outcome, with the text it applies to: when `text` changes
    // (another code tab, another row) the caption goes back to `label`.
    let mut last_copy = use_signal(|| None::<(String, bool)>);
    let outcome = last_copy()
        .filter(|(copied_text, _)| *copied_text == text)
        .map(|(_, copied)| copied);

    rsx! {
        button {
            class,
            r#type: "button",
            title: if outcome == Some(false) { "The browser refused the copy: select the text by hand" },
            onclick: {
                let text = text.clone();
                move |_| {
                    let copy = copy_text(&text);
                    let text = text.clone();
                    spawn(async move {
                        let copied = copy.await;
                        last_copy.set(Some((text, copied)));
                    });
                }
            },
            "{copy_button_caption(&label, outcome)}"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_button_caption_reports_each_outcome() {
        assert_eq!(copy_button_caption("📋 Copy", None), "📋 Copy");
        assert_eq!(copy_button_caption("📋 Copy", Some(true)), "✓ Copied");
        assert_eq!(copy_button_caption("📋 Copy", Some(false)), "✗ Copy failed");
    }
}
