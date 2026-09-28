//! Modal for inspecting one submission's stored ADES XML — the exact
//! document that was sent to MPC (`mpc_submissions.xml`). Reuses
//! `lineage_page::ades_export_modal`'s modal shell/mount convention and
//! `submission_dashboard::prepare_menu`'s copy/download closures verbatim
//! rather than inventing new mechanics for either.
//!
//! The XML shown on screen is a re-indented, syntax-colored *copy* of the
//! stored value ([`fink_fat_ades::xml_format::pretty_print_xml`] +
//! [`tokenize_xml`] below) — purely cosmetic, whitespace-only reformatting.
//! "Copy" copies that readable copy; "Download" writes the original,
//! byte-for-byte stored value, so the downloaded file always matches
//! exactly what MPC received.

use dioxus::prelude::*;

use super::data::get_submission_xml;

/// The download filename for a submission's ADES XML.
///
/// # Arguments
/// * `lineage_designation` — the submission's lineage designation.
///
/// # Return
/// The filename.
pub fn ades_file_name(lineage_designation: &str) -> String {
    format!("{lineage_designation}.ades.xml")
}

/// One highlighting category [`tokenize_xml`] assigns to a span of XML text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum XmlSpanKind {
    /// Structural characters: `<`, `</`, `<?`, `>`, `/>`, `?>`, `=`.
    Punctuation,
    /// An element or processing-instruction name.
    TagName,
    /// An attribute name.
    AttrName,
    /// A quoted attribute value (quotes included).
    AttrValue,
    /// Everything else: element text content, and the whitespace/newlines
    /// [`fink_fat_ades::xml_format::pretty_print_xml`] introduced between
    /// tags.
    Text,
}

/// One classified slice of the input, verbatim (concatenating every span's
/// `text` in order reconstructs the exact input string).
#[derive(Debug, Clone, PartialEq, Eq)]
struct XmlSpan {
    kind: XmlSpanKind,
    text: String,
}

fn push_span(spans: &mut Vec<XmlSpan>, kind: XmlSpanKind, text: &str) {
    if !text.is_empty() {
        spans.push(XmlSpan {
            kind,
            text: text.to_string(),
        });
    }
}

/// A small, dependency-free XML tokenizer for syntax-highlighting an
/// already-valid XML document (this app only ever tokenizes XML it
/// generated itself via [`fink_fat_ades::xml`], so a full, spec-complete XML
/// grammar isn't needed) — element/PI names, attribute names, quoted
/// attribute values, structural punctuation, and everything else (text
/// content and inter-tag whitespace) as distinct, colorable spans.
///
/// Never panics: unrecognized input past a malformed point is simply
/// emitted as one trailing [`XmlSpanKind::Text`] span rather than looping or
/// erroring — this is a display helper, not a validator (see
/// `fink_fat_ades::schema_validation` for actual ADES validation).
///
/// # Arguments
/// * `xml` — the XML text to tokenize.
///
/// # Return
/// The spans, in order; concatenating every span's text reproduces `xml`
/// exactly.
fn tokenize_xml(xml: &str) -> Vec<XmlSpan> {
    let bytes = xml.as_bytes();
    let len = bytes.len();
    let mut spans = Vec::new();
    let mut pos = 0;

    while pos < len {
        if bytes[pos] == b'<' {
            pos = tokenize_tag(xml, pos, &mut spans);
        } else {
            let start = pos;
            while pos < len && bytes[pos] != b'<' {
                pos += 1;
            }
            push_span(&mut spans, XmlSpanKind::Text, &xml[start..pos]);
        }
    }
    spans
}

/// Tokenizes one `<...>` construct (an element open/close/empty tag, or an
/// `<?...?>` declaration/processing instruction) starting at `xml.as_bytes()[pos] == b'<'`.
///
/// # Return
/// The byte position just past the construct's closing delimiter (or `xml.len()`
/// if the input ends before one is found — see [`tokenize_xml`]'s docs on
/// malformed input).
fn tokenize_tag(xml: &str, pos: usize, spans: &mut Vec<XmlSpan>) -> usize {
    let bytes = xml.as_bytes();
    let len = bytes.len();
    let mut i = pos;

    let is_special = i + 1 < len && matches!(bytes[i + 1], b'?' | b'/');
    let delim_end = if is_special { i + 2 } else { i + 1 };
    push_span(spans, XmlSpanKind::Punctuation, &xml[i..delim_end]);
    i = delim_end;

    let name_start = i;
    while i < len && !bytes[i].is_ascii_whitespace() && !matches!(bytes[i], b'>' | b'/' | b'?') {
        i += 1;
    }
    push_span(spans, XmlSpanKind::TagName, &xml[name_start..i]);

    loop {
        let ws_start = i;
        while i < len && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        push_span(spans, XmlSpanKind::Text, &xml[ws_start..i]);

        if i >= len {
            return i;
        }
        match bytes[i] {
            b'>' => {
                push_span(spans, XmlSpanKind::Punctuation, ">");
                return i + 1;
            }
            b'/' if i + 1 < len && bytes[i + 1] == b'>' => {
                push_span(spans, XmlSpanKind::Punctuation, "/>");
                return i + 2;
            }
            b'?' if i + 1 < len && bytes[i + 1] == b'>' => {
                push_span(spans, XmlSpanKind::Punctuation, "?>");
                return i + 2;
            }
            _ => {
                let attr_name_start = i;
                while i < len
                    && !matches!(bytes[i], b'=' | b'>' | b'/')
                    && !bytes[i].is_ascii_whitespace()
                {
                    i += 1;
                }
                if i == attr_name_start {
                    // Nothing recognizable at this position (malformed
                    // input) — bail out rather than looping forever.
                    push_span(spans, XmlSpanKind::Text, &xml[i..]);
                    return len;
                }
                push_span(spans, XmlSpanKind::AttrName, &xml[attr_name_start..i]);

                let ws2_start = i;
                while i < len && bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
                push_span(spans, XmlSpanKind::Text, &xml[ws2_start..i]);

                if i < len && bytes[i] == b'=' {
                    push_span(spans, XmlSpanKind::Punctuation, "=");
                    i += 1;

                    let ws3_start = i;
                    while i < len && bytes[i].is_ascii_whitespace() {
                        i += 1;
                    }
                    push_span(spans, XmlSpanKind::Text, &xml[ws3_start..i]);

                    if i < len && matches!(bytes[i], b'"' | b'\'') {
                        let quote = bytes[i];
                        let value_start = i;
                        i += 1;
                        while i < len && bytes[i] != quote {
                            i += 1;
                        }
                        if i < len {
                            i += 1; // consume the closing quote
                        }
                        push_span(spans, XmlSpanKind::AttrValue, &xml[value_start..i]);
                    }
                }
            }
        }
    }
}

/// daisyUI/Tailwind text-color class for one highlighting category.
fn xml_span_class(kind: XmlSpanKind) -> &'static str {
    match kind {
        XmlSpanKind::Punctuation => "text-base-content/50",
        XmlSpanKind::TagName => "text-primary font-semibold",
        XmlSpanKind::AttrName => "text-secondary",
        XmlSpanKind::AttrValue => "text-success",
        XmlSpanKind::Text => "",
    }
}

/// Modal showing one submission's full ADES XML, with copy/download
/// actions. **Always mounted** by its parent with `open` as a plain `bool`
/// prop — the same convention `lineage_page::ades_export_modal::AdesExportModal`
/// already uses in production, not to be "improved" here: the component
/// returns `rsx! {}` before declaring any hooks when `!open`, and the parent
/// flips its own signal back via `on_close`.
///
/// # Arguments
/// * `id` — the `mpc_submissions.id` row to show.
/// * `file_name` — the download filename (see [`ades_file_name`]).
/// * `open` — whether the modal is visible.
/// * `on_close` — called when the user dismisses the modal.
#[component]
pub fn AdesXmlModal(id: i64, file_name: String, open: bool, on_close: EventHandler<()>) -> Element {
    if !open {
        return rsx! {};
    }

    let xml_resource = use_resource(move || get_submission_xml(id));

    let copy_xml = move |_| {
        if let Some(Ok(xml)) = &*xml_resource.read() {
            let pretty =
                fink_fat_ades::xml_format::pretty_print_xml(xml).unwrap_or_else(|_| xml.clone());
            let eval = document::eval(
                "const data = await dioxus.recv();
                 await navigator.clipboard.writeText(data.text);",
            );
            let _ = eval.send(serde_json::json!({ "text": pretty }));
        }
    };

    let download_xml = {
        let file_name = file_name.clone();
        move |_| {
            if let Some(Ok(xml)) = &*xml_resource.read() {
                let eval = document::eval(
                    "const data = await dioxus.recv();
                     const blob = new Blob([data.xml], { type: 'application/xml' });
                     const url = URL.createObjectURL(blob);
                     const a = document.createElement('a');
                     a.href = url;
                     a.download = data.fileName;
                     document.body.appendChild(a);
                     a.click();
                     a.remove();
                     URL.revokeObjectURL(url);",
                );
                let _ = eval.send(serde_json::json!({ "xml": xml, "fileName": file_name }));
            }
        }
    };

    rsx! {
        div {
            class: "fixed inset-0 z-50 bg-black/40",
            onclick: move |_| on_close.call(()),
        }
        div { class: "fixed inset-0 z-50 flex items-center justify-center p-4",
            div {
                class: "card bg-base-100 shadow-xl w-full max-w-3xl max-h-[90vh] overflow-y-auto",
                onclick: move |evt| evt.stop_propagation(),
                div { class: "card-body gap-3",
                    div { class: "flex items-center justify-between",
                        h3 { class: "font-semibold", "ADES XML — {file_name}" }
                        button {
                            class: "btn btn-sm btn-circle btn-ghost",
                            r#type: "button",
                            onclick: move |_| on_close.call(()),
                            "✕"
                        }
                    }
                    match &*xml_resource.read() {
                        None => rsx! {
                            div { class: "flex justify-center py-6",
                                span { class: "loading loading-spinner loading-md" }
                            }
                        },
                        Some(Err(e)) => rsx! {
                            div { class: "alert alert-error", "Failed to load ADES XML: {e}" }
                        },
                        Some(Ok(xml)) => {
                            let pretty = fink_fat_ades::xml_format::pretty_print_xml(xml)
                                .unwrap_or_else(|_| xml.clone());
                            let spans = tokenize_xml(&pretty);
                            rsx! {
                                div { class: "flex gap-2",
                                    button {
                                        class: "btn btn-sm btn-outline",
                                        r#type: "button",
                                        onclick: copy_xml,
                                        "📋 Copy"
                                    }
                                    button {
                                        class: "btn btn-sm btn-outline",
                                        r#type: "button",
                                        onclick: download_xml,
                                        "⬇ Download"
                                    }
                                }
                                pre {
                                    class: "text-xs whitespace-pre-wrap bg-base-200 rounded p-3 overflow-x-auto",
                                    for (i, span) in spans.iter().enumerate() {
                                        span { key: "{i}", class: xml_span_class(span.kind), "{span.text}" }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ades_file_name_appends_the_expected_suffix() {
        assert_eq!(ades_file_name("FF2026abc"), "FF2026abc.ades.xml");
    }

    /// Concatenating every span's text must always reproduce the input
    /// exactly — the property that makes it safe to render spans as
    /// separately-styled elements without silently dropping or duplicating
    /// any character.
    fn assert_round_trips(xml: &str) {
        let spans = tokenize_xml(xml);
        let reconstructed: String = spans.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(reconstructed, xml);
    }

    #[test]
    fn tokenize_xml_round_trips_a_declaration_and_nested_elements() {
        assert_round_trips(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<ades version=\"2022\">\n  <obsBlock/>\n</ades>\n",
        );
    }

    #[test]
    fn tokenize_xml_round_trips_text_content() {
        assert_round_trips("<trkSub>elslilbd</trkSub>");
    }

    #[test]
    fn tokenize_xml_round_trips_malformed_input_without_panicking() {
        assert_round_trips("<a><b not-an-attr");
    }

    #[test]
    fn tokenize_xml_classifies_a_simple_element() {
        let spans = tokenize_xml("<trkSub>x</trkSub>");
        assert_eq!(
            spans,
            vec![
                XmlSpan {
                    kind: XmlSpanKind::Punctuation,
                    text: "<".to_string()
                },
                XmlSpan {
                    kind: XmlSpanKind::TagName,
                    text: "trkSub".to_string()
                },
                XmlSpan {
                    kind: XmlSpanKind::Punctuation,
                    text: ">".to_string()
                },
                XmlSpan {
                    kind: XmlSpanKind::Text,
                    text: "x".to_string()
                },
                XmlSpan {
                    kind: XmlSpanKind::Punctuation,
                    text: "</".to_string()
                },
                XmlSpan {
                    kind: XmlSpanKind::TagName,
                    text: "trkSub".to_string()
                },
                XmlSpan {
                    kind: XmlSpanKind::Punctuation,
                    text: ">".to_string()
                },
            ]
        );
    }

    #[test]
    fn tokenize_xml_classifies_a_self_closing_element_with_an_attribute() {
        let attr_spans = tokenize_xml("<a k=\"v\"/>");
        assert_eq!(
            attr_spans,
            vec![
                XmlSpan {
                    kind: XmlSpanKind::Punctuation,
                    text: "<".to_string()
                },
                XmlSpan {
                    kind: XmlSpanKind::TagName,
                    text: "a".to_string()
                },
                XmlSpan {
                    kind: XmlSpanKind::Text,
                    text: " ".to_string()
                },
                XmlSpan {
                    kind: XmlSpanKind::AttrName,
                    text: "k".to_string()
                },
                XmlSpan {
                    kind: XmlSpanKind::Punctuation,
                    text: "=".to_string()
                },
                XmlSpan {
                    kind: XmlSpanKind::AttrValue,
                    text: "\"v\"".to_string()
                },
                XmlSpan {
                    kind: XmlSpanKind::Punctuation,
                    text: "/>".to_string()
                },
            ]
        );
    }

    #[test]
    fn tokenize_xml_classifies_a_processing_declaration() {
        let spans = tokenize_xml("<?xml version=\"1.0\"?>");
        assert_eq!(
            spans[0],
            XmlSpan {
                kind: XmlSpanKind::Punctuation,
                text: "<?".to_string()
            }
        );
        assert_eq!(
            spans[1],
            XmlSpan {
                kind: XmlSpanKind::TagName,
                text: "xml".to_string()
            }
        );
        assert_eq!(
            spans.last().unwrap(),
            &XmlSpan {
                kind: XmlSpanKind::Punctuation,
                text: "?>".to_string()
            }
        );
    }
}
