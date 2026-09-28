//! Reformatting an already-serialized XML document for human display —
//! distinct from [`crate::xml`], which builds an [`crate::xml::AdesDocument`]
//! from scratch. [`ades_document_to_xml`](crate::xml::ades_document_to_xml)
//! produces compact, single-line XML (the exact bytes sent to MPC and stored
//! verbatim in `mpc_submissions.xml`); this module re-indents a *copy* of
//! that text for a viewer to show, without touching what was actually
//! submitted or stored.

use quick_xml::events::Event;
use quick_xml::reader::Reader;
use quick_xml::writer::Writer;
use std::io::Cursor;

/// Indentation width, in spaces, used by [`pretty_print_xml`].
const INDENT_WIDTH: usize = 2;

/// Re-indents an XML document for display, one element per line. Purely
/// cosmetic (whitespace-only) — does not change element/attribute content,
/// so it must never be used for anything that gets re-submitted or
/// persisted, only for showing a human a readable copy.
///
/// # Arguments
/// * `xml` — the XML text to re-indent (e.g. a stored `mpc_submissions.xml`
///   value).
///
/// # Return
/// The re-indented XML text.
///
/// # Errors
/// Returns a message describing the failure if `xml` isn't well-formed XML,
/// or isn't valid UTF-8 once rewritten (should not happen for XML that
/// parsed successfully in the first place).
pub fn pretty_print_xml(xml: &str) -> Result<String, String> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut writer = Writer::new_with_indent(Cursor::new(Vec::new()), b' ', INDENT_WIDTH);

    loop {
        match reader.read_event().map_err(|e| e.to_string())? {
            Event::Eof => break,
            event => writer.write_event(event).map_err(|e| e.to_string())?,
        }
    }

    String::from_utf8(writer.into_inner().into_inner()).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretty_print_xml_indents_nested_elements() {
        let compact = "<a><b><c>text</c></b></a>";
        let pretty = pretty_print_xml(compact).unwrap();
        assert_eq!(pretty, "<a>\n  <b>\n    <c>text</c>\n  </b>\n</a>");
    }

    #[test]
    fn pretty_print_xml_keeps_attributes_and_declaration() {
        let compact =
            r#"<?xml version="1.0" encoding="UTF-8"?><ades version="2022"><obsBlock/></ades>"#;
        let pretty = pretty_print_xml(compact).unwrap();
        assert!(pretty.starts_with(r#"<?xml version="1.0" encoding="UTF-8"?>"#));
        assert!(pretty.contains("<ades version=\"2022\">"));
        assert!(pretty.contains("  <obsBlock/>"));
    }

    #[test]
    fn pretty_print_xml_rejects_malformed_input() {
        assert!(pretty_print_xml("<a><b></a>").is_err());
    }

    #[test]
    fn pretty_print_xml_round_trips_a_real_ades_document() {
        let compact = r#"<?xml version="1.0" encoding="UTF-8"?>
<ades version="2022"><obsBlock><obsContext><observatory><mpcCode>X05</mpcCode></observatory><submitter><name>R Le Montagner</name></submitter><measurers><name>Vera C. Rubin Observatory (LSST)</name></measurers><telescope><design>Reflector</design><aperture>8.4</aperture><detector>CCD Mosaic</detector></telescope></obsContext><obsData><optical><trkSub>elslilbd</trkSub><mode>CCD</mode><stn>X05</stn><obsTime>2025-12-19T07:46:01.413116Z</obsTime><ra>151.848719</ra><dec>2.477959</dec><astCat>Gaia2</astCat><mag>23.80</mag><band>r</band></optical></obsData></obsBlock></ades>
"#;
        let pretty = pretty_print_xml(compact).unwrap();
        assert!(pretty.contains("<trkSub>elslilbd</trkSub>"));
        assert!(pretty.lines().count() > 5);
    }
}
