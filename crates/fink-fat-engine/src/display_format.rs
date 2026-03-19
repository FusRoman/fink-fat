/// Indent a multi-line string by a fixed number of spaces.
///
/// This helper is primarily intended for **pretty-printing** complex objects
/// (e.g. nested `Display` implementations) in a human-readable, indented form.
///
/// Each line of the input string `s` is prefixed with `spaces` ASCII spaces.
/// Line breaks are preserved exactly as in the original string.
///
/// This function performs **no trimming or normalization**:
/// - empty lines are kept,
/// - leading/trailing whitespace in `s` is preserved,
/// - indentation is applied uniformly to all lines.
///
/// Parameters
/// ----------
/// s : &str
///     Input string, potentially spanning multiple lines.
/// spaces : usize
///     Number of spaces to prepend to each line.
///
/// Returns
/// -------
/// String
///     A new string where each line of `s` is indented by `spaces` spaces.
///
/// Notes
/// -----
/// This utility is intentionally simple and allocation-friendly:
/// - it allocates a single padding string,
/// - it allocates a new `String` for the final result.
///
/// It is well-suited for formatting diagnostic output, CLI reports,
/// and structured `Display` implementations, but should not be used
/// in performance-critical inner loops.
pub fn indent_block(s: &str, spaces: usize) -> String {
    let pad = " ".repeat(spaces);
    s.lines()
        .map(|line| format!("{pad}{line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Format a 2D vector as "(x, y)" with a compact scientific style.
pub fn fmt_vec2(v: [f64; 2]) -> String {
    // Keep it short but stable for QA logs.
    // Use scientific notation: good for small rad values.
    format!("({:.6e}, {:.6e})", v[0], v[1])
}

/// Format a 2x2 matrix as "[[a, b], [c, d]]" with compact scientific notation.
pub fn fmt_mat2(m: [[f64; 2]; 2]) -> String {
    format!(
        "[[{:.6e}, {:.6e}], [{:.6e}, {:.6e}]]",
        m[0][0], m[0][1], m[1][0], m[1][1]
    )
}
