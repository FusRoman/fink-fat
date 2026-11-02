//! Pure query model of the "Submission" page: which submissions the user
//! wants to see (filters, date range, sort, page) and how that intent maps to
//! SQL.
//!
//! Everything here is free of I/O and of any server-only dependency, so it
//! compiles for the wasm client too (the page builds a [`SubmissionQuery`] from
//! its signals and ships it to the `list_submissions` server function) and is
//! unit-tested with a plain `cargo test`. The database layer
//! (`data.rs`) only executes the [`WhereClause`] produced here.
//!
//! (Items only the server needs are gated on the `server` feature so the wasm
//! build carries no dead code.)
//!
//! Safety of the generated SQL: user-controlled values (search text, dates)
//! never reach the SQL string — they are collected as bind parameters and the
//! string only contains `$n` placeholders and constant fragments.
//! Endpoint and verdict filters are closed enums, so an unknown value cannot
//! even be deserialized.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::homepage::interaction::SortDirection;

/// Number of submissions shown per page.
pub const SUBMISSIONS_PAGE_SIZE: i64 = 25;

/// Which MPC endpoint(s) the list is restricted to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum EndpointFilter {
    /// No restriction.
    #[default]
    All,
    /// Only `test` (MPC sandbox) submissions.
    Test,
    /// Only `production` submissions.
    Production,
}

impl EndpointFilter {
    /// Every variant, in display order.
    pub const ALL: [EndpointFilter; 3] = [Self::All, Self::Test, Self::Production];

    /// Human-readable label of the filter button.
    ///
    /// # Return
    /// The label.
    pub fn label(self) -> &'static str {
        match self {
            Self::All => "All",
            Self::Test => "Test",
            Self::Production => "Production",
        }
    }

    #[cfg(any(feature = "server", test))]
    /// The `mpc_submissions.endpoint` value this filter selects.
    ///
    /// # Return
    /// `None` for [`EndpointFilter::All`] (no restriction), otherwise the
    /// column value.
    pub fn column_value(self) -> Option<&'static str> {
        match self {
            Self::All => None,
            Self::Test => Some("test"),
            Self::Production => Some("production"),
        }
    }
}

/// Which verdict(s) the list is restricted to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerdictFilter {
    /// No restriction.
    #[default]
    All,
    /// Only `pending` submissions.
    Pending,
    /// Only `accepted` submissions.
    Accepted,
    /// Only `rejected` submissions.
    Rejected,
    /// Only `error` submissions (MPC never acknowledged, or id not found).
    Error,
}

impl VerdictFilter {
    /// Every variant, in display order.
    pub const ALL: [VerdictFilter; 5] = [
        Self::All,
        Self::Pending,
        Self::Accepted,
        Self::Rejected,
        Self::Error,
    ];

    /// Human-readable label of the filter button.
    ///
    /// # Return
    /// The label.
    pub fn label(self) -> &'static str {
        match self {
            Self::All => "All",
            Self::Pending => "Pending",
            Self::Accepted => "Accepted",
            Self::Rejected => "Rejected",
            Self::Error => "Error",
        }
    }

    #[cfg(any(feature = "server", test))]
    /// The `mpc_submissions.verdict` value this filter selects.
    ///
    /// # Return
    /// `None` for [`VerdictFilter::All`] (no restriction), otherwise the
    /// column value.
    pub fn column_value(self) -> Option<&'static str> {
        match self {
            Self::All => None,
            Self::Pending => Some("pending"),
            Self::Accepted => Some("accepted"),
            Self::Rejected => Some("rejected"),
            Self::Error => Some("error"),
        }
    }
}

/// A calendar day, `YYYY-MM-DD`, validated. Ordered chronologically.
///
/// Kept independent of `chrono` (server-only) so the wasm client can validate
/// and order the dates coming back from the calendar widget.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct IsoDate {
    year: i32,
    month: u32,
    day: u32,
}

impl IsoDate {
    /// Parses a strict `YYYY-MM-DD` string.
    ///
    /// # Arguments
    /// * `text` — the candidate date.
    ///
    /// # Return
    /// The date, or `None` if the shape is wrong or the day does not exist
    /// (month 13, 31 February, non-leap 29 February, ...).
    pub fn parse(text: &str) -> Option<Self> {
        let mut parts = text.split('-');
        let (year, month, day) = (parts.next()?, parts.next()?, parts.next()?);
        if parts.next().is_some() || year.len() != 4 || month.len() != 2 || day.len() != 2 {
            return None;
        }
        let all_digits = |s: &str| s.bytes().all(|b| b.is_ascii_digit());
        if !(all_digits(year) && all_digits(month) && all_digits(day)) {
            return None;
        }
        let date = Self {
            year: year.parse().ok()?,
            month: month.parse().ok()?,
            day: day.parse().ok()?,
        };
        let day_exists = (1..=days_in_month(date.year, date.month)).contains(&date.day);
        day_exists.then_some(date)
    }
}

impl fmt::Display for IsoDate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

/// Whether `year` is a Gregorian leap year.
///
/// # Arguments
/// * `year` — the year.
///
/// # Return
/// `true` for a leap year.
fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Number of days in a month.
///
/// # Arguments
/// * `year` — the year (matters for February).
/// * `month` — the month, `1..=12`.
///
/// # Return
/// The number of days; `0` for a month outside `1..=12`.
fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

/// An inclusive range of calendar days (UTC), `start <= end` by construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DateRange {
    start: IsoDate,
    end: IsoDate,
}

impl DateRange {
    /// Builds a range from two days in any order.
    ///
    /// # Arguments
    /// * `a`, `b` — the two bounds; the earlier becomes the start.
    ///
    /// # Return
    /// The normalized range.
    pub fn new(a: IsoDate, b: IsoDate) -> Self {
        Self {
            start: a.min(b),
            end: a.max(b),
        }
    }

    /// Builds a range from the dates reported by the calendar widget.
    ///
    /// The widget reports either only the two bounds or every day of the
    /// range depending on its mode/version, so the range is the min and max
    /// of whatever valid dates are present.
    ///
    /// # Arguments
    /// * `selected` — the `YYYY-MM-DD` strings reported by the widget.
    ///
    /// # Return
    /// The range, or `None` if no entry is a valid date. A single date yields
    /// a one-day range.
    pub fn from_selected<S: AsRef<str>>(selected: &[S]) -> Option<Self> {
        let mut dates = selected.iter().filter_map(|s| IsoDate::parse(s.as_ref()));
        let first = dates.next()?;
        let (min, max) = dates.fold((first, first), |(lo, hi), d| (lo.min(d), hi.max(d)));
        Some(Self::new(min, max))
    }

    #[cfg(any(feature = "server", test))]
    /// First day of the range (inclusive).
    ///
    /// # Return
    /// The start day.
    pub fn start(&self) -> IsoDate {
        self.start
    }

    #[cfg(any(feature = "server", test))]
    /// Last day of the range (inclusive).
    ///
    /// # Return
    /// The end day.
    pub fn end(&self) -> IsoDate {
        self.end
    }

    /// Label of the picker button.
    ///
    /// # Return
    /// `"2026-09-29"` for a one-day range, `"2026-09-01 → 2026-09-29"`
    /// otherwise.
    pub fn label(&self) -> String {
        if self.start == self.end {
            self.start.to_string()
        } else {
            format!("{} → {}", self.start, self.end)
        }
    }
}

/// Everything the user chose to see: filters, date range, search, sort, page.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubmissionQuery {
    pub endpoint: EndpointFilter,
    pub verdict: VerdictFilter,
    /// Free text matched (case-insensitively, as a substring) against the
    /// lineage designation and the MPC submission id. Blank means no filter.
    pub search: String,
    /// Restricts `submitted_at` to these days (UTC, inclusive).
    pub date_range: Option<DateRange>,
    /// Direction of the `submitted_at` sort.
    pub sort: SortDirection,
    /// Zero-based page index.
    pub page: i64,
}

impl Default for SubmissionQuery {
    fn default() -> Self {
        Self {
            endpoint: EndpointFilter::default(),
            verdict: VerdictFilter::default(),
            search: String::new(),
            date_range: None,
            sort: SortDirection::Desc,
            page: 0,
        }
    }
}

impl SubmissionQuery {
    /// Whether any filter narrows the list (sort and page do not count).
    ///
    /// # Return
    /// `true` if at least one of endpoint, verdict, search or date range is
    /// set.
    pub fn has_active_filters(&self) -> bool {
        self.endpoint != EndpointFilter::All
            || self.verdict != VerdictFilter::All
            || !self.search.trim().is_empty()
            || self.date_range.is_some()
    }
}

#[cfg(any(feature = "server", test))]
/// A `WHERE` clause with its bind parameters, ready to be executed.
///
/// All parameters are text: dates are cast in SQL (`$n::date`), so the
/// database layer binds every entry of [`Self::binds`] as a string.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WhereClause {
    /// The clause including the leading `WHERE`, or empty when unfiltered.
    pub sql: String,
    /// Values for `$1..$n`, in order.
    pub binds: Vec<String>,
}

#[cfg(any(feature = "server", test))]
impl WhereClause {
    /// The placeholder number the next extra bind (e.g. `LIMIT`) should use.
    ///
    /// # Return
    /// `binds.len() + 1`.
    pub fn next_placeholder(&self) -> usize {
        self.binds.len() + 1
    }
}

#[cfg(any(feature = "server", test))]
/// Escapes `%`, `_` and `\` so user text is matched literally by `ILIKE`.
///
/// # Arguments
/// * `text` — raw user text.
///
/// # Return
/// The escaped text (the escape character is the SQL default, `\`).
pub fn escape_like(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for c in text.chars() {
        if matches!(c, '\\' | '%' | '_') {
            escaped.push('\\');
        }
        escaped.push(c);
    }
    escaped
}

#[cfg(any(feature = "server", test))]
/// Turns the search box text into an `ILIKE` pattern.
///
/// # Arguments
/// * `search` — raw search box text.
///
/// # Return
/// `%escaped%`, or `None` when the trimmed text is empty.
pub fn like_pattern(search: &str) -> Option<String> {
    let trimmed = search.trim();
    (!trimmed.is_empty()).then(|| format!("%{}%", escape_like(trimmed)))
}

#[cfg(any(feature = "server", test))]
/// Builds the `WHERE` clause of a [`SubmissionQuery`].
///
/// # Arguments
/// * `query` — the user's query; sort and page are ignored here.
///
/// # Return
/// The clause and its binds. Conditions are AND-ed in a fixed order:
/// endpoint, verdict, search, date start, date end (end is exclusive on the
/// following midnight so the last day is fully included).
pub fn build_where_clause(query: &SubmissionQuery) -> WhereClause {
    let mut conditions: Vec<String> = Vec::new();
    let mut binds: Vec<String> = Vec::new();

    if let Some(endpoint) = query.endpoint.column_value() {
        binds.push(endpoint.to_string());
        conditions.push(format!("endpoint = ${}", binds.len()));
    }
    if let Some(verdict) = query.verdict.column_value() {
        binds.push(verdict.to_string());
        conditions.push(format!("verdict = ${}", binds.len()));
    }
    if let Some(pattern) = like_pattern(&query.search) {
        binds.push(pattern);
        let n = binds.len();
        conditions.push(format!(
            "(lineage_designation ILIKE ${n} OR submission_id ILIKE ${n})"
        ));
    }
    if let Some(range) = query.date_range {
        binds.push(range.start().to_string());
        conditions.push(format!(
            "submitted_at >= (${}::date)::timestamp AT TIME ZONE 'UTC'",
            binds.len()
        ));
        binds.push(range.end().to_string());
        conditions.push(format!(
            "submitted_at < (${}::date + 1)::timestamp AT TIME ZONE 'UTC'",
            binds.len()
        ));
    }

    let sql = if conditions.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", conditions.join(" AND "))
    };
    WhereClause { sql, binds }
}

#[cfg(any(feature = "server", test))]
/// The `ORDER BY` clause for a sort direction; `id` breaks ties so pages are
/// stable when several submissions share a timestamp.
///
/// # Arguments
/// * `sort` — direction of the `submitted_at` sort.
///
/// # Return
/// A constant SQL fragment.
pub fn order_by_clause(sort: SortDirection) -> &'static str {
    match sort {
        SortDirection::Asc => "ORDER BY submitted_at ASC, id ASC",
        SortDirection::Desc => "ORDER BY submitted_at DESC, id DESC",
    }
}

/// Number of pages needed for `total` rows.
///
/// # Arguments
/// * `total` — number of matching rows (negative is treated as 0).
/// * `page_size` — rows per page, must be positive.
///
/// # Return
/// The page count, at least 1 so an empty list still has "page 1 of 1".
pub fn total_pages(total: i64, page_size: i64) -> i64 {
    let page_size = page_size.max(1);
    ((total.max(0) + page_size - 1) / page_size).max(1)
}

#[cfg(any(feature = "server", test))]
/// Clamps a requested page into `0..total_pages`.
///
/// # Arguments
/// * `page` — requested zero-based page.
/// * `total` — number of matching rows.
/// * `page_size` — rows per page.
///
/// # Return
/// A valid zero-based page index.
pub fn clamp_page(page: i64, total: i64, page_size: i64) -> i64 {
    page.clamp(0, total_pages(total, page_size) - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(text: &str) -> IsoDate {
        IsoDate::parse(text).unwrap()
    }

    #[test]
    fn endpoint_filter_maps_to_column_values() {
        assert_eq!(EndpointFilter::All.column_value(), None);
        assert_eq!(EndpointFilter::Test.column_value(), Some("test"));
        assert_eq!(
            EndpointFilter::Production.column_value(),
            Some("production")
        );
        assert_eq!(EndpointFilter::ALL.len(), 3);
        assert_eq!(EndpointFilter::Production.label(), "Production");
    }

    #[test]
    fn verdict_filter_maps_to_column_values() {
        let values: Vec<_> = VerdictFilter::ALL
            .iter()
            .map(|v| v.column_value())
            .collect();
        assert_eq!(
            values,
            [
                None,
                Some("pending"),
                Some("accepted"),
                Some("rejected"),
                Some("error")
            ]
        );
        assert_eq!(VerdictFilter::Rejected.label(), "Rejected");
    }

    #[test]
    fn unknown_filter_value_fails_to_deserialize() {
        assert!(serde_json::from_str::<EndpointFilter>("\"Staging\"").is_err());
        assert!(serde_json::from_str::<VerdictFilter>("\"Maybe\"").is_err());
    }

    #[test]
    fn iso_date_accepts_valid_days() {
        assert_eq!(date("2026-09-29").to_string(), "2026-09-29");
        assert!(IsoDate::parse("2024-02-29").is_some());
        assert!(IsoDate::parse("2000-02-29").is_some());
    }

    #[test]
    fn iso_date_rejects_invalid_days() {
        for bad in [
            "",
            "2026",
            "2026-09",
            "2026-09-29-01",
            "2026-13-01",
            "2026-00-10",
            "2026-04-31",
            "2023-02-29",
            "1900-02-29",
            "2026-9-29",
            "26-09-29",
            "2026-09-2x",
            "+026-09-29",
            "2026-09-00",
        ] {
            assert!(IsoDate::parse(bad).is_none(), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn iso_dates_order_chronologically() {
        assert!(date("2025-12-31") < date("2026-01-01"));
        assert!(date("2026-01-09") < date("2026-01-10"));
    }

    #[test]
    fn date_range_swaps_reversed_bounds() {
        let range = DateRange::new(date("2026-09-30"), date("2026-09-01"));
        assert_eq!(range.start(), date("2026-09-01"));
        assert_eq!(range.end(), date("2026-09-30"));
        assert_eq!(range.label(), "2026-09-01 → 2026-09-30");
    }

    #[test]
    fn date_range_single_day_label() {
        let range = DateRange::new(date("2026-09-29"), date("2026-09-29"));
        assert_eq!(range.label(), "2026-09-29");
    }

    #[test]
    fn date_range_from_selected_uses_min_and_max() {
        let range = DateRange::from_selected(&["2026-09-03", "2026-09-01", "2026-09-02"]).unwrap();
        assert_eq!(range.start(), date("2026-09-01"));
        assert_eq!(range.end(), date("2026-09-03"));
    }

    #[test]
    fn date_range_from_selected_ignores_invalid_entries() {
        let range = DateRange::from_selected(&["garbage", "2026-09-05"]).unwrap();
        assert_eq!(range.label(), "2026-09-05");
        assert!(DateRange::from_selected(&["garbage"]).is_none());
        assert!(DateRange::from_selected::<&str>(&[]).is_none());
    }

    #[test]
    fn escape_like_neutralizes_wildcards() {
        assert_eq!(escape_like("50%_off\\"), "50\\%\\_off\\\\");
        assert_eq!(escape_like("FF2026abc"), "FF2026abc");
    }

    #[test]
    fn like_pattern_trims_and_wraps() {
        assert_eq!(like_pattern("  ab_c "), Some("%ab\\_c%".to_string()));
        assert_eq!(like_pattern("   "), None);
        assert_eq!(like_pattern(""), None);
    }

    #[test]
    fn default_query_has_no_where_clause() {
        let clause = build_where_clause(&SubmissionQuery::default());
        assert_eq!(clause.sql, "");
        assert!(clause.binds.is_empty());
        assert_eq!(clause.next_placeholder(), 1);
        assert!(!SubmissionQuery::default().has_active_filters());
    }

    #[test]
    fn where_clause_numbers_placeholders_in_order() {
        let query = SubmissionQuery {
            endpoint: EndpointFilter::Production,
            verdict: VerdictFilter::Accepted,
            search: "FF2026".to_string(),
            date_range: Some(DateRange::new(date("2026-09-01"), date("2026-09-29"))),
            ..SubmissionQuery::default()
        };
        let clause = build_where_clause(&query);
        assert_eq!(
            clause.binds,
            [
                "production",
                "accepted",
                "%FF2026%",
                "2026-09-01",
                "2026-09-29"
            ]
        );
        assert_eq!(
            clause.sql,
            "WHERE endpoint = $1 AND verdict = $2 \
             AND (lineage_designation ILIKE $3 OR submission_id ILIKE $3) \
             AND submitted_at >= ($4::date)::timestamp AT TIME ZONE 'UTC' \
             AND submitted_at < ($5::date + 1)::timestamp AT TIME ZONE 'UTC'"
        );
        assert_eq!(clause.next_placeholder(), 6);
        assert!(query.has_active_filters());
    }

    #[test]
    fn where_clause_skips_unset_filters() {
        let query = SubmissionQuery {
            verdict: VerdictFilter::Pending,
            ..SubmissionQuery::default()
        };
        let clause = build_where_clause(&query);
        assert_eq!(clause.sql, "WHERE verdict = $1");
        assert_eq!(clause.binds, ["pending"]);
    }

    #[test]
    fn where_clause_never_inlines_user_text() {
        let query = SubmissionQuery {
            search: "'; DROP TABLE mpc_submissions; --".to_string(),
            ..SubmissionQuery::default()
        };
        let clause = build_where_clause(&query);
        assert!(!clause.sql.contains("DROP"));
        assert_eq!(clause.binds.len(), 1);
    }

    #[test]
    fn blank_search_is_not_an_active_filter() {
        let query = SubmissionQuery {
            search: "   ".to_string(),
            ..SubmissionQuery::default()
        };
        assert!(!query.has_active_filters());
        assert_eq!(build_where_clause(&query).sql, "");
    }

    #[test]
    fn order_by_clause_follows_direction_with_tiebreak() {
        assert_eq!(
            order_by_clause(SortDirection::Desc),
            "ORDER BY submitted_at DESC, id DESC"
        );
        assert_eq!(
            order_by_clause(SortDirection::Asc),
            "ORDER BY submitted_at ASC, id ASC"
        );
    }

    #[test]
    fn total_pages_rounds_up_and_is_at_least_one() {
        assert_eq!(total_pages(0, 25), 1);
        assert_eq!(total_pages(1, 25), 1);
        assert_eq!(total_pages(25, 25), 1);
        assert_eq!(total_pages(26, 25), 2);
        assert_eq!(total_pages(50, 25), 2);
        assert_eq!(total_pages(-3, 25), 1);
    }

    #[test]
    fn clamp_page_stays_within_bounds() {
        assert_eq!(clamp_page(-4, 100, 25), 0);
        assert_eq!(clamp_page(2, 100, 25), 2);
        assert_eq!(clamp_page(9, 100, 25), 3);
        assert_eq!(clamp_page(5, 0, 25), 0);
    }

    #[test]
    fn query_round_trips_through_json() {
        let query = SubmissionQuery {
            endpoint: EndpointFilter::Test,
            date_range: Some(DateRange::new(date("2026-01-01"), date("2026-01-02"))),
            page: 3,
            ..SubmissionQuery::default()
        };
        let json = serde_json::to_string(&query).unwrap();
        assert_eq!(
            serde_json::from_str::<SubmissionQuery>(&json).unwrap(),
            query
        );
    }
}
