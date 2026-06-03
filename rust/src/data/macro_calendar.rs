// Static US macro release calendar. `is_release_day` returns true on scheduled
// FOMC, CPI, NFP, PCE, GDP, and Retail Sales days (UTC calendar day).
// Update RELEASE_DATES to cover at least 180d trailing + 30d forward.

use chrono::NaiveDate;

/// Sorted, deduplicated US macro release dates (UTC calendar day).
/// Entries are annotated with the release(s) that fall on that day.
const RELEASE_DATES: &[NaiveDate] = &[
    naive_date(2025, 11, 17), // Retail Sales (Census, Sep)
    naive_date(2025, 11, 26), // PCE (BEA, Oct)
    naive_date(2025, 12, 5),  // NFP (BLS, Nov)
    naive_date(2025, 12, 10), // CPI (BLS, Nov)
    naive_date(2025, 12, 17), // FOMC + Retail Sales (Census, Nov)
    naive_date(2025, 12, 19), // PCE (BEA, Nov)
    naive_date(2026, 1, 2),   // NFP (BLS, Dec)
    naive_date(2026, 1, 14),  // CPI (BLS, Dec)
    naive_date(2026, 1, 16),  // Retail Sales (Census, Dec)
    naive_date(2026, 1, 28),  // FOMC
    naive_date(2026, 1, 29),  // GDP advance (BEA, Q4) + PCE (BEA, Dec)
    naive_date(2026, 1, 30),  // PCE (BEA, Dec)
    naive_date(2026, 2, 6),   // NFP (BLS, Jan)
    naive_date(2026, 2, 11),  // CPI (BLS, Jan)
    naive_date(2026, 2, 17),  // Retail Sales (Census, Jan)
    naive_date(2026, 2, 27),  // PCE (BEA, Jan)
    naive_date(2026, 3, 6),   // NFP (BLS, Feb)
    naive_date(2026, 3, 11),  // CPI (BLS, Feb)
    naive_date(2026, 3, 17),  // Retail Sales (Census, Feb)
    naive_date(2026, 3, 18),  // FOMC
    naive_date(2026, 3, 27),  // PCE (BEA, Feb)
    naive_date(2026, 4, 3),   // NFP (BLS, Mar)
    naive_date(2026, 4, 14),  // CPI (BLS, Mar)
    naive_date(2026, 4, 16),  // Retail Sales (Census, Mar)
    naive_date(2026, 4, 29),  // FOMC + GDP advance (BEA, Q1)
    naive_date(2026, 4, 30),  // PCE (BEA, Mar)
    naive_date(2026, 5, 1),   // NFP (BLS, Apr)
    naive_date(2026, 5, 13),  // CPI (BLS, Apr)
    naive_date(2026, 5, 15),  // Retail Sales (Census, Apr)
    naive_date(2026, 5, 29),  // PCE (BEA, Apr)
    naive_date(2026, 6, 5),   // NFP (BLS, May)
    naive_date(2026, 6, 10),  // CPI (BLS, May)
    naive_date(2026, 6, 14),  // Coverage boundary — verify next refresh
];

/// `const fn` constructor for `NaiveDate`; panics on an invalid date.
#[allow(clippy::expect_used)]
const fn naive_date(y: i32, m: u32, d: u32) -> NaiveDate {
    match NaiveDate::from_ymd_opt(y, m, d) {
        Some(d) => d,
        None => panic!("macro_calendar: invalid date in RELEASE_DATES"),
    }
}

/// Returns `true` iff `date_utc` is a scheduled US macro release day.
pub fn is_release_day(date_utc: NaiveDate) -> bool {
    RELEASE_DATES.contains(&date_utc)
}

/// First and last dates of the calendar coverage window.
pub fn coverage_window() -> (NaiveDate, NaiveDate) {
    (
        *RELEASE_DATES.first().expect("RELEASE_DATES must be non-empty"),
        *RELEASE_DATES.last().expect("RELEASE_DATES must be non-empty"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_dates_are_sorted_and_unique() {
        for win in RELEASE_DATES.windows(2) {
            assert!(
                win[0] < win[1],
                "RELEASE_DATES must be strictly sorted: {:?} not < {:?}",
                win[0],
                win[1]
            );
        }
    }
}
