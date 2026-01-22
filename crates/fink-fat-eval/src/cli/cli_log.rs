#[macro_export]
macro_rules! log {
    ($cli:expr, $($arg:tt)*) => {{
        if !$cli.quiet && $cli.verbose >= 1 {
            println!($($arg)*);
        }
    }};
}

#[macro_export]
macro_rules! log2 {
    ($cli:expr, $($arg:tt)*) => {{
        if !$cli.quiet && $cli.verbose >= 2 {
            println!($($arg)*);
        }
    }};
}

#[macro_export]
macro_rules! log_section {
    ($cli:expr, $title:expr) => {{
        if !$cli.quiet {
            println!("\n=============== {} ==================\n", $title);
        }
    }};
}

#[macro_export]
macro_rules! log_timing {
    ($cli:expr, $label:expr, $dur:expr) => {{
        if !$cli.quiet && $cli.verbose >= 1 {
            println!("{}: {:.1?}", $label, $dur);
        }
    }};
}

#[macro_export]
macro_rules! buflog {
    ($buf:expr, $cli:expr, $($arg:tt)*) => {{
        if !$cli.quiet && $cli.verbose >= 1 {
            use std::fmt::Write as _;
            let _ = writeln!($buf, $($arg)*);
        }
    }};
}

#[macro_export]
macro_rules! buflog2 {
    ($buf:expr, $cli:expr, $($arg:tt)*) => {{
        if !$cli.quiet && $cli.verbose >= 2 {
            use std::fmt::Write as _;
            let _ = writeln!($buf, $($arg)*);
        }
    }};
}

#[macro_export]
macro_rules! buflog_section {
    ($buf:expr, $cli:expr, $title:expr) => {{
        if !$cli.quiet {
            use std::fmt::Write as _;
            let _ = writeln!($buf, "\n=============== {} ==================\n", $title);
        }
    }};
}

#[macro_export]
macro_rules! buflog_timing {
    ($buf:expr, $cli:expr, $label:expr, $dur:expr) => {{
        if !$cli.quiet && $cli.verbose >= 1 {
            use std::fmt::Write as _;
            let _ = writeln!($buf, "{}: {:.1?}", $label, $dur);
        }
    }};
}
