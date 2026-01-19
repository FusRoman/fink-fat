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
