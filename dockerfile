FROM rust:1.97.1-trixie

# System dependencies required to compile common Rust crates
# (openssl, pkg-config, etc.) + curl to download binaries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libssl-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# wasm target for Dioxus web build
RUN rustup target add wasm32-unknown-unknown

# cargo-binstall: installs precompiled binaries instead of building from source
RUN curl -L --proto '=https' --tlsv1.2 -sSf \
    https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh \
    | bash

# Dioxus CLI via binstall (fast, no compilation)
RUN cargo binstall dioxus-cli --no-confirm --force

# tailwindcss-extra (standalone Tailwind CLI + daisyUI)
# Adjust version/arch if needed: https://github.com/dobicinaitis/tailwind-cli-extra/releases
ARG TAILWIND_EXTRA_VERSION=v2.4.4
RUN curl -sLo /usr/local/bin/tailwindcss-extra \
    "https://github.com/dobicinaitis/tailwind-cli-extra/releases/download/${TAILWIND_EXTRA_VERSION}/tailwindcss-extra-linux-x64" \
    && chmod +x /usr/local/bin/tailwindcss-extra

# WORKDIR points to the workspace root
WORKDIR /app

# Copy the whole workspace (required so cargo can find the root Cargo.toml
# and resolve inherited workspace dependencies)
COPY . .

# Run commands from the actual crate directory
WORKDIR /app/crates/fink-fat-explorer

EXPOSE 8080

# Run the Tailwind watcher and the Dioxus server in parallel
CMD ["sh", "-c", "tailwindcss-extra -i ./tailwind.css -o ./assets/main.css --watch & dx serve --addr 0.0.0.0"]