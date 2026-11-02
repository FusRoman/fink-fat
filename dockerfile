# syntax=docker/dockerfile:1
#
# fink-fat-explorer images. One file, two runnable targets that serve the
# same application and differ only in how it is compiled and run:
#
#   dev   `dx serve` against the bind-mounted source tree: debug profile,
#         hot reload, Tailwind in watch mode. For local development and tests.
#   prod  `dx bundle --release` (optimized server binary + wasm-opt'd client),
#         copied into a slim runtime image with no toolchain and no source.
#
# Select one with `docker build --target <dev|prod>` or through the matching
# `docker-compose.yml` service. `prod` is the last stage, so a plain
# `docker build .` produces the production image.

# ---------------------------------------------------------------------------
# toolchain: everything needed to compile the explorer, shared by `dev` and
# `builder` so both always use the exact same Rust/dx/Tailwind versions.
# ---------------------------------------------------------------------------
FROM rust:1.97.1-trixie AS toolchain

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

# Dioxus CLI via binstall (fast, no compilation). Pinned so the CLI always
# matches the `dioxus` crate version in Cargo.lock.
ARG DIOXUS_CLI_VERSION=0.7.10
RUN cargo binstall dioxus-cli@${DIOXUS_CLI_VERSION} --no-confirm --force

# tailwindcss-extra (standalone Tailwind CLI + daisyUI)
# Adjust version/arch if needed: https://github.com/dobicinaitis/tailwind-cli-extra/releases
ARG TAILWIND_EXTRA_VERSION=v2.4.4
RUN curl -sLo /usr/local/bin/tailwindcss-extra \
    "https://github.com/dobicinaitis/tailwind-cli-extra/releases/download/${TAILWIND_EXTRA_VERSION}/tailwindcss-extra-linux-x64" \
    && chmod +x /usr/local/bin/tailwindcss-extra

# WORKDIR points to the workspace root
WORKDIR /app

# Copy only what's needed to build the workspace (root Cargo.toml/Cargo.lock
# are required so cargo can resolve inherited workspace dependencies). This
# is deliberately explicit, not `COPY . .`, so unrelated large data files
# added anywhere in the repo never end up in the image.
COPY Cargo.toml Cargo.lock ./
COPY .cargo ./.cargo
COPY LICENSE README.md katex-header.html ./
COPY src ./src
COPY crates ./crates

# ---------------------------------------------------------------------------
# dev: `dx serve` in debug mode. `docker-compose.yml` bind-mounts the
# workspace over /app, so the sources copied above are only a fallback.
# ---------------------------------------------------------------------------
FROM toolchain AS dev

# Run commands from the actual crate directory
WORKDIR /app/crates/fink-fat-explorer

EXPOSE 8080

# Run the Tailwind watcher and the Dioxus server in parallel
CMD ["sh", "-c", "tailwindcss-extra -i ./tailwind.css -o ./assets/main.css --watch </dev/null > /tmp/tailwind.log 2>&1; echo \"tailwind exited with code $?\" >> /tmp/tailwind.log & dx serve --addr 0.0.0.0"]

# ---------------------------------------------------------------------------
# builder: one-shot optimized build of the fullstack app.
# ---------------------------------------------------------------------------
FROM toolchain AS builder

WORKDIR /app/crates/fink-fat-explorer

# Regenerate the stylesheet from `tailwind.css` once, minified, so the image
# never ships a stale `assets/main.css` left over from a dev session.
RUN tailwindcss-extra -i ./tailwind.css -o ./assets/main.css --minify

# `--release` uses the workspace `[profile.release]` (thin LTO, one codegen
# unit) for the server and runs wasm-opt on the client. Debug symbols are
# turned off explicitly: dx keeps them in the wasm by default, which only
# makes the bundle every visitor downloads larger.
#
# The cargo registry and target directory are BuildKit cache mounts, so a
# rebuild only recompiles what changed. Cache mounts are not part of the
# image, hence the separate `--out-dir` outside of them.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    dx bundle --web --release --debug-symbols false --out-dir /app/dist

# ---------------------------------------------------------------------------
# prod: runtime-only image. Holds the server binary, its `public/` client
# bundle and the engine config — no compiler, no sources.
# ---------------------------------------------------------------------------
FROM debian:trixie-slim AS prod

# libssl: reqwest's default native-tls backend (Fink, MPC, Skybot calls).
# ca-certificates: HTTPS to those services and to the ephemeris download.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3t64 \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged runtime user. Its cache directory is where outfit stores the
# downloaded JPL ephemeris and photom the MPC observatory list; it is created
# here so a named volume mounted on it inherits this ownership.
RUN useradd --create-home --uid 10001 explorer \
    && mkdir -p /home/explorer/.cache \
    && chown explorer:explorer /home/explorer/.cache

WORKDIR /app

# `dx bundle` lays out the fullstack app as a `server` executable next to
# its `public/` client bundle, which the server resolves relative to its own
# path — so the bundle directory is copied as a whole.
COPY --from=builder /app/dist ./
COPY engine_config.best.yaml ./engine_config.best.yaml

USER explorer

# `IP`/`PORT` are read by the Dioxus server at startup (it defaults to
# 127.0.0.1, unreachable from outside the container).
ENV IP=0.0.0.0 \
    PORT=8080 \
    KALMAN_CONFIG_PATH=/app/engine_config.best.yaml

EXPOSE 8080

CMD ["./server"]
