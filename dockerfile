# syntax=docker/dockerfile:1
#
# fink-fat images. One file, four runnable targets:
#
#   engine    `fink-fat` (release), in a slim runtime image: runs
#             `fink-fat stream` in production (published to GHCR by CI).
#   explorer  fink-fat-explorer, `dx bundle --release` (optimized server binary
#             + wasm-opt'd client) in a slim runtime image (published to GHCR).
#   prod      alias of `explorer`, kept for the local `docker-compose.yml`.
#   dev       explorer with `dx serve` against the bind-mounted source tree:
#             debug profile, hot reload, Tailwind in watch mode.
#
# Select one with `docker build --target <engine|explorer|prod|dev>` or through
# the matching `docker-compose.yml` service. `prod` is the last stage, so a
# plain `docker build .` produces the explorer production image.
#
#   sources ─┬─────────────────────────► engine-builder ──► engine
#   rust-base┴► toolchain ─┬─► builder ──► explorer ──► prod
#                          └─► dev

# ---------------------------------------------------------------------------
# sources: the workspace files needed to build, copied once and reused by
# every compiling stage. Deliberately explicit, not `COPY . .`, so unrelated
# large data files added anywhere in the repo never end up in an image (see
# also `.dockerignore`). Root Cargo.toml/Cargo.lock are required so cargo can
# resolve inherited workspace dependencies.
# ---------------------------------------------------------------------------
FROM scratch AS sources
COPY Cargo.toml Cargo.lock /app/
COPY .cargo /app/.cargo
COPY LICENSE README.md katex-header.html /app/
COPY src /app/src
COPY crates /app/crates

# ---------------------------------------------------------------------------
# rust-base: Rust and the system libraries every crate of the workspace needs
# to compile, shared by the engine and the explorer so both use the exact same
# compiler.
# ---------------------------------------------------------------------------
FROM rust:1.97.1-trixie AS rust-base

# build-essential/make + libssl-dev: librdkafka, built from its bundled sources
# with its own `./configure` and SSL (the `kafka` feature of fink-fat-stream);
# libssl-dev also serves reqwest's native-tls. curl: binary downloads below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libssl-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# engine-builder: one-shot release build of the `fink-fat` binary.
# ---------------------------------------------------------------------------
FROM rust-base AS engine-builder

COPY --from=sources /app /app

# The cargo registry and target directory are BuildKit cache mounts, so a
# local rebuild only recompiles what changed. Cache mounts are not part of the
# image, hence the copy of the binary out of them.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --release --locked -p fink-fat --bin fink-fat \
    && install -D -m 0755 target/release/fink-fat /out/fink-fat

# ---------------------------------------------------------------------------
# engine: runtime-only image of `fink-fat stream`. No compiler, no sources, no
# configuration and no secret: the stream configuration (with the engine
# configuration it points to) is bind-mounted read-only on /config, and the
# engine's `storage_path` (snapshot, journal, outbox, logs) on /data.
# ---------------------------------------------------------------------------
FROM debian:trixie-slim AS engine

# libssl: librdkafka (SASL_SSL to Fink's Kafka) and reqwest (HTTPS).
# ca-certificates: the brokers' and the ephemeris servers' certificates.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3t64 \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged runtime user (fixed UID: the host's data directory is chowned to
# it). Its cache directory holds the downloaded JPL ephemeris; it is created
# here so a named volume mounted on it inherits this ownership.
RUN useradd --create-home --uid 10001 finkfat \
    && mkdir -p /home/finkfat/.cache /data /config \
    && chown finkfat:finkfat /home/finkfat/.cache /data

COPY --from=engine-builder /out/fink-fat /usr/local/bin/fink-fat

USER finkfat
WORKDIR /data

# `fink-fat` is the entrypoint, so `docker run <image> <subcommand> …` runs any
# other subcommand (e.g. `--help`); the default command is the stream.
ENTRYPOINT ["fink-fat"]
CMD ["stream", "--config", "/config/stream.yaml", "--logs"]

# ---------------------------------------------------------------------------
# toolchain: everything needed to compile the explorer, shared by `dev` and
# `builder` so both always use the exact same Rust/dx/Tailwind versions.
# ---------------------------------------------------------------------------
FROM rust-base AS toolchain

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

# The workspace sources, last: editing them never invalidates the tool layers.
COPY --from=sources /app /app

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
# explorer: runtime-only image. Holds the server binary and its `public/`
# client bundle — no compiler, no sources. The engine config is NOT baked in:
# it is bind-mounted at runtime on `KALMAN_CONFIG_PATH` (see
# `docker-compose.yml`), so any config file can be used without rebuilding.
# ---------------------------------------------------------------------------
FROM debian:trixie-slim AS explorer

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

USER explorer

# `IP`/`PORT` are read by the Dioxus server at startup (it defaults to
# 127.0.0.1, unreachable from outside the container). `KALMAN_CONFIG_PATH`
# is the fixed in-container mount point of the engine config file.
ENV IP=0.0.0.0 \
    PORT=8080 \
    KALMAN_CONFIG_PATH=/app/config/engine_config.yaml

EXPOSE 8080

CMD ["./server"]

# ---------------------------------------------------------------------------
# prod: alias of `explorer` for the local `docker-compose.yml` (`app-prod`),
# and the last stage, so a plain `docker build .` builds the explorer.
# ---------------------------------------------------------------------------
FROM explorer AS prod
