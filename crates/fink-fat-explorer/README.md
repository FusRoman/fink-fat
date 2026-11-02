# fink-fat-explorer

Web explorer for fink-fat products: an (a, e) population map, 3D orbits, a
lineage table, per-lineage pages, orbit fitting, cross-matching (Skybot/CND)
and ADES/MPC submission. It is a Dioxus 0.7 fullstack app: an Axum server
(SSR + server functions, backed by PostgreSQL) and a wasm client.

**SSR** stands for *server-side rendering*. On each page request, the server
runs the Dioxus components itself and sends back ready-made HTML, so the
browser can show the page before the wasm client has finished downloading.
The wasm client then *hydrates* that HTML: it attaches to the existing markup
and makes it interactive, rather than rebuilding the page from scratch. Data
is loaded afterwards through *server functions* (`#[server]`), plain HTTP
endpoints (`POST /api/...`) that the client calls like ordinary async Rust
functions.

Database setup and loading fink-fat products into PostgreSQL are covered in
[`test_postgress_db.md`](test_postgress_db.md).

## Builds: `dev` and `prod`

The app ships as two Docker targets that run **the same application**. They
differ only in how it is compiled and run:

|                  | `dev` (testing / development)                  | `prod` (production)                                         |
|------------------|------------------------------------------------|-------------------------------------------------------------|
| Compose service  | `app-dev` (profile `dev`)                      | `app-prod` (profile `prod`)                                 |
| Container        | `fink-fat-explorer-dev`                        | `fink-fat-explorer-prod`                                    |
| Host port        | `8080`                                         | `8081`                                                      |
| How it runs      | `dx serve`: compiles at container start        | prebuilt `server` binary, no compilation at start           |
| Cargo profile    | `dev` (explorer at opt-level 0, deps at 3)     | `release` (thin LTO, 1 codegen unit), wasm-opt, no debug symbols |
| Sources          | workspace bind-mounted on `/app`, hot reload   | baked into the image at build time, no source code inside   |
| CSS              | Tailwind in `--watch` mode                     | generated once, minified, at build time                     |
| Runtime image    | `rust:1.97.1-trixie` + toolchain (~2 GB)       | `debian:trixie-slim` (~130 MB), non-root user `explorer`    |
| Client wasm      | unoptimized debug build                        | ~2.9 MB, optimized                                          |
| Engine config    | `KALMAN_CONFIG_PATH` from `.env`               | `/app/engine_config.best.yaml`, copied into the image       |
| Restart policy   | none                                           | `unless-stopped`                                            |

Use `dev` to work on the code and test changes. Never expose it to users: it
is a debug build with a dev server and hot reload, and noticeably slower,
especially in the browser.

Both targets live in the workspace-root [`dockerfile`](../../dockerfile) and
are wired up in [`docker-compose.yml`](../../docker-compose.yml). They sit at
the workspace root because the crate inherits its dependencies from the
workspace `Cargo.toml`. Run every command below from the workspace root
(`fink-fat/`).

### Dockerfile stages

```
toolchain ─┬─> dev                        (dx serve)
           └─> builder ──(/app/dist)──> prod   (debian slim + server binary)
```

- `toolchain`: Rust, the `wasm32-unknown-unknown` target, `dioxus-cli`
  (pinned to `0.7.10`, the `dioxus` version in `Cargo.lock`) and
  `tailwindcss-extra`. `dev` and `builder` share it, so both always use the
  same tool versions. When you upgrade `dioxus`, also bump the
  `DIOXUS_CLI_VERSION` build arg.
- `builder`: `tailwindcss-extra --minify`, then
  `dx bundle --web --release --debug-symbols false --out-dir /app/dist`. The
  cargo registry and `target/` are BuildKit cache mounts, so later builds only
  recompile what changed.
- `prod`: copies `/app/dist`, which contains a `server` executable next to
  its `public/` client bundle, plus `engine_config.best.yaml`.

`prod` is the last stage, so a bare `docker build .` produces the production
image.

## Running the dev build

```bash
docker compose --profile dev up -d          # db + pgAdmin + app-dev
docker compose --profile dev up -d --build  # after editing the dockerfile
```

The app is served on http://localhost:8080. The first start compiles the
whole app inside the container. The `cargo-target` named volume keeps
`target/` across container recreations, so later starts are incremental.
Code changes on the host are picked up through the bind mount.

```bash
docker compose logs -f app-dev
docker compose restart app-dev   # e.g. the dx watcher missed a change
```

Recompile the CSS by hand (the watcher normally does it):

```bash
docker compose exec app-dev tailwindcss-extra -i ./tailwind.css -o ./assets/main.css
```

Only `app-dev` has the Rust toolchain. Any `cargo` command, such as
`fink-fat convert` (see [`test_postgress_db.md`](test_postgress_db.md)), has
to go through it.

## Running the production build

```bash
docker compose --profile prod up -d --build   # build (if needed) and start
```

The app is served on http://localhost:8081. Inside the container it listens
on `8080`, published on `8081` so it can run next to `app-dev` on the same
host.

- **First build**: about 8 minutes (release build with LTO, plus wasm-opt).
  Later builds reuse the BuildKit cache.
- **Redeploy after a code change**: run the same command. `--build` rebuilds
  the image and Compose recreates the container. Nothing is recompiled at
  container start.
- **Logs**: `docker compose logs -f app-prod`
- **Stop**: `docker compose --profile prod stop app-prod`

To build the image without Compose:

```bash
docker build --target prod -t fink-fat-explorer:prod .
docker run --rm -p 8081:8080 \
  -e DATABASE_URL=postgres://finkfat:finkfat@<db-host>:5432/finkfat_test \
  fink-fat-explorer:prod
```

### Runtime configuration

| Variable             | Where it is set                    | Meaning                                                  |
|----------------------|------------------------------------|----------------------------------------------------------|
| `DATABASE_URL`       | `docker-compose.yml`               | PostgreSQL connection string (required)                  |
| `KALMAN_CONFIG_PATH` | image (`/app/engine_config.best.yaml`) | engine config used by the Kalman replay and the 3D view |
| `IP`                 | image (`0.0.0.0`)                  | listen address. The Dioxus server defaults to `127.0.0.1`, which is unreachable from outside the container |
| `PORT`               | image (`8080`)                     | listen port inside the container                         |

The production container does **not** read the crate's `.env` file:
`.dockerignore` keeps every `.env` out of the image. The engine config is
copied into the image, so changing `engine_config.best.yaml` requires a
rebuild. To change it without rebuilding, bind-mount a file and override
`KALMAN_CONFIG_PATH`.

### Volumes

| Volume           | Mounted on                 | Content                                                         |
|------------------|----------------------------|-----------------------------------------------------------------|
| `explorer-cache` | `/home/explorer/.cache` (prod) | JPL ephemeris downloaded by outfit and the MPC observatory list from photom, kept across restarts |
| `cargo-target`   | `/app/target` (dev)        | incremental build artifacts of the dev build                    |

Without `explorer-cache`, every prod restart would download the ephemeris
again.

### Quick check of a production image

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8081/   # 200: SSR is up
docker compose logs app-prod | grep "Registering: POST /api"      # server functions registered
```

The page works without a database, but every data component reports an
error until `DATABASE_URL` reaches a populated database.

## Shared services

| Service    | URL / address                                                 |
|------------|---------------------------------------------------------------|
| PostgreSQL | `localhost:5432` (`db:5432` from inside the Compose network)  |
| pgAdmin    | http://localhost:5050 (`admin@admin.com` / `admin`)           |

`db` and `pgadmin` have no profile, so `docker compose up -d` alone starts
them without starting either explorer. Both explorer services currently point
to the same `finkfat_test` database.

```bash
docker compose ps
docker compose down       # stop everything (all profiles)
docker compose down -v    # also wipe volumes: database, pgAdmin, caches
```

## Build context

[`.dockerignore`](../../.dockerignore) is an allowlist: only `Cargo.toml`,
`Cargo.lock`, `.cargo/`, `LICENSE`, `README.md`, `katex-header.html`,
`engine_config.best.yaml`, `src/` and `crates/` are sent to the Docker daemon,
without any `target/`, `.venv/`, `__pycache__/`, `.env` or the local
`tailwindcss-extra` binary. If the `dockerfile` starts copying a new file,
also add it to the allowlist, or the build fails with "not found".

## Migrating from the former single `app` service

The old `app` service (container `fink-fat-explore-app`, image `fink-fat-app`)
is replaced by `app-dev`. Compose reports its container as an orphan.
Remove it with:

```bash
docker compose --profile dev up -d --remove-orphans
docker image rm fink-fat-app   # optional: frees ~2 GB
```

## Without Docker

```bash
cd crates/fink-fat-explorer
./tailwindcss-extra -i ./tailwind.css -o ./assets/main.css --watch   # terminal 1
dx serve                                                              # terminal 2 (dev)
```

Production bundle on the host, with the same command as the `builder` stage:

```bash
dx bundle --web --release --debug-symbols false --out-dir dist
cd dist
IP=0.0.0.0 PORT=8080 DATABASE_URL=... KALMAN_CONFIG_PATH=... ./server
```

Keep `server` and `public/` together, the way the `prod` image does
(`WORKDIR /app`).
