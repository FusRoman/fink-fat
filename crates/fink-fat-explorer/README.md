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
| Engine config    | `KALMAN_CONFIG_PATH` from `.env`               | host file `$ENGINE_CONFIG` bind-mounted read-only, no rebuild needed |
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
  -v "$PWD/engine_config.best.yaml:/app/config/engine_config.yaml:ro" \
  fink-fat-explorer:prod
```

### Runtime configuration

| Variable             | Where it is set                    | Meaning                                                  |
|----------------------|------------------------------------|----------------------------------------------------------|
| `DATABASE_URL`       | `docker-compose.yml`               | PostgreSQL connection string (required)                  |
| `KALMAN_CONFIG_PATH` | image (`/app/config/engine_config.yaml`) | engine config used by the Kalman replay and the 3D view; mount point of `$ENGINE_CONFIG` |
| `IP`                 | image (`0.0.0.0`)                  | listen address. The Dioxus server defaults to `127.0.0.1`, which is unreachable from outside the container |
| `PORT`               | image (`8080`)                     | listen port inside the container                         |

The production container does **not** read the crate's `.env` file:
`.dockerignore` keeps every `.env` out of the image.

The engine config is not part of the image. Compose bind-mounts the host file
named by `ENGINE_CONFIG` (path relative to the workspace root, default
`./engine_config.best.yaml`) read-only on `KALMAN_CONFIG_PATH`:

```bash
ENGINE_CONFIG=./engine_config_lsst.best.yaml docker compose --profile prod up -d --build
```

To make the choice persistent, put `ENGINE_CONFIG=./engine_config_lsst.best.yaml`
in a `.env` file at the workspace root (read by Compose for interpolation only,
never sent to the container). Switching config, or editing the file, only needs
a restart: `docker compose --profile prod up -d --force-recreate app-prod`.
If the file does not exist, Compose refuses to start the container.

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

## Inspecting the database from a terminal

### Connecting

From the workspace root, open `psql` inside the `db` container (nothing to
install on the host):

```bash
docker compose exec db psql -U finkfat -d finkfat_test
```

From the host, if a `psql` client is installed:

```bash
psql postgresql://finkfat:finkfat@localhost:5432/finkfat_test
```

Run a single query without opening an interactive session (`-T`: no TTY,
so it also works in scripts and pipes):

```bash
docker compose exec -T db psql -U finkfat -d finkfat_test -c "SELECT count(*) FROM branches;"
```

The explorer images do not include `psql`: always go through the `db`
container or the host.

### psql commands

| Command              | Effect                                                         |
|----------------------|----------------------------------------------------------------|
| `\l`                 | list databases                                                 |
| `\c finkfat_test`    | connect to another database                                    |
| `\dt` / `\dt+`       | list tables (`+`: with their size on disk)                     |
| `\d branches`        | columns, indexes and foreign keys of a table                   |
| `\di+`               | list indexes with their size                                   |
| `\x auto`            | expanded display (one column per line) when rows are too wide  |
| `\timing on`         | print the execution time of each query                         |
| `\pset pager off`    | disable the pager (`less`) for long results                    |
| `\copy (SELECT ...) TO 'out.csv' CSV HEADER` | export a query result to a CSV file, written on the machine running `psql` (inside the container when using `docker compose exec`) |
| `\q`                 | quit                                                           |

### Tables

Loaded by `fink-fat convert`. These tables are `UNLOGGED` and emptied
(`TRUNCATE ... CASCADE`) then refilled on every convert run:

| Table                   | Content                                                              |
|-------------------------|----------------------------------------------------------------------|
| `observations`          | one row per alert: position, magnitude, filter, `mjd_tt`, `night_id` |
| `branches`              | one row per track branch; `lineage_id` / `lineage_designation` group the branches of a lineage |
| `branch_observations`   | ordered list (`position`) of the observations of each branch         |
| `kf_bank`               | per-branch Kalman bank summary (steps, absolute magnitude estimate)  |
| `hypotheses`            | Kalman hypotheses of each branch, weighted by `log_weight` (the largest table) |
| `kf_state`              | Kalman state and covariance of each hypothesis, `dynamic_family`     |
| `archived_trajectories` | trajectories archived by the tracker, with their last state, `a` and `e` |

Written by the explorer itself. They are append-only, have no foreign keys
to the tables above and survive a convert run:

| Table                | Content                                                       |
|----------------------|---------------------------------------------------------------|
| `orbit_fits`         | successful orbit fits (single-lineage and bulk)               |
| `orbit_fit_failures` | failed bulk-fit attempts, one row per branch and attempt      |
| `cnd_queries`        | CND cross-match queries and their hits                        |
| `skybot_bulk_jobs`   | Skybot bulk cross-match jobs: status, progress, logs          |
| `skybot_obs_status`  | Skybot result per observation (`hits = []` if no match)       |
| `mpc_submissions`    | ADES submissions to the MPC, with the sent XML and the verdict |

### Useful queries

Table sizes and row counts. `reltuples` is an estimate maintained by
Postgres, instant even on `hypotheses`, whereas `count(*)` scans the whole
table:

```sql
SELECT relname AS table,
       reltuples::bigint AS estimated_rows,
       pg_size_pretty(pg_total_relation_size(oid)) AS total_size
FROM pg_class
WHERE relkind = 'r' AND relnamespace = 'public'::regnamespace
ORDER BY pg_total_relation_size(oid) DESC;
```

A lineage and its branches:

```sql
SELECT branch_id, designation, n_nights, arc_length_days, n_real_updates, cumulative_llr
FROM branches
WHERE lineage_designation = '<designation>'
ORDER BY cumulative_llr DESC;
```

Observations of a branch, in track order:

```sql
SELECT bo.position, o.id, o.night_id, o.mjd_tt, o.ra, o.dec, o.magnitude, o.filter, o.object_id
FROM branch_observations bo
JOIN observations o ON o.id = bo.obs_id
WHERE bo.branch_id = <branch_id>
ORDER BY bo.position;
```

Best hypothesis of a branch and its Kalman state:

```sql
SELECT h.hypothesis_id, h.log_weight, k.dynamic_family, k.epoch, k.rho, k.rho_dot, k.nis_ema
FROM hypotheses h
JOIN kf_state k USING (hypothesis_id)
WHERE h.branch_id = <branch_id>
ORDER BY h.log_weight DESC
LIMIT 1;
```

Branches per dynamic family (scans `hypotheses` and `kf_state`: expect
several minutes on a full database):

```sql
SELECT k.dynamic_family, count(*)
FROM branches b
CROSS JOIN LATERAL (
    SELECT hypothesis_id FROM hypotheses
    WHERE branch_id = b.branch_id
    ORDER BY log_weight DESC LIMIT 1
) best
JOIN kf_state k USING (hypothesis_id)
GROUP BY 1
ORDER BY 2 DESC;
```

Latest orbit fits, and bulk-fit failures:

```sql
SELECT lineage_designation, branch_id, fitted_at, fit_method, converged,
       normalised_rms, n_observations_used, keplerian
FROM orbit_fits
ORDER BY fitted_at DESC
LIMIT 20;

SELECT count(DISTINCT branch_id) AS failed_branches FROM orbit_fit_failures;
```

Skybot jobs and MPC submissions:

```sql
SELECT id, status, started_at, finished_at, processed_observations,
       total_observations, matched_observations, error
FROM skybot_bulk_jobs
ORDER BY started_at DESC;

SELECT id, lineage_designation, trk_sub, endpoint, submitted_at, verdict
FROM mpc_submissions
ORDER BY submitted_at DESC;
```

Queries currently running, for example to spot a slow explorer query:

```sql
SELECT pid, now() - query_start AS duration, state, left(query, 120) AS query
FROM pg_stat_activity
WHERE datname = 'finkfat_test' AND state <> 'idle'
ORDER BY duration DESC;
```

`SELECT pg_cancel_backend(<pid>);` stops one of them. To see a query's plan,
prefix it with `EXPLAIN (ANALYZE, BUFFERS)`: the query is actually executed.

> `app-dev` and `app-prod` share the same database: write statements
> (`UPDATE`, `DELETE`, `TRUNCATE`) affect both. Run
> `BEGIN READ ONLY;` first to explore without risk, and `ROLLBACK;` at the end.

## Build context

[`.dockerignore`](../../.dockerignore) is an allowlist: only `Cargo.toml`,
`Cargo.lock`, `.cargo/`, `LICENSE`, `README.md`, `katex-header.html`,
`src/` and `crates/` are sent to the Docker daemon,
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
