#!/bin/sh
# Regenerate docs/schema/ (one page per table, columns, ER diagram) with
# tbls, from the database at $DATABASE_URL (default: the repo's
# docker-compose `db` service; start it with `docker compose up -d db`).
#
# 1. create every table, documented by the COMMENT ON statements of
#    sql/tables/** (cargo example `create_schema`);
# 2. run tbls — the local binary if installed, its pinned Docker image
#    otherwise — with the crate's .tbls.yml.
#
# Commit the result: CI (`schema-docs` job) fails when docs/schema/ is stale
# (`tbls diff`) or when a table or column has no comment (`tbls lint`).
set -eu

TBLS_VERSION="v1.96.1"
DATABASE_URL="${DATABASE_URL:-postgres://finkfat:finkfat@localhost:5432/finkfat_test}"

crate_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$crate_dir"

DATABASE_URL="$DATABASE_URL" cargo run --quiet -p fink-fat-sql --example create_schema

# tbls' Postgres driver needs the TLS mode spelled out for a local server.
case "$DATABASE_URL" in
    *sslmode=*) TBLS_DSN="$DATABASE_URL" ;;
    *\?*) TBLS_DSN="$DATABASE_URL&sslmode=disable" ;;
    *) TBLS_DSN="$DATABASE_URL?sslmode=disable" ;;
esac
export TBLS_DSN

# `-c`: tbls does not find .tbls.yml on its own. `--user`: without it the
# container writes the docs as root.
if command -v tbls >/dev/null 2>&1; then
    tbls doc -c .tbls.yml --rm-dist
else
    docker run --rm --user "$(id -u):$(id -g)" --network host -e TBLS_DSN \
        -v "$crate_dir:/work" -w /work \
        "ghcr.io/k1low/tbls:$TBLS_VERSION" doc -c .tbls.yml --rm-dist
fi
echo "Schema docs written to $crate_dir/docs/schema"
