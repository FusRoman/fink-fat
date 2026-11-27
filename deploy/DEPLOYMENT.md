# Deploying and updating fink-fat

How `ff-ops deploy` puts the services on the machines, and the procedures to deploy a new
version or a changed configuration. The infrastructure itself (network, volumes, VMs) is
described in the [README](README.md); this document starts once `ff-ops init` is green.

- [What runs where](#what-runs-where)
- [Bundles](#bundles)
- [Image versions](#image-versions)
- [The deployment sequence](#the-deployment-sequence)
- [Procedures](#procedures)
- [When a deployment fails](#when-a-deployment-fails)

## What runs where

| Role | Containers | Persistent data |
| --- | --- | --- |
| db | `postgres` | `/srv/pgdata` (Cinder volume) |
| compute | `stream` (`fink-fat stream`) | `/srv/finkfat` (Cinder volume): engine snapshot, night journal, archive, daily log files |
| explorer | `app` (the explorer), `caddy` (HTTPS on 80/443), `pgadmin` (loopback port 5050) | Docker volumes: certificates (`caddy-data`), pgAdmin settings, downloaded ephemeris |

Every container has `restart: unless-stopped`: it restarts after a crash and after a reboot
of its VM, without ff-ops.

## Bundles

A *bundle* is everything a role needs to run, gathered on the workstation and sent to
`/opt/finkfat` on its VM.

| File on the VM | db | compute | explorer | Comes from |
| --- | :-: | :-: | :-: | --- |
| `compose.yml` | ✔ | ✔ | ✔ | `deploy/compose/<role>.yml` (versioned) |
| `config/engine_config.yaml` | | ✔ | ✔ | `deploy/config/engine_config.yaml` (versioned) |
| `config/stream.yaml` | | ✔ | | `deploy/config/stream.yaml` (**local**, gitignored) |
| `Caddyfile` | | | ✔ | generated from `explorer_domain` (`infra.local.yaml`) |
| `.env` | ✔ | ✔ | ✔ | generated from `infra.yaml` and `secrets.env` |
| `deployed.json` | ✔ | ✔ | ✔ | written after a successful deployment |

### Versioned, local and generated files

- **Versioned files** are the same for everyone: compose files, engine configuration
  (`config/engine_config.yaml`, whose `storage_path` must stay `/data/storage`, the compute's
  data volume; a test checks it).
- **Local files** exist on the operator's workstation only. `config/stream.yaml` names the
  Kafka broker and the consumer group of the production instance, which are not published:
  the consumer has no SASL, and a client joining the same consumer group would take
  partitions, hence alerts, from the production stream. It is created once from the
  versioned template:

  ```bash
  cp config/stream.example.yaml config/stream.yaml && chmod 600 config/stream.yaml
  # then replace the two PLACEHOLDER values
  ```

  `ff-ops config check` and `ff-ops doctor` fail if it is missing or tracked by git.
- **Generated files** are rebuilt at every deployment and never edited by hand.

### The `.env`

Each role gets only the variables its containers need:

| Role | Variables |
| --- | --- |
| db | `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `DB_BIND_IP`, `PG_*` (Postgres settings sized from the VM's memory) |
| compute | `FFAT_REGISTRY`, `FFAT_TAG`, `FINK_FAT_DATABASE_URL`, `FINK_FAT_KAFKA_PASSWORD` |
| explorer | `FFAT_REGISTRY`, `FFAT_TAG`, `DATABASE_URL`, `PGADMIN_EMAIL`, `PGADMIN_PASSWORD` |

Docker Compose reads the `.env` next to `compose.yml` and replaces every `${VARIABLE}` of the
compose file with its value. Values are single-quoted, so Compose takes them literally (a `$`
in a password is safe). The file has mode `600`.

### Transport

The bundle travels as **one compressed tar archive on the standard input of `ssh`**, unpacked
in `/opt/finkfat` by `base64 -d | tar -x`. Consequences:

- no secret ever appears on a command line (visible to every user of a machine through `ps`,
  and in `--dry-run` output), nor in a file on the workstation;
- the VMs need nothing beyond `tar` and `base64`;
- files are overwritten; a file that is no longer part of the bundle stays on the VM, unused.

### The deployment record and drift

After a successful deployment, `deployed.json` records the image tag, a **fingerprint** of
the bundle's files (a hash of their names and contents, the `.env` excluded) and the date. It
holds no secret. `ff-ops status` compares it with the current state of `deploy/`:

| Column | ✔ when | Otherwise |
| --- | --- | --- |
| version | the deployed tag equals `images.tag` of `infra.yaml` | `(<tag> wanted)` and a warning |
| config | the deployed fingerprint equals what `ff-ops deploy` would send now | `⚠ drift` |

A drift means that `deploy/` (or `explorer_domain`, or the local `stream.yaml`) changed since
the last deployment of that role: `ff-ops deploy <role>` aligns the machine. Since
`stream.yaml` is local, two workstations with different copies see each other's deployments
as a drift.

## Image versions

The compute and explorer images come from GHCR:

```yaml
# compose/compute.yml
image: ${FFAT_REGISTRY}/fink-fat-engine:${FFAT_TAG}
```

```
infra.yaml (images.registry, images.tag)   or   ff-ops deploy --tag <tag>
        │
        ▼
.env of the role:  FFAT_REGISTRY='ghcr.io/<owner>'  FFAT_TAG='v1.1.0'
        │  (sent in the bundle)
        ▼
docker compose pull / up  →  ghcr.io/<owner>/fink-fat-engine:v1.1.0
```

`images.tag` in `infra.yaml` is **the version that should run**: `ff-ops deploy` uses it by
default and `ff-ops status` compares the machines with it. `--tag` overrides it for one
deployment only (a test, an emergency rollback); align `infra.yaml` afterwards, otherwise the
next plain `ff-ops deploy` brings the default tag back.

To see the image names a machine resolved (without printing the `.env`'s secrets):

```bash
pdm run ff-ops ssh compute -- 'cd /opt/finkfat && docker compose config --images'
```

The db image (`postgres:<version>`) and the explorer's `caddy` and `pgadmin` images are pinned
in the compose files themselves, not by `images.tag`.

## The deployment sequence

```
ff-ops deploy [roles…] [--tag <tag>]
  check that every file of every requested role exists     (before touching any machine)
  "Proceed?"
  for each role, in the order db → compute → explorer:
    1. read the versioned and local files
    2. ssh: create /opt/finkfat; on the compute, give /srv/finkfat to UID 10001
    3. ssh: read the memory (db only, to size Postgres)
    4. build the bundle and its .env, in memory
    5. ssh: unpack the bundle (archive on stdin)
    6. ssh: docker login (only if GHCR_READ_TOKEN is set; never for the db)
    7. ssh: docker compose pull, then docker compose up -d --remove-orphans
    8. health checks
    9. ssh: write deployed.json (only if no check failed)
  a role in error stops the loop: the next roles are not touched
```

`docker compose up -d` recreates only the containers whose image or configuration changed:
redeploying with nothing changed restarts nothing.

| Role | Health check |
| --- | --- |
| db | `pg_isready` in the container, retried for up to two minutes (the first start runs `initdb`) |
| compute | the stream is still running, without any restart, 30 seconds after the start |
| explorer | Caddy reaches the explorer, retried for up to two minutes; then `https://<explorer_domain>/` answers from the workstation with a valid certificate |

The code is in [`finkfat_ops/deploy.py`](finkfat_ops/deploy.py) (its module docstring maps
each step to its function); `--dry-run` prints every command without running it.

## Procedures

Every procedure ends the same way: `pdm run ff-ops status` shows ✔ everywhere, then the
changed versioned files are committed.

### First deployment

Once the three stacks are applied and `ff-ops init` is green:

1. `config/stream.yaml` created from its template (see [Local files](#versioned-local-and-generated-files));
   `explorer_domain` set in `infra.local.yaml` (a name derived from the floating IP,
   `<a-b-c-d>.sslip.io`, needs no DNS request).
2. `pdm run ff-ops doctor`: everything ✔.
3. `pdm run ff-ops deploy`.
4. `pdm run ff-ops logs compute stream -f`: the schema migration is applied, then the stream
   reports `caught up with the Kafka stream` once the retained backlog is consumed.

### Deploy a new version

1. Publish a GitHub release (tag `vX.Y.Z`). CI runs, and once it is green the images are
   published as `vX.Y.Z`. Check them with `pdm run ff-ops images list`.
2. Read the release's changes for **database migrations**
   (`crates/fink-fat-sql/sql/migrations/`): see [Migrations](#migrations) below.
3. Set `images.tag: vX.Y.Z` in `infra.yaml`.
4. `pdm run ff-ops deploy` (the db is not affected by the tag: it is left as is).
5. `pdm run ff-ops status`: `vX.Y.Z` on the compute and the explorer, no warning.
6. Commit `infra.yaml`.

Restarting the stream loses nothing: on `SIGTERM` it flushes its SQL outbox, keeps the open
night in its journal, and resumes from its committed Kafka offsets.

### Roll back to the previous version

Without a migration in between: `pdm run ff-ops deploy --tag <previous tag>`, then set the
same tag back in `infra.yaml`.

### Migrations

The migrations are compiled into the engine image and applied by the stream when it starts.
A binary refuses to run against a database holding a migration it does not know, so **once a
new version has migrated the database, the previous image no longer starts**: rolling back
then means restoring the database. Before deploying a release that carries a migration, take
a snapshot of the database volume:

```bash
pdm run ff-ops volume snapshot db
```

The full sequence (stop, snapshot, migrate, restore on failure) is planned as
`ff-ops update`; see [Database migrations](README.md#database-migrations).

### Change the fink-fat configuration

| File | Then |
| --- | --- |
| `config/engine_config.yaml` | `pdm run ff-ops deploy compute explorer` (both must use the same file) |
| `config/stream.yaml` (local) | `pdm run ff-ops deploy compute` |

- Keep `storage_path: /data/storage` in `config/engine_config.yaml`: it is where the compute's
  data volume is mounted.
- Some engine parameters change the meaning of the existing snapshot (for instance the
  geometry of the buckets). Review such a change before deploying it.
- A new `group_id` in `stream.yaml` starts a new consumer group, which begins at the oldest
  alert still retained by Fink.

### Change a pinned image (Postgres, Caddy, pgAdmin)

Edit the tag in `compose/<role>.yml`, then `pdm run ff-ops deploy <role>`.

- **Postgres minor versions** (`19.0` → `19.1`) are safe. The stream retries its writes while
  Postgres restarts.
- **Postgres major versions** (`19` → `20`) are **not** an image change: the image keeps its
  data in a directory named after the major version (`/srv/pgdata/<major>/docker`), so a new
  major version would start an **empty** database next to the old one. A major upgrade needs
  `pg_upgrade` or a dump and restore, prepared as a separate operation.

### Change the domain

Set `explorer_domain` in `infra.local.yaml` (after the DNS record of the new name points to the
floating IP), then `pdm run ff-ops deploy explorer`. Caddy obtains the new certificate on its
own.

## When a deployment fails

- **A file is missing**: `deploy` stops before touching any machine and names the file, with
  how to create it.
- **A step fails** (SSH, `docker compose pull`…): `deploy` stops with the command's error. The
  role keeps running what it ran before unless `up -d` already replaced its containers.
- **A health check fails**: the new containers are running (or restarting), `deployed.json` is
  not updated, and the next roles are not deployed. `ff-ops status` shows the role in error.
  Read `pdm run ff-ops logs <role>`, then either fix and redeploy, or go back to the previous
  tag (`--tag`), which is safe as long as no migration was applied.

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| db: `postgres accepts connections` fails | wrong settings, full disk | `ff-ops logs db` |
| compute: `stream running, no restart` fails | invalid `stream.yaml` or engine configuration, database unreachable | `ff-ops logs compute stream` |
| explorer: HTTPS check fails | DNS record not pointing to the floating IP yet, port 80 closed | `ff-ops logs explorer caddy` |
| `status`: `⚠ drift` | `deploy/` changed since the last deployment | `ff-ops deploy <role>` |
| `status`: `runs X, infra.yaml says Y` | deployed with `--tag`, or `infra.yaml` not deployed yet | align `infra.yaml`, then `ff-ops deploy` |
