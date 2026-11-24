# ff-ops — fink-fat production tooling

`ff-ops` manages the production instance of Fink-FAT on an OpenStack cloud. It
creates the cloud resources, prepares the machines, deploys the services with Docker
Compose, and reports their state. It is a Python tool living in `deploy/`, separate from
the Rust workspace, and it drives the `openstack`, `ssh`, `docker` and `git` command-line
tools.

The production chain is `Fink Kafka → fink-fat stream → Postgres → fink-fat-explorer`,
spread over three machines:

```text
                       Internet
                          │  80/443 (everyone)   22 (allowed ranges only)
                  [floating IP — the only public address]
 ┌───────────── explorer  (bastion) ──────────────┐
 │  Caddy (HTTPS) ─► fink-fat-explorer   pgAdmin   │
 └──────────┬──────────────────────────────────────┘
            │  private network  ffat-<env>-net
   ┌────────┴──────────┐              ┌──────────────────────┐
   │ compute           │ ── 5432 ───► │ db                   │
   │ fink-fat stream   │              │ Postgres             │
   │ volume: storage   │              │ volume: PGDATA       │
   └────────┬──────────┘              └──────────────────────┘
            └─► Fink Kafka, JPL, MPC, ghcr.io
```

| Role | Runs | Persistent data |
| --- | --- | --- |
| `compute` | `fink-fat stream` (Kafka client of the Fink broker + tracking engine) | engine `storage_path`: snapshot, archive, night journal, SQL outbox — the source of truth |
| `db` | Postgres | the explorer database (rebuildable from the snapshot) |
| `explorer` | `fink-fat-explorer`, Caddy, pgAdmin; SSH bastion | ephemeris cache, TLS certificates |

## Design

- **Infrastructure as code with Heat.** The cloud resources are described by three
  OpenStack Heat stacks, generated from the configuration:

  | Stack | Content | Lifecycle |
  | --- | --- | --- |
  | `ffat-<env>-net` | network, subnet, router, security groups, the explorer's floating IP | almost never changes |
  | `ffat-<env>-storage` | data volumes, `deletion_policy: Retain` | must never be lost |
  | `ffat-<env>-servers` | fixed-IP ports, VMs, volume attachments, floating IP association | can be destroyed and recreated |

- **Docker images from CI.** Images are built by GitHub Actions only when CI is green,
  and pushed to GHCR (`ghcr.io/fusroman/fink-fat-{engine,explorer}`). The VMs only pull
  them.
- **OpenStack is the single source of state.** `ff-ops` keeps no state file of its
  own: every command queries the cloud.
- **Pure core, thin I/O layer.** Configuration models, resource names, templates and
  command lines are built by pure functions. Only `runner.py` executes anything, through
  one of three runners:
  - `SubprocessRunner` runs the commands for real;
  - `DryRunRunner` (`--dry-run`) prints them instead;
  - `FakeRunner` answers from recorded responses (tests).

## Layout

| Path | Role |
| --- | --- |
| [`finkfat_ops/config.py`](finkfat_ops/config.py) | Pydantic models of the three configuration files, merge and validation. |
| [`finkfat_ops/naming.py`](finkfat_ops/naming.py) | Names of every managed resource (`ffat-<env>-*`) and SSH host alias. |
| [`finkfat_ops/runner.py`](finkfat_ops/runner.py) | `Command` and the three runners (subprocess, dry-run, fake). Every command has a timeout. |
| [`finkfat_ops/checks.py`](finkfat_ops/checks.py) | Local safety checks: sensitive files untracked by git and readable by their owner only. |
| [`finkfat_ops/openstack.py`](finkfat_ops/openstack.py) | `openstack` command builder bound to the cloud profile (`--os-cloud`), project verification. |
| [`finkfat_ops/heat/`](finkfat_ops/heat/) | Heat templates built from the configuration (`net.py`), and their YAML rendering. |
| [`finkfat_ops/stack.py`](finkfat_ops/stack.py) | Stack commands, parsing of their output, and the replacement guard. |
| [`finkfat_ops/infra.py`](finkfat_ops/infra.py) | Stack orchestration: render, preview, guard, confirmation, apply. |
| [`finkfat_ops/access.py`](finkfat_ops/access.py) | Source ranges allowed to SSH into the bastion (stable and dynamic). |
| [`finkfat_ops/cli.py`](finkfat_ops/cli.py) | Typer command-line interface (wiring only). |
| [`tests/`](tests/) | pytest suite. It never contacts the cloud. |
| [`infra.yaml`](infra.yaml) | Versioned deployment structure. |
| [`infra.local.example.yaml`](infra.local.example.yaml), [`secrets.example.env`](secrets.example.env) | Templates of the two gitignored files. |

## Configuration

Three files are merged into one validated deployment:

| File | In git | Content |
| --- | --- | --- |
| `infra.yaml` | yes | Structure: environment name, OS image and its default SSH user, flavors, private network, fixed IPs, volume sizes, default image tag. Nothing personal. |
| `infra.local.yaml` | **no** | Cloud profile (`clouds.yaml` entry), expected project name, source ranges allowed to SSH into the bastion, explorer domain, SSH private key, DNS servers of the private subnet. |
| `secrets.env` | **no** | Postgres and pgAdmin passwords, Fink Kafka password, GHCR read token. |

The repository is public. A list of the ranges allowed to reach the bastion does not
belong in it, so the SSH ranges live in `infra.local.yaml` like the secrets.

Validation rejects:

- an unknown key (a typo is never silently ignored);
- a local-only key (`openstack`, `admin_cidrs`, `explorer_domain`) found in `infra.yaml`;
- a fixed IP outside the subnet, on the gateway, or duplicated;
- `compute` or `db` without a data volume;
- a public IP on any machine other than the explorer;
- SSH open to `0.0.0.0/0`;
- a missing or too short password.

Secrets never appear in error messages or object representations.

## Quick start

Requires Python 3.12 and pdm. The OpenStack CLI and
its Heat plugin are installed in the project environment, at pinned versions.

```bash
cd deploy
pdm install                                   # creates deploy/.venv
cp infra.local.example.yaml infra.local.yaml  # fill in cloud profile, project, SSH ranges, domain
cp secrets.example.env secrets.env && chmod 600 secrets.env   # fill in the passwords
pdm run ff-ops config check
```

If some OpenStack endpoints (Heat in particular) are only reachable from the institution
network, use a VPN or `sshuttle` before running cloud commands.

When it creates a stack, Heat creates a Keystone trust on the operator's behalf. Keystone
refuses to create a trust from a token obtained with an application credential, even an
unrestricted one, and the request then fails as a bare `HTTP 500 Internal Error`
(`RemoteError`). The cloud profile used by ff-ops therefore authenticates with the user's
password (`auth_type: password`), with no password stored in `clouds.yaml`:

```yaml
clouds:
  my-cloud-profile:
    auth_type: password
    auth:
      auth_url: https://keystone.example.org:5000/v3
      username: <user>
      user_domain_name: <user domain>
      project_id: <project id>
    region_name: <region>
    identity_api_version: 3
```

With `openstack.ask_password: true` in `infra.local.yaml`, ff-ops asks for the password
once per command (or reads `OS_PASSWORD` when it is set) and passes it to every `openstack`
call through the environment, never on the command line. It is not asked for commands that
contact nothing (`infra render`, `--dry-run`).

## Commands

| Command | Status | Effect |
| --- | --- | --- |
| `ff-ops config check` | available | Validate the three files without contacting anything. Also checks that the sensitive files are untracked by git and private. |
| `ff-ops doctor` | planned | Check the workstation (CLI, Heat plugin, cloud profile, project, endpoint reachability, SSH key, allowed IP) and print the fix for each problem. |
| `ff-ops infra render / plan / apply / show / destroy` | available | Generate the Heat templates, preview the changes (`stack update --dry-run`), apply them. A replacement guard refuses any change that would replace or delete a protected resource (network, volumes, `db` and `compute` VMs and their data attachments, floating IP) unless it is named with `--allow-replace` and retyped. `destroy` requires retyping the stack name. `plan` without a stack previews the three stacks in order; `apply` and `render` without a stack only print their help. |
| `ff-ops access list / allow-me / revoke` | available | Manage the source ranges allowed to SSH into the bastion (see below). |
| `ff-ops up` | planned | First deployment: `net` → `storage` → `servers` → first-boot checks → services. Idempotent. |
| `ff-ops status` | planned | One table: VMs, containers, deployed version, configuration drift, disk usage, certificate expiry, tracking progress, schema version. |
| `ff-ops update / rollback` | planned | Deploy an image tag or changed configuration in the safe order `db` → `compute` → `explorer`, with health checks and automatic rollback. |
| `ff-ops config edit / diff / push` | planned | Change the production fink-fat configuration files, validated by `fink-fat config check` from the target image before anything is deployed. |
| `ff-ops volume list / snapshot` | available | List the data volumes (size, status, attachment, created or adopted by the storage stack, or orphan); take a Cinder snapshot of a role's data volume, even while attached. |
| `ff-ops ssh <role> [-- command]` | available | Open a shell on a machine through the bastion, or run a command. `--refresh` rewrites `.state/ssh_config` from the servers stack. |
| `ff-ops init [roles…]` | available | Wait for cloud-init on each machine (bastion first), then check Docker, the data volume mount and the clock synchronisation. |
| `ff-ops vm list / console / reboot / stop / start` | available | List the VMs, show the end of a console log (boot messages, even without SSH), power actions (confirmation for reboot and stop). |
| `ff-ops volume extend / logs / tunnel` | planned | Volume resizing, logs, pgAdmin tunnel. |

`--dry-run` is a global option: it prints every external command instead of running it.

`openstack stack update --dry-run` reports as updated or replaced every resource that
merely references another one (`get_resource`), even when nothing changes. `plan` and
`apply` therefore first compare the deployed template (`openstack stack template show`)
and parameters with the new ones: when nothing changes, Heat is not asked at all; otherwise
Heat's preview is kept only for the resources whose own definition, or one of the
parameters they use, changes. The discarded entries are listed on a separate line. A real
replacement still shows through the resource whose definition changes, and the guard sees
it.

For the same reason, Heat's preview may announce the replacement of a resource whose only
change is an updatable property (adding DNS servers to the subnet, for instance). Such a
replacement is counted as an update when every property that changes is `update_allowed`
in Heat's schema of the type (`openstack orchestration resource type show`), which is the
rule Heat applies during the real update. A single non-updatable property keeps the
replacement, and the guard.

Every command that talks to the cloud first checks that the profile of
`infra.local.yaml` (`openstack.cloud`) authenticates into the expected project
(`openstack.project`, a name or an ID), and stops otherwise.

## The `net` stack

| Resource | Content |
| --- | --- |
| `net`, `subnet` | Private network. Fixed IPs live below offset 100 of the subnet; DHCP only hands out addresses from offset 100 upwards, so no other port can take a fixed IP. |
| `router`, `router_interface` | Outbound Internet access (NAT) through the external network. Nothing comes in this way. |
| `sg_explorer` | TCP 80 and 443 from anywhere (web site, certificate issuance). |
| `sg_admin_ssh` | TCP 22 from the allowed source ranges only (bastion). |
| `sg_internal_ssh` | TCP 22 from members of `sg_explorer` (the bastion reaches compute and db). |
| `sg_db` | TCP 5432 from members of `sg_compute` and `sg_explorer`. |
| `sg_compute` | No rule: a label referenced by `sg_db`. |
| `fip_explorer` | The explorer's floating IP (web site and SSH bastion), protected by the guard. It lives here rather than in the servers stack so that recreating the servers keeps the address and its DNS record; the servers stack associates it with the explorer's port. |

Security groups only filter inbound traffic; outbound traffic stays open.

## The `storage` stack

One `OS::Cinder::Volume` per data volume of `infra.yaml` (`vol_engine` for compute,
`vol_pgdata` for db), with `deletion_policy: Retain`: deleting the stack, or removing a
volume from it, keeps the volume and its data. A disabled role keeps its volume too. The
volume resources are protected by the replacement guard.

A retained volume is no longer managed by any stack. To manage it again without creating a
new, empty volume, each volume has a template parameter `<resource>_existing_id`:

| Parameter | Effect |
| --- | --- |
| empty (default) | Heat creates the volume. |
| a volume ID | The resource's condition is false: Heat creates nothing, and the output `<resource>_id` returns that ID. The volume is *adopted*. |

`ff-ops infra plan|apply storage` sets these parameters itself. It looks up each volume
by name in Cinder and compares it with the stack's current parameters and outputs:

| Cinder volumes with the name | Decision |
| --- | --- |
| none | created by the stack |
| one, created by the stack | left as is |
| one, already adopted by the stack | stays adopted |
| one, managed by no stack (e.g. after `infra destroy storage`) | adopted (`= adopted:` line in the preview) |
| several | refused: check which one holds the data, rename or delete the others |
| none, but the stack had adopted one | refused: restore it from a snapshot first |

The template never depends on the cloud: `infra render storage` writes exactly the file
that `apply` sends, and only the parameters change. The outputs `vol_engine_id` and
`vol_pgdata_id` are what the `servers` stack attaches.

An adopted volume keeps its own size: `size_gb` in `infra.yaml` only applies when the
stack creates the volume.

## The `servers` stack

Its parameters are the outputs of the `net` stack (network, subnet, security groups) and of
the `storage` stack (data volume IDs): ff-ops reads them before every preview and apply,
and refuses to go on while either stack is missing or failed. The stack therefore holds no
ID of its own and can be destroyed and recreated; the data stays on the storage volumes.

| Resource | Content |
| --- | --- |
| `port_<role>` | Neutron port with the role's fixed IP and security groups (explorer: `explorer` + `admin-ssh`; db: `db` + `internal-ssh`; compute: `compute` + `internal-ssh`). |
| `root_<role>` | Boot volume (`root_gb`) created from `os_image`, named `ffat-<env>-<role>-root`; deleted with the VM. |
| `server_<role>` | The VM, booting from `root_<role>`. `user_data_update_policy: IGNORE` (editing cloud-init never rebuilds it), `flavor_update_policy: RESIZE` (a new flavor resizes it). |
| `attach_<role>` | Attachment of the role's data volume (compute, db). |
| `fip_explorer_assoc` | Association of the net stack's floating IP with the explorer's port. The address itself lives in the net stack: destroying and recreating this stack, or the explorer alone, never changes it. |

A disabled role (`enabled: false`) gets no resource: applying deletes its VM and boot
volume, while its data volume stays in the storage stack. The db and compute VMs, their
data attachments are protected by the replacement guard.

### First boot (cloud-init)

Every VM receives the same model of `user_data`, with no secret in it (it is readable
from the VM's metadata service and from `openstack stack show`):

- `chrony` for the clock, `dnf-automatic` restricted to security updates;
- Docker CE and its compose plugin from the Docker repository, the `local` log driver
  (rotated container logs), the SSH user in the `docker` group;
- for compute and db, `/usr/local/sbin/ffat-mount-volume`: it waits for the data volume
  (Heat attaches it after the boot), creates an ext4 file system **only if the disk is
  blank**, records it in `/etc/fstab` through its stable `/dev/disk/by-id/virtio-…` path
  with `nofail`, and mounts it.

`ff-ops init` then waits for `cloud-init status --wait` and checks the result.

### SSH

After `infra apply servers`, ff-ops writes `.state/ssh_config`: the explorer is reached on
its floating IP, compute and db through it (`ProxyJump`). Host keys go to ff-ops' own
`.state/known_hosts`, under a stable alias per machine; when a VM is recreated, its old key
is removed there automatically. `ssh_identity_file` in `infra.local.yaml` selects the
private key of the Nova keypair (otherwise `ssh` uses its defaults and agent).

## SSH access to the bastion

The allowed source ranges come from two places:

| Source | Managed by | Typical content |
| --- | --- | --- |
| `admin_cidrs` in `infra.local.yaml` | the operator, by hand (never rewritten by ff-ops) | stable ranges, e.g. the institution egress range |
| `dynamic_cidrs` in `.state/access.yaml` | `ff-ops access` | an address that changes, e.g. a home IP |

`ff-ops access allow-me` detects the current public IPv4 address (through
`api.ipify.org`), replaces the previous dynamic address with it (`--temporary` adds it
instead), and updates the `net` stack. The update is refused if anything other than
`sg_admin_ssh` would change, and the state file is only written once the update
succeeded. Losing SSH access never locks the operator out: the rules are changed through
the OpenStack API, not through SSH.

## Database migrations

The Postgres schema is migrated by refinery (see the
[`fink-fat-sql` README](../crates/fink-fat-sql/README.md)). The migrations are
compiled into the `fink-fat` binary, and a binary refuses to run against a database that
holds a migration it does not know. A migration therefore cannot be undone by rolling
back the image. When an update carries pending migrations, `ff-ops update` will:

1. announce them in its plan;
2. stop the stream and Postgres cleanly;
3. take a Cinder snapshot of the database volume;
4. migrate;
5. restart the services.

Rolling back past a migration is an explicit restore from that snapshot. It is never
automatic. Migrations must stay compatible with the previous release (expand / contract),
since the old explorer briefly runs against the new schema.

## Security

| Secret | Lives in | Never in |
| --- | --- | --- |
| OpenStack password | typed at each ff-ops command (or `OS_PASSWORD`); `clouds.yaml` holds no secret | the repository, the VMs, any file |
| Passwords and tokens | `secrets.env` (gitignored), then a per-role `.env` (mode 600) on each VM | Heat templates, cloud-init `user_data`, images |
| GHCR write access | the ephemeral `GITHUB_TOKEN` of the CI job | any workstation |
| SSH private key | the operator's machine | the bastion, the VMs |

Only ports 80 and 443 of the explorer are open to the Internet. SSH is limited to the
configured ranges and to key authentication. Postgres listens on the private network
only.

## Tests and code quality

```bash
cd deploy
pdm run pytest              # test suite
pdm run ruff format .       # formatting
pdm run ruff check .        # linting: pycodestyle, pyflakes, isort, naming, pyupgrade,
                            # bugbear, simplify, pathlib, pylint, numpy-style docstrings
```

The ruff configuration lives in `pyproject.toml`. Both `ruff format --check .` and
`ruff check .` must pass before a change is committed: the repository's pre-commit hook
(`.husky/pre-commit`) runs them, and the `deploy` job of the CI runs them again together
with the test suite, from the locked dependencies (`pdm install --check`).

The suite uses `FakeRunner` and recorded OpenStack responses. Apart from `git`, it
only runs a few harmless local commands, and it never contacts the cloud.
