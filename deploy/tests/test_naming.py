"""Tests of finkfat_ops.naming."""

from finkfat_ops import naming
from finkfat_ops.naming import Role, SecurityGroup, Stack


def test_every_resource_shares_the_environment_prefix() -> None:
    names = [
        naming.stack_name("prod", Stack.NET),
        naming.network_name("prod"),
        naming.subnet_name("prod"),
        naming.router_name("prod"),
        naming.security_group_name("prod", SecurityGroup.DB),
        naming.server_name("prod", Role.COMPUTE),
        naming.port_name("prod", Role.DB),
        naming.data_volume_name("prod", "pgdata"),
        naming.root_volume_name("prod", Role.EXPLORER),
    ]
    assert all(name.startswith("ffat-prod-") for name in names)


def test_names_are_the_documented_ones() -> None:
    assert naming.stack_name("prod", Stack.STORAGE) == "ffat-prod-storage"
    assert (
        naming.security_group_name("prod", SecurityGroup.INTERNAL_SSH)
        == "ffat-prod-sg-internal-ssh"
    )
    assert naming.data_volume_name("prod", "pgdata") == "ffat-prod-vol-pgdata"
    assert naming.port_name("prod", Role.DB) == "ffat-prod-db-port"
    assert naming.root_volume_name("prod", Role.DB) == "ffat-prod-db-root"
    assert naming.ssh_host_alias(Role.EXPLORER) == "ffat-explorer"


def test_environments_never_collide() -> None:
    assert naming.server_name("prod", Role.DB) != naming.server_name("staging", Role.DB)
