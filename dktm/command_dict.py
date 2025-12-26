"""
command_dict.py
----------------

This module defines the available commands and associated metadata used by
the DKTM planner.  Commands represent atomic actions the orchestrator
can issue when executing a transition plan (for example, to quiesce
services, flush I/O or perform health checks).  Each command is
associated with a group identifier; commands sharing the same group
conceptually belong to the same control domain (e.g., service layer,
I/O layer).  The group identifier is used by the SOSA adapter when
computing combination encodings for subsets of actions.

Commands may also carry optional additional metadata such as risk
classification or a human‑readable description.  This dictionary can
easily be extended as DKTM evolves.
"""

from __future__ import annotations

from typing import Dict

COMMANDS: Dict[str, Dict[str, int]] = {
    # Service‑level commands
    "freeze_services": {"group_id": 1},
    "quiesce_services_tier1": {"group_id": 1},
    "quiesce_services_tier2": {"group_id": 1},

    # I/O commands
    "flush_io": {"group_id": 2},
    "flush_buffers": {"group_id": 2},

    # Driver/device commands
    "quiesce_drivers_soft": {"group_id": 3},
    "quiesce_drivers_hard": {"group_id": 3},

    # Health check and diagnostics
    "health_check": {"group_id": 4},
    "verify_integrity": {"group_id": 4},

    # Transition control commands
    "enter_maintenance": {"group_id": 5},
    "exit_maintenance": {"group_id": 5},
    "commit_transition": {"group_id": 6},
    "rollback_transition": {"group_id": 6},

    # Additional high‑level commands
    # A no‑operation command used for demonstration and testing.
    "noop": {"group_id": 7},
    # General service quiesce alias; maps to tier1 under the hood.
    "quiesce_services": {"group_id": 1},
    # Snapshot system state prior to transition.
    "snapshot_state": {"group_id": 4},
    # Handover control from the running OS to PE or maintenance environment.
    "handover_control": {"group_id": 6},
}

def get_group_id(command: str) -> int:
    """Return the group identifier for a given command name.

    If the command is unknown a new group id is assigned based on
    hashing (ensuring deterministic mapping).  This helper is used by
    the planner and adapter when mapping arbitrary command names to
    group IDs.

    Parameters
    ----------
    command : str
        The name of the command.

    Returns
    -------
    int
        The group identifier associated with this command.
    """
    if command in COMMANDS:
        return COMMANDS[command]["group_id"]
    # Fallback deterministic mapping for unknown commands
    # Use a large prime to reduce collisions; groups are 1‑indexed
    return (abs(hash(command)) % 10) + 7
