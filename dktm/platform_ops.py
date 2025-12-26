"""
Platform operations abstraction
------------------------------

This module selects the appropriate platform implementation (Windows or
POSIX) for executing system‑level operations required by DKTM.  These
operations include committing a one‑time boot sequence, creating and
removing transition markers, rebooting the machine and handing over
control to a secondary environment.

At runtime, the correct backend is chosen based on ``sys.platform``.
On unsupported platforms the operations are implemented as no‑ops with
logging.  This allows the DKTM orchestrator to be exercised in
development and testing environments without modifying the system.
"""

from __future__ import annotations

import sys
from typing import List, Optional

import logging

if sys.platform.startswith("win"):
    from .platform_windows import PlatformOps  # type: ignore
else:
    from .platform_posix import PlatformOps  # type: ignore

logger = logging.getLogger("dktm.platform_ops")


def commit_transition(
    winpe_entry_ids: List[str],
    marker_path: str,
    auto_reboot: bool,
    dry_run: bool = False
) -> None:
    """Commit a transition to a secondary environment.

    For Windows this will set the bootsequence to point to the first
    WinPE entry in ``winpe_entry_ids``, write a marker file at
    ``marker_path`` and optionally reboot the machine.  On POSIX
    systems the call is logged but otherwise ignored.

    Parameters
    ----------
    winpe_entry_ids : List[str]
        List of BCD entry GUIDs for WinPE environments.
    marker_path : str
        Path to the transition marker file.
    auto_reboot : bool
        If True, reboot immediately after committing.
    dry_run : bool
        If True, log actions without executing them.
    """
    ops = PlatformOps(
        winpe_entry_ids=winpe_entry_ids,
        marker_path=marker_path,
        dry_run=dry_run
    )
    ops.commit_transition(auto_reboot=auto_reboot)


def rollback_transition(
    winpe_entry_ids: List[str],
    marker_path: str,
    dry_run: bool = False
) -> None:
    """Rollback a previously committed transition.

    On Windows this restores the previous boot order and removes the
    marker file.  On POSIX systems the call is logged but otherwise
    ignored.

    Parameters
    ----------
    winpe_entry_ids : List[str]
        List of BCD entry GUIDs for WinPE environments.
    marker_path : str
        Path to the transition marker file.
    dry_run : bool
        If True, log actions without executing them.
    """
    ops = PlatformOps(
        winpe_entry_ids=winpe_entry_ids,
        marker_path=marker_path,
        dry_run=dry_run
    )
    ops.rollback_transition()


def reboot(dry_run: bool = False) -> None:
    """Reboot the system immediately.

    On supported platforms this will invoke an immediate reboot.  On
    others the call is logged.  Use this with caution.

    Parameters
    ----------
    dry_run : bool
        If True, log actions without executing them.
    """
    ops = PlatformOps(winpe_entry_ids=[], marker_path="", dry_run=dry_run)
    ops.reboot()


def handover_control(dry_run: bool = False) -> None:
    """Perform any pre‑handover tasks before jumping to PE.

    This may include finalizing logs, persisting state, or performing
    final health checks before transitioning.

    Parameters
    ----------
    dry_run : bool
        If True, log actions without executing them.
    """
    ops = PlatformOps(winpe_entry_ids=[], marker_path="", dry_run=dry_run)
    ops.handover_control()