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
from typing import List

import logging

if sys.platform.startswith("win"):
    from .platform_windows import PlatformOps  # type: ignore
else:
    from .platform_posix import PlatformOps  # type: ignore

logger = logging.getLogger("dktm.platform_ops")


def commit_transition(winpe_entry_ids: List[str], marker_path: str, auto_reboot: bool) -> None:
    """Commit a transition to a secondary environment.

    For Windows this will set the bootsequence to point to the first
    WinPE entry in ``winpe_entry_ids``, write a marker file at
    ``marker_path`` and optionally reboot the machine.  On POSIX
    systems the call is logged but otherwise ignored.
    """
    ops = PlatformOps(winpe_entry_ids=winpe_entry_ids, marker_path=marker_path)
    ops.commit_transition(auto_reboot=auto_reboot)


def rollback_transition(winpe_entry_ids: List[str], marker_path: str) -> None:
    """Rollback a previously committed transition.

    On Windows this restores the previous boot order and removes the
    marker file.  On POSIX systems the call is logged but otherwise
    ignored.
    """
    ops = PlatformOps(winpe_entry_ids=winpe_entry_ids, marker_path=marker_path)
    ops.rollback_transition()


def reboot() -> None:
    """Reboot the system immediately.

    On supported platforms this will invoke an immediate reboot.  On
    others the call is logged.  Use this with caution.
    """
    ops = PlatformOps(winpe_entry_ids=[], marker_path="")
    ops.reboot()


def handover_control() -> None:
    """Perform any pre‑handover tasks before jumping to PE.

    Currently this is a no‑op placeholder.  It may be extended to
    include final checks or state dumps before transitioning.
    """
    ops = PlatformOps(winpe_entry_ids=[], marker_path="")
    ops.handover_control()