"""
POSIX platform operations for DKTM
----------------------------------

This module implements stubbed platform operations for POSIX systems
such as Linux and macOS.  The real DKTM transition mechanism is
intended for Windows, so POSIX implementations simply log actions
without changing system state.  These stubs allow the DKTM code to
run unmodified on developer machines and in CI pipelines.
"""

from __future__ import annotations

import logging
import os
from typing import List


class PlatformOps:
    def __init__(self, winpe_entry_ids: List[str], marker_path: str) -> None:
        self.winpe_entry_ids = winpe_entry_ids
        self.marker_path = marker_path
        self.logger = logging.getLogger("dktm.platform.posix")

    def commit_transition(self, auto_reboot: bool = False) -> None:
        """Stub implementation of commit_transition for POSIX.

        Creates a marker file to simulate a pending transition and logs
        the requested bootsequence.  No boot order changes are made.
        """
        self.logger.info(
            "[POSIX] commit_transition called with winpe_entry_ids=%s, marker_path=%s, auto_reboot=%s",
            self.winpe_entry_ids,
            self.marker_path,
            auto_reboot,
        )
        # Write marker file
        if self.marker_path:
            try:
                with open(self.marker_path, "w", encoding="utf-8") as f:
                    f.write("pending_transition")
                self.logger.debug("Marker file written to %s", self.marker_path)
            except Exception as exc:
                self.logger.error("Failed to write marker file %s: %s", self.marker_path, exc)
        # Optionally simulate reboot
        if auto_reboot:
            self.reboot()

    def rollback_transition(self) -> None:
        """Stub implementation of rollback_transition for POSIX.

        Removes the marker file if present.
        """
        self.logger.info(
            "[POSIX] rollback_transition called with winpe_entry_ids=%s, marker_path=%s",
            self.winpe_entry_ids,
            self.marker_path,
        )
        if self.marker_path and os.path.isfile(self.marker_path):
            try:
                os.remove(self.marker_path)
                self.logger.debug("Marker file %s removed", self.marker_path)
            except Exception as exc:
                self.logger.error("Failed to remove marker file %s: %s", self.marker_path, exc)

    def reboot(self) -> None:
        """Stub implementation of reboot for POSIX.

        Logs an informational message rather than actually rebooting.
        """
        self.logger.info("[POSIX] Reboot requested (stub, no action)")

    def handover_control(self) -> None:
        """Stub implementation of handover_control for POSIX.

        Logs a message indicating that control handover would occur.
        """
        self.logger.info("[POSIX] handover_control invoked (stub)")