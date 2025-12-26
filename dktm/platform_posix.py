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
from typing import List, Optional


class PlatformOps:
    def __init__(
        self,
        winpe_entry_ids: List[str],
        marker_path: str,
        dry_run: bool = False,
        backup_path: Optional[str] = None
    ) -> None:
        """Initialize POSIX platform operations (stub implementation).

        Parameters
        ----------
        winpe_entry_ids : List[str]
            List of WinPE entry IDs (unused on POSIX).
        marker_path : str
            Path to the transition marker file.
        dry_run : bool
            If True, log actions without executing them.
        backup_path : str, optional
            Backup path (unused on POSIX).
        """
        self.winpe_entry_ids = winpe_entry_ids
        self.marker_path = marker_path
        self.dry_run = dry_run
        self.logger = logging.getLogger("dktm.platform.posix")

    def commit_transition(self, auto_reboot: bool = False) -> None:
        """Stub implementation of commit_transition for POSIX.

        Creates a marker file to simulate a pending transition and logs
        the requested bootsequence.  No boot order changes are made.
        """
        mode = "[DRY-RUN]" if self.dry_run else "[POSIX]"
        self.logger.info(
            "%s commit_transition called with winpe_entry_ids=%s, marker_path=%s, auto_reboot=%s",
            mode,
            self.winpe_entry_ids,
            self.marker_path,
            auto_reboot,
        )
        # Write marker file
        if self.marker_path and not self.dry_run:
            try:
                with open(self.marker_path, "w", encoding="utf-8") as f:
                    f.write("pending_transition")
                self.logger.debug("Marker file written to %s", self.marker_path)
            except Exception as exc:
                self.logger.error("Failed to write marker file %s: %s", self.marker_path, exc)
        elif self.dry_run:
            self.logger.info("[DRY-RUN] Would write marker to %s", self.marker_path)
        # Optionally simulate reboot
        if auto_reboot:
            self.reboot()

    def rollback_transition(self) -> None:
        """Stub implementation of rollback_transition for POSIX.

        Removes the marker file if present.
        """
        mode = "[DRY-RUN]" if self.dry_run else "[POSIX]"
        self.logger.info(
            "%s rollback_transition called with winpe_entry_ids=%s, marker_path=%s",
            mode,
            self.winpe_entry_ids,
            self.marker_path,
        )
        if self.marker_path:
            if self.dry_run:
                self.logger.info("[DRY-RUN] Would remove marker: %s", self.marker_path)
            elif os.path.isfile(self.marker_path):
                try:
                    os.remove(self.marker_path)
                    self.logger.debug("Marker file %s removed", self.marker_path)
                except Exception as exc:
                    self.logger.error("Failed to remove marker file %s: %s", self.marker_path, exc)

    def reboot(self) -> None:
        """Stub implementation of reboot for POSIX.

        Logs an informational message rather than actually rebooting.
        """
        mode = "[DRY-RUN]" if self.dry_run else "[POSIX]"
        self.logger.info("%s Reboot requested (stub, no action)", mode)

    def handover_control(self) -> None:
        """Stub implementation of handover_control for POSIX.

        Logs a message indicating that control handover would occur.
        """
        mode = "[DRY-RUN]" if self.dry_run else "[POSIX]"
        self.logger.info("%s handover_control invoked (stub)", mode)