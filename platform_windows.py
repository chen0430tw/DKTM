"""
Windows platform operations for DKTM
------------------------------------

This module provides placeholder implementations of the low‑level
operations required to switch into a Windows PE environment via
``bcdedit /bootsequence``.  Real implementations should modify the
Boot Configuration Data (BCD), create a marker to signal a pending
transition and reboot the system.  These functions currently log
their intent without performing any changes to avoid side effects in
development environments.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import List


class PlatformOps:
    def __init__(self, winpe_entry_ids: List[str], marker_path: str) -> None:
        self.winpe_entry_ids = winpe_entry_ids
        self.marker_path = marker_path
        self.logger = logging.getLogger("dktm.platform.windows")

    def commit_transition(self, auto_reboot: bool = False) -> None:
        """Prepare the system to boot into WinPE on next reboot.

        In a real implementation this would use ``bcdedit /bootsequence``
        to set a one‑time boot entry and write a marker file.  It may
        also trigger an immediate reboot if requested.  Here we log
        actions instead to avoid modifying the host.
        """
        if not self.winpe_entry_ids:
            self.logger.warning("No WinPE entry IDs configured; cannot commit transition")
            return
        entry_id = self.winpe_entry_ids[0]
        self.logger.info("[Windows] Setting bootsequence to %s", entry_id)
        # Simulate bcdedit call
        try:
            # In a real implementation one would call:
            # subprocess.run(["bcdedit", "/bootsequence", entry_id], check=True)
            pass
        except Exception as exc:
            self.logger.error("Failed to set bootsequence: %s", exc)
            return
        # Write marker file
        if self.marker_path:
            try:
                with open(self.marker_path, "w", encoding="utf-8") as f:
                    f.write(entry_id)
                self.logger.debug("Marker file written to %s", self.marker_path)
            except Exception as exc:
                self.logger.error("Failed to write marker file %s: %s", self.marker_path, exc)
        # Auto reboot if requested
        if auto_reboot:
            self.reboot()

    def rollback_transition(self) -> None:
        """Undo a previously committed WinPE boot sequence.

        Real implementations would remove the bootsequence override and
        delete the marker file.  Here we only remove the marker.
        """
        self.logger.info("[Windows] Rolling back transition")
        # Remove marker file
        if self.marker_path and os.path.isfile(self.marker_path):
            try:
                os.remove(self.marker_path)
                self.logger.debug("Marker file %s removed", self.marker_path)
            except Exception as exc:
                self.logger.error("Failed to remove marker file %s: %s", self.marker_path, exc)

    def reboot(self) -> None:
        """Reboot the machine immediately.

        Executes a shutdown command.  In development we log the
        action instead.
        """
        self.logger.info("[Windows] Reboot requested")
        try:
            # Real call would be:
            # subprocess.run(["shutdown", "/r", "/t", "0"], check=True)
            pass
        except Exception as exc:
            self.logger.error("Failed to reboot: %s", exc)

    def handover_control(self) -> None:
        """Perform any final tasks before handing off to PE.

        Currently a no‑op; real implementations might finalise logs or
        persist state.
        """
        self.logger.info("[Windows] Handover control invoked (stub)")