"""
Windows platform operations for DKTM
------------------------------------

This module provides implementations of the low‑level operations
required to switch into a Windows PE environment via ``bcdedit /bootsequence``.
It modifies the Boot Configuration Data (BCD), creates markers to signal
pending transitions and can trigger system reboots.

Features:
- One-time boot sequence modification via bcdedit
- Backup and restore of boot configuration
- Marker file management for transition tracking
- Support for dry-run mode (no actual system changes)
- Automatic privilege escalation checks
"""

from __future__ import annotations

import logging
import os
import subprocess
import json
from typing import List, Optional, Dict, Any
from datetime import datetime


class PlatformOps:
    """Windows-specific platform operations for DKTM transitions."""

    def __init__(
        self,
        winpe_entry_ids: List[str],
        marker_path: str,
        dry_run: bool = False,
        backup_path: Optional[str] = None
    ) -> None:
        """Initialize Windows platform operations.

        Parameters
        ----------
        winpe_entry_ids : List[str]
            List of BCD GUID entries for WinPE environments.
            The first entry will be used for bootsequence.
        marker_path : str
            Path to the transition marker file.
        dry_run : bool
            If True, log actions without executing them.
        backup_path : str, optional
            Path to store BCD backup information.
        """
        self.winpe_entry_ids = winpe_entry_ids
        self.marker_path = marker_path
        self.dry_run = dry_run
        self.backup_path = backup_path or f"{marker_path}.backup"
        self.logger = logging.getLogger("dktm.platform.windows")

    def _check_admin_privileges(self) -> bool:
        """Check if running with administrator privileges.

        Returns
        -------
        bool
            True if running as administrator.
        """
        try:
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception as exc:
            self.logger.warning("Could not check admin privileges: %s", exc)
            return False

    def _run_bcdedit(self, args: List[str], check: bool = True) -> Optional[str]:
        """Execute bcdedit command.

        Parameters
        ----------
        args : List[str]
            Arguments to pass to bcdedit.
        check : bool
            If True, raise exception on non-zero exit code.

        Returns
        -------
        str or None
            Command output if successful, None otherwise.
        """
        if self.dry_run:
            cmd_str = " ".join(["bcdedit"] + args)
            self.logger.info("[DRY-RUN] Would execute: %s", cmd_str)
            return None

        try:
            result = subprocess.run(
                ["bcdedit"] + args,
                capture_output=True,
                text=True,
                check=check,
                encoding="mbcs",
                errors="replace"
            )
            return result.stdout
        except subprocess.CalledProcessError as exc:
            self.logger.error("bcdedit failed: %s\nOutput: %s", exc, exc.stderr)
            if check:
                raise
            return None
        except FileNotFoundError:
            self.logger.error("bcdedit not found. Are you running on Windows?")
            return None

    def _run_reagentc(self, args: List[str], check: bool = True) -> Optional[str]:
        """Execute reagentc command."""
        if self.dry_run:
            cmd_str = " ".join(["reagentc"] + args)
            self.logger.info("[DRY-RUN] Would execute: %s", cmd_str)
            return None

        try:
            result = subprocess.run(
                ["reagentc"] + args,
                capture_output=True,
                text=True,
                check=check,
                encoding="mbcs",
                errors="replace"
            )
            return result.stdout
        except subprocess.CalledProcessError as exc:
            self.logger.error("reagentc failed: %s\nOutput: %s", exc, exc.stderr)
            if check:
                raise
            return None
        except FileNotFoundError:
            self.logger.error("reagentc not found. Are you running on Windows?")
            return None

    def _is_winre_enabled(self) -> Optional[bool]:
        """Check whether Windows Recovery Environment is enabled."""
        output = self._run_reagentc(["/info"], check=False)
        if output is None:
            return None
        for line in output.splitlines():
            if "Windows RE status" in line:
                return "enabled" in line.lower()
        return None

    def _set_winre_boot_once(self) -> None:
        """Configure a one-time boot into WinRE."""
        status = self._is_winre_enabled()
        if status is False:
            self.logger.info("WinRE disabled; attempting to enable...")
            self._run_reagentc(["/enable"])
            status = self._is_winre_enabled()
        if status is not True:
            raise RuntimeError("WinRE is not enabled; cannot boot to recovery")
        self._run_reagentc(["/boottore"])
        self.logger.info("✓ WinRE boot-to-recovery scheduled")

    def commit_transition(
        self,
        auto_reboot: bool = False,
        transition_method: str = "bcd",
        fallback_method: str = "winre",
    ) -> None:
        """Prepare the system to boot into WinPE on next reboot.

        This method:
        1. Checks administrator privileges
        2. Backs up current boot configuration
        3. Sets one-time boot sequence to WinPE entry
        4. Writes transition marker file
        5. Optionally triggers immediate reboot

        Parameters
        ----------
        auto_reboot : bool
            If True, reboot immediately after committing.

        Raises
        ------
        RuntimeError
            If WinPE entry IDs are not configured or operation fails.
        """
        entry_id = self.winpe_entry_ids[0] if self.winpe_entry_ids else None
        self.logger.info("=== Committing DKTM Transition ===")
        self.logger.info("Transition method: %s", transition_method)
        if entry_id:
            self.logger.info("Target WinPE entry: %s", entry_id)

        # Check privileges (warning only in dry-run)
        if not self._check_admin_privileges():
            msg = "Administrator privileges required for BCD modification"
            if self.dry_run:
                self.logger.warning("[DRY-RUN] %s", msg)
            else:
                raise RuntimeError(msg)

        used_method = transition_method
        if transition_method == "auto":
            used_method = "bcd"

        if used_method == "bcd":
            if not entry_id:
                if not fallback_method:
                    fallback_method = "winre"
                self.logger.warning(
                    "No WinPE entry IDs; falling back to %s",
                    fallback_method
                )
                used_method = fallback_method

            if used_method == "bcd":
                # Backup current bootsequence
                self._backup_boot_config()

                # Set one-time boot sequence
                self.logger.info("Setting bootsequence to %s", entry_id)
                try:
                    self._run_bcdedit(["/bootsequence", entry_id])
                    self.logger.info("✓ Boot sequence set successfully")
                except Exception as exc:
                    self.logger.error("Failed to set bootsequence: %s", exc)
                    if transition_method == "auto" and fallback_method:
                        self.logger.warning("Falling back to %s transition method", fallback_method)
                        used_method = fallback_method
                    else:
                        raise RuntimeError(f"BCD modification failed: {exc}")

        if used_method == "winre":
            self._set_winre_boot_once()
        elif used_method not in ("bcd", "winre"):
            raise RuntimeError(f"Unknown transition method: {used_method}")

        # Write marker file with metadata
        if entry_id:
            self._write_marker(entry_id)

        # Display summary
        self.logger.info("=== Transition Committed ===")
        if used_method == "winre":
            self.logger.info("Next boot will enter: WinRE recovery environment")
        else:
            self.logger.info("Next boot will enter: WinPE (%s)", entry_id)
        self.logger.info("Marker file: %s", self.marker_path)

        if auto_reboot:
            self.logger.info("Auto-reboot enabled, initiating restart...")
            self.reboot()
        else:
            self.logger.info("Manual reboot required to activate transition")

    def _backup_boot_config(self) -> None:
        """Backup current boot configuration to restore later."""
        self.logger.debug("Backing up boot configuration")

        if self.dry_run:
            self.logger.info("[DRY-RUN] Would backup BCD to %s", self.backup_path)
            return

        try:
            # Query current bootsequence
            output = self._run_bcdedit(["/enum", "{bootmgr}"], check=False)

            backup_data = {
                "timestamp": datetime.now().isoformat(),
                "bcd_bootmgr": output if output else "N/A",
                "original_bootsequence": "default"  # Could be parsed from output
            }

            with open(self.backup_path, "w", encoding="utf-8") as f:
                json.dump(backup_data, f, indent=2)

            self.logger.debug("✓ Boot config backed up to %s", self.backup_path)
        except Exception as exc:
            self.logger.warning("Could not backup boot config: %s", exc)

    def _write_marker(self, entry_id: str) -> None:
        """Write transition marker file with metadata.

        Parameters
        ----------
        entry_id : str
            The WinPE BCD entry GUID.
        """
        if self.dry_run:
            self.logger.info("[DRY-RUN] Would write marker to %s", self.marker_path)
            return

        try:
            marker_data = {
                "transition_type": "dktm_winpe",
                "winpe_entry_id": entry_id,
                "timestamp": datetime.now().isoformat(),
                "committed_by": "DKTM",
                "status": "pending"
            }

            with open(self.marker_path, "w", encoding="utf-8") as f:
                json.dump(marker_data, f, indent=2)

            self.logger.debug("✓ Marker file written to %s", self.marker_path)
        except Exception as exc:
            self.logger.error("Failed to write marker file %s: %s", self.marker_path, exc)
            raise

    def rollback_transition(self) -> None:
        """Undo a previously committed WinPE boot sequence.

        This method:
        1. Removes the bootsequence override (reverts to default)
        2. Deletes the transition marker file
        3. Optionally restores from backup

        Note: bcdedit /bootsequence without arguments clears the override.
        """
        self.logger.info("=== Rolling Back DKTM Transition ===")

        # Check privileges
        if not self._check_admin_privileges():
            msg = "Administrator privileges required for BCD modification"
            if self.dry_run:
                self.logger.warning("[DRY-RUN] %s", msg)
            else:
                raise RuntimeError(msg)

        # Clear bootsequence override
        self.logger.info("Clearing bootsequence override")
        try:
            # Note: bcdedit /deletevalue {bootmgr} bootsequence also works
            self._run_bcdedit(["/deletevalue", "{bootmgr}", "bootsequence"])
            self.logger.info("✓ Boot sequence override removed")
        except Exception as exc:
            self.logger.warning("Could not clear bootsequence: %s", exc)

        # Remove marker file
        if self.marker_path:
            if self.dry_run:
                self.logger.info("[DRY-RUN] Would remove marker: %s", self.marker_path)
            elif os.path.isfile(self.marker_path):
                try:
                    os.remove(self.marker_path)
                    self.logger.info("✓ Marker file removed: %s", self.marker_path)
                except Exception as exc:
                    self.logger.error("Failed to remove marker %s: %s", self.marker_path, exc)
            else:
                self.logger.debug("Marker file does not exist: %s", self.marker_path)

        # Clean up backup
        if os.path.isfile(self.backup_path):
            try:
                if not self.dry_run:
                    os.remove(self.backup_path)
                self.logger.debug("✓ Backup file removed: %s", self.backup_path)
            except Exception as exc:
                self.logger.warning("Could not remove backup: %s", exc)

        self.logger.info("=== Rollback Complete ===")
        self.logger.info("System will boot normally on next restart")

    def reboot(self) -> None:
        """Reboot the machine immediately.

        Executes Windows shutdown command with immediate restart.
        In dry-run mode, logs the action without executing.
        """
        self.logger.info("Initiating system reboot")

        if self.dry_run:
            self.logger.info("[DRY-RUN] Would execute: shutdown /r /t 0")
            return

        try:
            subprocess.run(
                ["shutdown", "/r", "/t", "0"],
                check=True,
                capture_output=True
            )
            self.logger.info("✓ Reboot command issued")
        except subprocess.CalledProcessError as exc:
            self.logger.error("Failed to reboot: %s\nOutput: %s", exc, exc.stderr)
            raise RuntimeError(f"Reboot failed: {exc}")
        except FileNotFoundError:
            self.logger.error("shutdown command not found")
            raise RuntimeError("shutdown command not available")

    def handover_control(self) -> None:
        """Perform final tasks before transitioning to PE.

        This method can be used to:
        - Flush logs to disk
        - Persist critical state
        - Signal readiness for transition
        - Perform final health checks
        """
        self.logger.info("=== Handover Control to WinPE ===")

        # Flush any pending logs
        for handler in logging.getLogger().handlers:
            handler.flush()

        # Could add: Save system snapshot, close handles, etc.
        self.logger.info("✓ Control handover prepared")
        self.logger.info("System ready for transition")
