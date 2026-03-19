"""
Windows platform operations for DKTM
-----------------------------------

This module provides implementations of the low-level operations
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
from typing import List, Optional
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

    # ------------------------------------------------------------------
    # BCD file helpers (fallback when bcdedit is blocked)
    # ------------------------------------------------------------------
    _BOOTMGR_GUID = "{9dea862c-5cdd-4e70-acc1-f32b344d4795}"
    _BCD_FILE_PATH = r"C:\Boot\BCD"
    _BCD_BACKUP_PATH = r"C:\Boot\BCD.dktm.bak"
    _TMP_HIVE = r"HKLM\TmpDKTM"
    _BOOTSEQUENCE_ELEM = "24000002"
    _DISPLAYORDER_ELEM = "24000001"

    # Our own WinPE GUID — fixed because we write it ourselves via bcd_add_winpe.py.
    # The WinRE GUID is machine-specific and must be discovered at runtime.
    _DKTM_WINPE_GUID = "{7619dcc9-fafe-11d9-b411-000476eba25f}"

    def _bcd_file_available(self) -> bool:
        """Return True if C:\\Boot\\BCD is present and readable."""
        return os.path.isfile(self._BCD_FILE_PATH)

    def _discover_bcd_entries(self) -> List[str]:
        """Discover bootable WinPE/WinRE GUIDs from the BCD at runtime.

        Search order:
        1. Our own DKTM WinPE entry (fixed GUID, added by bcd_add_winpe.py).
        2. Any WinRE OS loader found in BCD (type 0x10200003 with a
           description containing "recovery" or "RE", case-insensitive).

        Returns a list of GUIDs in priority order.  The list may be empty
        if neither entry is present in the current BCD.
        """
        import winreg

        if not self._bcd_file_available():
            return []

        r = subprocess.run(
            ["reg", "load", self._TMP_HIVE, self._BCD_FILE_PATH],
            capture_output=True, encoding="mbcs", errors="replace"
        )
        if r.returncode != 0:
            self.logger.debug("_discover_bcd_entries: reg load failed")
            return []

        found: List[str] = []
        try:
            hive_short = self._TMP_HIVE[len("HKLM\\"):]
            objects_path = f"{hive_short}\\Objects"

            # Enumerate all BCD objects
            try:
                k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, objects_path,
                                   0, winreg.KEY_READ)
            except OSError:
                return []

            guids: List[str] = []
            idx = 0
            while True:
                try:
                    guids.append(winreg.EnumKey(k, idx))
                    idx += 1
                except OSError:
                    break
            winreg.CloseKey(k)

            # Check each object
            our_winpe_present = False
            winre_guids: List[str] = []

            for guid in guids:
                base = f"{objects_path}\\{guid}"
                try:
                    dk = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                        base + "\\Description", 0, winreg.KEY_READ)
                    obj_type, _ = winreg.QueryValueEx(dk, "Type")
                    winreg.CloseKey(dk)
                except OSError:
                    continue

                if isinstance(obj_type, bytes):
                    obj_type = int.from_bytes(obj_type, "little")

                if guid.lower() == self._DKTM_WINPE_GUID.lower():
                    our_winpe_present = True
                    continue

                # WinRE OS loaders have type 0x10200003
                # Detect by device element containing "winre.wim" (locale-independent)
                if obj_type == 0x10200003:
                    try:
                        ek = winreg.OpenKey(
                            winreg.HKEY_LOCAL_MACHINE,
                            base + "\\Elements\\11000001", 0, winreg.KEY_READ
                        )
                        dev_val, dev_type = winreg.QueryValueEx(ek, "Element")
                        winreg.CloseKey(ek)
                        if dev_type == winreg.REG_BINARY:
                            decoded = bytes(dev_val).decode(
                                "utf-16-le", errors="ignore"
                            ).lower()
                            if "winre.wim" in decoded:
                                winre_guids.append(guid)
                    except OSError:
                        pass

            if our_winpe_present:
                found.append(self._DKTM_WINPE_GUID)
            found.extend(winre_guids)

        finally:
            subprocess.run(
                ["reg", "unload", self._TMP_HIVE],
                capture_output=True, encoding="mbcs", errors="replace"
            )

        self.logger.info("Auto-discovered BCD entries: %s", found)
        return found

    def _resolve_winpe_entry_ids(self) -> List[str]:
        """Return the effective WinPE entry list.

        If the instance was configured with explicit IDs (from config.yaml),
        use those.  Otherwise, auto-discover from the BCD.  Always logs the
        source of the IDs so it's clear where they came from.
        """
        if self.winpe_entry_ids:
            self.logger.info(
                "Using configured WinPE entry IDs: %s", self.winpe_entry_ids
            )
            return self.winpe_entry_ids

        discovered = self._discover_bcd_entries()
        if discovered:
            self.logger.info(
                "Auto-discovered WinPE entry IDs: %s", discovered
            )
        else:
            self.logger.warning(
                "No WinPE/WinRE BCD entries found — "
                "run bcd_add_winpe.py or configure winpe_entry_ids in config.yaml"
            )
        return discovered

    def _backup_bcd_file(self) -> None:
        """Copy C:\\Boot\\BCD to C:\\Boot\\BCD.dktm.bak before modification."""
        import shutil
        shutil.copy2(self._BCD_FILE_PATH, self._BCD_BACKUP_PATH)
        self.logger.info("✓ BCD backed up to %s", self._BCD_BACKUP_PATH)

    def _restore_bcd_file(self) -> None:
        """Overwrite C:\\Boot\\BCD from backup, recovering from a failed write."""
        import shutil
        if os.path.isfile(self._BCD_BACKUP_PATH):
            shutil.copy2(self._BCD_BACKUP_PATH, self._BCD_FILE_PATH)
            self.logger.warning("BCD restored from backup %s", self._BCD_BACKUP_PATH)
        else:
            self.logger.error("Backup %s not found — cannot restore BCD", self._BCD_BACKUP_PATH)

    def _verify_bcd_integrity(self, expected_bootsequence_guid: Optional[str] = None) -> None:
        """Load BCD and verify critical invariants.

        Checks performed:
        1. bootmgr object is present and queryable.
        2. displayorder element (24000001) exists and is non-empty — proves
           the bootmgr entry itself is intact.
        3. bootsequence element (24000002) state matches expectation:
           - If ``expected_bootsequence_guid`` is given, the element must
             exist with the correct 16-byte GUID encoding.
           - If ``None``, the element must be absent (post-rollback state).

        Parameters
        ----------
        expected_bootsequence_guid : str or None
            GUID string expected in the bootsequence element, or ``None``
            to assert the element has been removed.

        Raises
        ------
        RuntimeError
            If any invariant is violated.
        """
        import winreg

        r = subprocess.run(
            ["reg", "load", self._TMP_HIVE, self._BCD_FILE_PATH],
            capture_output=True, encoding="mbcs", errors="replace"
        )
        if r.returncode != 0:
            raise RuntimeError(f"Integrity check: reg load failed: {r.stderr.strip()}")

        issues = []
        try:
            bootmgr = "{" + self._BOOTMGR_GUID.strip("{}") + "}"
            hive_short = self._TMP_HIVE[len("HKLM\\"):]

            # --- 1. bootmgr object exists ---
            bootmgr_path = f"{hive_short}\\Objects\\{bootmgr}"
            try:
                k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, bootmgr_path,
                                   0, winreg.KEY_READ)
                winreg.CloseKey(k)
            except OSError:
                issues.append("bootmgr object missing from BCD")

            # --- 2. displayorder element present and non-empty ---
            disp_path = f"{bootmgr_path}\\Elements\\{self._DISPLAYORDER_ELEM}"
            try:
                k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, disp_path,
                                   0, winreg.KEY_READ)
                val, _ = winreg.QueryValueEx(k, "Element")
                winreg.CloseKey(k)
                if not val:
                    issues.append("displayorder element is empty")
            except OSError:
                issues.append("displayorder element (24000001) missing from bootmgr")

            # --- 3. bootsequence element state ---
            boot_path = f"{bootmgr_path}\\Elements\\{self._BOOTSEQUENCE_ELEM}"
            try:
                k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, boot_path,
                                   0, winreg.KEY_READ)
                val, _ = winreg.QueryValueEx(k, "Element")
                winreg.CloseKey(k)
                if expected_bootsequence_guid is None:
                    issues.append("bootsequence element still present after rollback")
                else:
                    expected_bytes = bytes.fromhex(
                        self._guid_to_binary_hex(expected_bootsequence_guid)
                    )
                    if val != expected_bytes:
                        issues.append(
                            f"bootsequence mismatch: got {val.hex()} "
                            f"expected {expected_bytes.hex()}"
                        )
            except OSError:
                if expected_bootsequence_guid is not None:
                    issues.append("bootsequence element missing after commit")
                # else: absent after rollback — correct
        finally:
            subprocess.run(
                ["reg", "unload", self._TMP_HIVE],
                capture_output=True, encoding="mbcs", errors="replace"
            )

        if issues:
            raise RuntimeError("BCD integrity check failed: " + "; ".join(issues))
        self.logger.info("✓ BCD integrity verified")

    def _guid_to_binary_hex(self, guid_str: str) -> str:
        """Convert a GUID string to 16-byte Windows binary hex (little-endian fields).

        Windows GUID binary layout:
          Data1 (4 B, LE) | Data2 (2 B, LE) | Data3 (2 B, LE) | Data4 (8 B, BE)
        """
        g = guid_str.strip("{}").replace("-", "")
        d1 = bytes.fromhex(g[0:8])[::-1].hex()
        d2 = bytes.fromhex(g[8:12])[::-1].hex()
        d3 = bytes.fromhex(g[12:16])[::-1].hex()
        d4 = g[16:32]  # already big-endian
        return d1 + d2 + d3 + d4

    def _enable_backup_restore_privileges(self) -> None:
        """Enable SeBackupPrivilege and SeRestorePrivilege on the current process token.

        These privileges allow reading and writing registry keys regardless of
        the key's DACL, which is required to modify the BCD hive.
        """
        import ctypes
        import ctypes.wintypes

        advapi32 = ctypes.windll.advapi32
        kernel32 = ctypes.windll.kernel32

        advapi32.OpenProcessToken.argtypes = [
            ctypes.c_void_p,
            ctypes.wintypes.DWORD,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        advapi32.OpenProcessToken.restype = ctypes.wintypes.BOOL

        token = ctypes.c_void_p()
        if not advapi32.OpenProcessToken(-1, 0x0028, ctypes.byref(token)):
            raise RuntimeError(
                f"OpenProcessToken failed: {kernel32.GetLastError()}"
            )

        class _LUID(ctypes.Structure):
            _fields_ = [
                ("LowPart", ctypes.wintypes.DWORD),
                ("HighPart", ctypes.wintypes.LONG),
            ]

        class _LA(ctypes.Structure):
            _fields_ = [("Luid", _LUID), ("Attributes", ctypes.wintypes.DWORD)]

        class _TP(ctypes.Structure):
            _fields_ = [
                ("Count", ctypes.wintypes.DWORD),
                ("Privs", _LA * 2),
            ]

        tp = _TP()
        tp.Count = 2
        for i, name in enumerate(("SeBackupPrivilege", "SeRestorePrivilege")):
            advapi32.LookupPrivilegeValueW(None, name, ctypes.byref(tp.Privs[i].Luid))
            tp.Privs[i].Attributes = 0x00000002  # SE_PRIVILEGE_ENABLED

        advapi32.AdjustTokenPrivileges(token.value, False, ctypes.byref(tp), 0, None, None)
        kernel32.CloseHandle(token)

    def _reg_write_bcd_bootsequence(self, winpe_guid: str) -> None:
        """Write bootsequence via winreg with REG_OPTION_BACKUP_RESTORE."""
        import ctypes
        import winreg

        advapi32 = ctypes.windll.advapi32
        advapi32.RegCreateKeyExW.argtypes = [
            ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_wchar_p,
            ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_ulong),
        ]
        advapi32.RegCreateKeyExW.restype = ctypes.c_long
        advapi32.RegSetValueExW.argtypes = [
            ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_ulong,
            ctypes.c_ulong, ctypes.c_char_p, ctypes.c_ulong,
        ]
        advapi32.RegSetValueExW.restype = ctypes.c_long

        # Hive root is the short name under HKLM (without "HKLM\")
        hive_short = self._TMP_HIVE[len("HKLM\\"):]
        bootmgr = "{" + self._BOOTMGR_GUID.strip("{}") + "}"
        subkey = (
            f"{hive_short}\\Objects\\{bootmgr}"
            f"\\Elements\\{self._BOOTSEQUENCE_ELEM}"
        )
        normalized = "{" + winpe_guid.strip("{}") + "}"
        data = bytes.fromhex(self._guid_to_binary_hex(normalized))

        hkey = ctypes.c_void_p()
        disp = ctypes.c_ulong()
        ret = advapi32.RegCreateKeyExW(
            ctypes.c_void_p(0x80000002),  # HKEY_LOCAL_MACHINE
            subkey,
            0, None,
            4,          # REG_OPTION_BACKUP_RESTORE – bypasses DACL
            0x0006,     # KEY_SET_VALUE | KEY_CREATE_SUB_KEY
            None,
            ctypes.byref(hkey), ctypes.byref(disp),
        )
        if ret != 0:
            raise RuntimeError(f"RegCreateKeyExW failed: {ret}")

        ret = advapi32.RegSetValueExW(hkey.value, "Element", 0, 3, data, len(data))
        advapi32.RegCloseKey(hkey.value)
        if ret != 0:
            raise RuntimeError(f"RegSetValueExW failed: {ret}")

    def _reg_delete_bcd_bootsequence(self) -> None:
        """Delete bootsequence element via winreg with REG_OPTION_BACKUP_RESTORE."""
        import ctypes

        advapi32 = ctypes.windll.advapi32
        advapi32.RegOpenKeyExW.argtypes = [
            ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_ulong,
            ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p),
        ]
        advapi32.RegOpenKeyExW.restype = ctypes.c_long
        advapi32.RegDeleteKeyW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        advapi32.RegDeleteKeyW.restype = ctypes.c_long

        hive_short = self._TMP_HIVE[len("HKLM\\"):]
        bootmgr = "{" + self._BOOTMGR_GUID.strip("{}") + "}"
        parent_subkey = f"{hive_short}\\Objects\\{bootmgr}\\Elements"

        hparent = ctypes.c_void_p()
        ret = advapi32.RegOpenKeyExW(
            ctypes.c_void_p(0x80000002),
            parent_subkey,
            4,          # REG_OPTION_BACKUP_RESTORE
            0x000F003F, # KEY_ALL_ACCESS
            ctypes.byref(hparent),
        )
        if ret != 0:
            self.logger.warning("Could not open Elements key for delete: %d", ret)
            return

        r = advapi32.RegDeleteKeyW(hparent.value, self._BOOTSEQUENCE_ELEM)
        advapi32.RegCloseKey(hparent.value)
        if r != 0:
            self.logger.warning("RegDeleteKeyW returned: %d (element may not exist)", r)

    def _commit_via_bcd_file(self, winpe_guid: str) -> None:
        """Set bootsequence by directly editing C:\\Boot\\BCD.

        Process: enable privileges → reg load → winreg write → reg unload.
        The hive is always unloaded, even if writing fails.

        Parameters
        ----------
        winpe_guid : str
            BCD entry GUID for the WinPE environment, e.g.
            ``{e18e67af-6278-11e6-a822-9a545abf3b29}``.

        Raises
        ------
        RuntimeError
            If the hive cannot be loaded or the element write fails.
        """
        self.logger.info("Using direct BCD file method (bcdedit not available)")
        self._enable_backup_restore_privileges()
        self._backup_bcd_file()

        r = subprocess.run(
            ["reg", "load", self._TMP_HIVE, self._BCD_FILE_PATH],
            capture_output=True, encoding="mbcs", errors="replace"
        )
        if r.returncode != 0:
            raise RuntimeError(f"reg load BCD failed: {r.stderr.strip()}")
        self.logger.info("✓ BCD hive loaded")

        success = False
        try:
            normalized = "{" + winpe_guid.strip("{}") + "}"
            bootmgr = "{" + self._BOOTMGR_GUID.strip("{}") + "}"
            self._reg_write_bcd_bootsequence(winpe_guid)
            self.logger.info("✓ Bootsequence element written (%s → %s)", bootmgr, normalized)
            success = True
        finally:
            subprocess.run(
                ["reg", "unload", self._TMP_HIVE],
                capture_output=True, encoding="mbcs", errors="replace"
            )
            if success:
                self.logger.info("✓ BCD hive flushed and unloaded")
            else:
                self.logger.warning("BCD hive unloaded after write error")

        # Integrity check — restore backup on failure
        try:
            self._verify_bcd_integrity(expected_bootsequence_guid=winpe_guid)
        except RuntimeError as exc:
            self.logger.error("BCD integrity check failed after commit: %s", exc)
            self.logger.error("Restoring BCD from backup...")
            self._restore_bcd_file()
            raise

    def _rollback_via_bcd_file(self) -> None:
        """Remove the bootsequence element from C:\\Boot\\BCD directly.

        Raises
        ------
        RuntimeError
            If the hive cannot be loaded.
        """
        self.logger.info("Removing bootsequence via direct BCD file method")
        self._enable_backup_restore_privileges()
        self._backup_bcd_file()

        r = subprocess.run(
            ["reg", "load", self._TMP_HIVE, self._BCD_FILE_PATH],
            capture_output=True, encoding="mbcs", errors="replace"
        )
        if r.returncode != 0:
            raise RuntimeError(f"reg load BCD failed: {r.stderr.strip()}")
        self.logger.info("✓ BCD hive loaded")

        try:
            self._reg_delete_bcd_bootsequence()
            self.logger.info("✓ Bootsequence element removed")
        finally:
            subprocess.run(
                ["reg", "unload", self._TMP_HIVE],
                capture_output=True, encoding="mbcs", errors="replace"
            )
            self.logger.info("✓ BCD hive unloaded")

        # Integrity check — restore backup on failure
        try:
            self._verify_bcd_integrity(expected_bootsequence_guid=None)
        except RuntimeError as exc:
            self.logger.error("BCD integrity check failed after rollback: %s", exc)
            self.logger.error("Restoring BCD from backup...")
            self._restore_bcd_file()
            raise

    # ------------------------------------------------------------------
    # bcdedit / reagentc helpers
    # ------------------------------------------------------------------
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
        effective_ids = self._resolve_winpe_entry_ids()
        entry_id = effective_ids[0] if effective_ids else None
        self.logger.info("=== Committing DKTM Transition ===")
        self.logger.info("Transition method: %s", transition_method)
        if entry_id:
            self.logger.info("Target WinPE entry: %s", entry_id)

        # Validate configuration — use effective_ids (discovered), not the raw
        # configured list which may intentionally be empty (auto-discover mode).
        if not effective_ids and transition_method == "bcd":
            if not fallback_method:
                fallback_method = "winre"
            self.logger.warning("No WinPE/WinRE entry found in BCD; falling back to %s", fallback_method)
            transition_method = fallback_method

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
                # should have been handled above, but guard anyway
                if not fallback_method:
                    fallback_method = "winre"
                self.logger.warning("No WinPE entry IDs; falling back to %s", fallback_method)
                used_method = fallback_method

            if used_method == "bcd":
                # Backup current bootsequence
                self._backup_boot_config()

                # Set one-time boot sequence — try bcdedit first, then BCD file
                self.logger.info("Setting bootsequence to %s", entry_id)
                bcd_set = False
                try:
                    self._run_bcdedit(["/bootsequence", entry_id])
                    self.logger.info("✓ Boot sequence set via bcdedit")
                    bcd_set = True
                except Exception as exc:
                    self.logger.warning("bcdedit failed (%s); trying direct BCD file", exc)

                if not bcd_set:
                    if self._bcd_file_available():
                        try:
                            self._commit_via_bcd_file(entry_id)
                            bcd_set = True
                        except Exception as exc2:
                            self.logger.error("Direct BCD file method failed: %s", exc2)
                    else:
                        self.logger.warning("C:\\Boot\\BCD not found; cannot use file method")

                if not bcd_set:
                    if transition_method == "auto" and fallback_method:
                        self.logger.warning("Falling back to %s transition method", fallback_method)
                        used_method = fallback_method
                    else:
                        raise RuntimeError("All BCD modification methods failed")

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

        # Clear bootsequence override — try bcdedit first, then BCD file
        self.logger.info("Clearing bootsequence override")
        cleared = False
        try:
            self._run_bcdedit(["/deletevalue", "{bootmgr}", "bootsequence"])
            self.logger.info("✓ Boot sequence override removed via bcdedit")
            cleared = True
        except Exception as exc:
            self.logger.warning("bcdedit deletevalue failed (%s); trying direct BCD file", exc)

        if not cleared and self._bcd_file_available():
            try:
                self._rollback_via_bcd_file()
                cleared = True
            except Exception as exc2:
                self.logger.warning("Direct BCD file rollback failed: %s", exc2)

        if not cleared:
            self.logger.warning("Could not remove bootsequence override; it may expire on next boot")

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
        """Perform final tasks before transitioning to PE."""
        self.logger.info("=== Handover Control to WinPE ===")
        for handler in logging.getLogger().handlers:
            handler.flush()
        self.logger.info("✓ Control handover prepared")
        self.logger.info("System ready for transition")

    def freeze_services(self) -> None:
        """Stop non-essential services before transition.

        Stops a conservative set of services that may be writing data
        (spooler, SysMain, WSearch). Failures are logged but not fatal.
        """
        services = ["spooler", "SysMain", "WSearch"]
        self.logger.info("Freezing non-essential services: %s", ", ".join(services))

        if self.dry_run:
            for svc in services:
                self.logger.info("[DRY-RUN] Would stop service: %s", svc)
            return

        for svc in services:
            try:
                result = subprocess.run(
                    ["sc", "stop", svc],
                    capture_output=True, text=True,
                    encoding="mbcs", errors="replace"
                )
                if result.returncode == 0:
                    self.logger.info("✓ Stopped service: %s", svc)
                else:
                    # Not running or access denied — not fatal
                    self.logger.debug("Service %s: %s", svc, result.stderr.strip())
            except Exception as exc:
                self.logger.debug("Could not stop service %s: %s", svc, exc)

    def flush_buffers(self) -> None:
        """Flush filesystem write buffers on all fixed volumes.

        Uses FlushFileBuffers via ctypes on each fixed drive volume
        to ensure pending writes reach disk before reboot.
        Requires administrator privileges.
        """
        self.logger.info("Flushing filesystem buffers...")

        if self.dry_run:
            self.logger.info("[DRY-RUN] Would flush filesystem buffers")
            return

        import ctypes
        import string

        GENERIC_WRITE = 0x40000000
        FILE_SHARE_READ = 0x00000001
        FILE_SHARE_WRITE = 0x00000002
        OPEN_EXISTING = 3
        INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

        flushed = 0
        for letter in string.ascii_uppercase:
            path = f"\\\\.\\{letter}:"
            # Check drive exists
            if ctypes.windll.kernel32.GetDriveTypeW(f"{letter}:\\") != 3:  # DRIVE_FIXED = 3
                continue
            h = ctypes.windll.kernel32.CreateFileW(
                path, GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                None, OPEN_EXISTING, 0, None
            )
            if h == INVALID_HANDLE_VALUE:
                self.logger.debug("Could not open volume %s: (access denied, skipping)", path)
                continue
            ok = ctypes.windll.kernel32.FlushFileBuffers(h)
            ctypes.windll.kernel32.CloseHandle(h)
            if ok:
                self.logger.info("✓ Flushed volume %s:", letter)
                flushed += 1
            else:
                self.logger.warning("FlushFileBuffers failed for %s:", letter)

        if flushed == 0:
            self.logger.warning("No volumes flushed (admin privileges may be required)")
        else:
            self.logger.info("✓ Flushed %d volume(s)", flushed)

    def health_check(self) -> None:
        """Verify system is ready for hot restart.

        Checks:
        - Administrator privileges
        - System drive has at least 100 MB free
        - bcdedit is accessible
        """
        self.logger.info("Running pre-restart health check...")
        issues = []

        # 1. Admin privileges
        if not self._check_admin_privileges():
            issues.append("Not running as administrator")

        # 2. Disk space on system drive
        import ctypes
        sys_drive = os.environ.get("SystemDrive", "C:") + "\\"
        free_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            sys_drive, None, None, ctypes.byref(free_bytes)
        )
        free_mb = free_bytes.value // (1024 * 1024)
        if free_mb < 100:
            issues.append(f"Low disk space on {sys_drive}: {free_mb} MB free (need 100 MB)")
        else:
            self.logger.info("✓ Disk space OK: %d MB free on %s", free_mb, sys_drive)

        # 3. BCD access — prefer bcdedit; fall back to direct BCD file
        if not self.dry_run:
            bcdedit_ok = False
            try:
                subprocess.run(
                    ["bcdedit", "/enum", "{bootmgr}"],
                    capture_output=True, check=True,
                    encoding="mbcs", errors="replace"
                )
                self.logger.info("✓ bcdedit accessible")
                bcdedit_ok = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

            if not bcdedit_ok:
                if self._bcd_file_available():
                    self.logger.info(
                        "✓ bcdedit not available; C:\\Boot\\BCD present (file method will be used)"
                    )
                else:
                    issues.append("bcdedit not accessible and C:\\Boot\\BCD not found")

        # 4. Boot entry availability
        if not self.dry_run:
            effective_ids = self._resolve_winpe_entry_ids()
            if not effective_ids:
                issues.append(
                    "No bootable WinPE or WinRE entry found — "
                    "run bcd_add_winpe.py or ensure WinRE is enabled (reagentc /info)"
                )
            elif effective_ids[0].lower() == self._DKTM_WINPE_GUID.lower():
                self.logger.info("✓ Boot entry: DKTM clean WinPE (%s)", effective_ids[0])
            else:
                self.logger.info(
                    "✓ Boot entry: WinRE fallback (%s) — clean WinPE not found",
                    effective_ids[0]
                )

        if issues:
            for issue in issues:
                self.logger.error("✗ %s", issue)
            raise RuntimeError("Health check failed: " + "; ".join(issues))

        self.logger.info("✓ Health check passed")
