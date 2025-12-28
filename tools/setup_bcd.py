#!/usr/bin/env python3
"""
DKTM BCD Auto-Configurator
===========================

Automatically creates and configures BCD entry for DKTM WinPE.

Usage:
    python tools/setup_bcd.py --pe-path C:\\WinPE --save-config

Features:
- Creates BCD entry for WinPE
- Configures ramdisk options
- Saves GUID to configuration
- Validates setup
"""

import os
import sys
import subprocess
import re
import logging
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import argparse


class BCDConfigurator:
    """Automated BCD configurator for DKTM WinPE."""

    def __init__(self, pe_path: str = r"C:\WinPE"):
        """Initialize BCD configurator.

        Parameters
        ----------
        pe_path : str
            Path to WinPE installation.
        """
        self.pe_path = Path(pe_path)
        self.winpe_guid: Optional[str] = None
        self.logger = logging.getLogger("dktm.setup_bcd")

    def check_admin(self) -> bool:
        """Check administrator privileges.

        Returns
        -------
        bool
            True if running as administrator.
        """
        try:
            import ctypes
            is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
            if not is_admin:
                self.logger.error("✗ Administrator privileges required!")
                self.logger.error("  Please run this script as Administrator")
            return is_admin
        except Exception:
            return False

    def verify_pe_files(self) -> bool:
        """Verify WinPE files exist.

        Returns
        -------
        bool
            True if all required files exist.
        """
        self.logger.info(f"Verifying WinPE files at {self.pe_path}...")

        required_files = [
            self.pe_path / "sources" / "boot.wim",
            self.pe_path / "boot" / "boot.sdi",
        ]

        for file_path in required_files:
            if not file_path.exists():
                self.logger.error(f"✗ Missing required file: {file_path}")
                return False

        self.logger.info("✓ All required files found")
        return True

    def create_bcd_entry(self) -> bool:
        """Create BCD entry for DKTM WinPE.

        Returns
        -------
        bool
            True if successful.
        """
        self.logger.info("Creating BCD entry for DKTM WinPE...")

        try:
            # Create new OS loader entry
            result = subprocess.run(
                ["bcdedit", "/create", "/d", "DKTM WinPE Hot Restart", "/application", "osloader"],
                capture_output=True,
                text=True,
                check=True
            )

            # Extract GUID from output
            # Output format: "The entry {xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx} was successfully created."
            match = re.search(r'\{([0-9a-f-]+)\}', result.stdout, re.IGNORECASE)
            if not match:
                self.logger.error("✗ Could not extract GUID from bcdedit output")
                return False

            self.winpe_guid = "{" + match.group(1) + "}"
            self.logger.info(f"✓ Created BCD entry: {self.winpe_guid}")
            return True

        except subprocess.CalledProcessError as exc:
            self.logger.error(f"✗ BCD entry creation failed: {exc.stderr}")
            return False

    def configure_ramdisk_options(self) -> bool:
        """Configure ramdisk options for WinPE.

        Returns
        -------
        bool
            True if successful.
        """
        self.logger.info("Configuring ramdisk options...")

        # Get system drive letter
        system_drive = Path(self.pe_path).drive

        commands = [
            ["bcdedit", "/set", "{ramdiskoptions}", "ramdisksdidevice", f"partition={system_drive}"],
            ["bcdedit", "/set", "{ramdiskoptions}", "ramdisksdipath", "\\WinPE\\boot\\boot.sdi"],
        ]

        for cmd in commands:
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as exc:
                self.logger.error(f"✗ Command failed: {' '.join(cmd)}")
                self.logger.error(f"  Error: {exc.stderr}")
                return False

        self.logger.info("✓ Ramdisk options configured")
        return True

    def configure_winpe_entry(self) -> bool:
        """Configure WinPE BCD entry.

        Returns
        -------
        bool
            True if successful.
        """
        if not self.winpe_guid:
            self.logger.error("✗ No GUID available for configuration")
            return False

        self.logger.info(f"Configuring WinPE entry {self.winpe_guid}...")

        # Get system drive and construct paths
        system_drive = Path(self.pe_path).drive
        wim_path = f"[{system_drive}]\\WinPE\\sources\\boot.wim"

        commands = [
            # Set device to ramdisk
            ["bcdedit", "/set", self.winpe_guid, "device", f"ramdisk={wim_path},{{ramdiskoptions}}"],
            ["bcdedit", "/set", self.winpe_guid, "osdevice", f"ramdisk={wim_path},{{ramdiskoptions}}"],

            # Set boot path (UEFI)
            ["bcdedit", "/set", self.winpe_guid, "path", "\\Windows\\System32\\boot\\winload.efi"],

            # Set system root
            ["bcdedit", "/set", self.winpe_guid, "systemroot", "\\Windows"],

            # Enable WinPE mode
            ["bcdedit", "/set", self.winpe_guid, "winpe", "yes"],

            # Enable HAL detection
            ["bcdedit", "/set", self.winpe_guid, "detecthal", "yes"],
        ]

        for cmd in commands:
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as exc:
                # Try alternative boot loader path for BIOS
                if "path" in cmd:
                    self.logger.warning("UEFI path failed, trying BIOS path...")
                    cmd_bios = cmd.copy()
                    cmd_bios[-1] = "\\Windows\\System32\\boot\\winload.exe"
                    try:
                        subprocess.run(cmd_bios, check=True, capture_output=True)
                        continue
                    except subprocess.CalledProcessError:
                        pass

                self.logger.error(f"✗ Command failed: {' '.join(cmd)}")
                self.logger.error(f"  Error: {exc.stderr.decode('utf-8', errors='ignore')}")
                return False

        self.logger.info("✓ WinPE entry configured")
        return True

    def add_to_display_order(self, add_to_menu: bool = False) -> bool:
        """Add WinPE entry to boot menu.

        Parameters
        ----------
        add_to_menu : bool
            If True, add to permanent display order.

        Returns
        -------
        bool
            True if successful.
        """
        if not add_to_menu:
            self.logger.info("Skipping display order (one-time boot only)")
            return True

        if not self.winpe_guid:
            return False

        self.logger.info("Adding to boot menu display order...")

        try:
            subprocess.run(
                ["bcdedit", "/displayorder", self.winpe_guid, "/addlast"],
                check=True,
                capture_output=True
            )
            self.logger.info("✓ Added to boot menu")
            return True
        except subprocess.CalledProcessError as exc:
            self.logger.warning(f"Could not add to display order: {exc}")
            return True  # Non-critical

    def verify_configuration(self) -> bool:
        """Verify BCD configuration.

        Returns
        -------
        bool
            True if configuration is valid.
        """
        if not self.winpe_guid:
            return False

        self.logger.info("Verifying BCD configuration...")

        try:
            result = subprocess.run(
                ["bcdedit", "/enum", self.winpe_guid],
                capture_output=True,
                text=True,
                check=True,
                encoding="mbcs",
                errors="replace"
            )

            # Check for required fields
            required_fields = ["device", "osdevice", "winpe"]
            for field in required_fields:
                if field not in result.stdout.lower():
                    self.logger.error(f"✗ Missing field: {field}")
                    return False

            self.logger.info("✓ BCD configuration verified")
            return True

        except subprocess.CalledProcessError:
            self.logger.error("✗ Could not verify configuration")
            return False

    def save_to_config(self, config_path: str = "dktm_config.yaml") -> bool:
        """Save WinPE GUID to DKTM configuration file.

        Parameters
        ----------
        config_path : str
            Path to configuration file.

        Returns
        -------
        bool
            True if successful.
        """
        if not self.winpe_guid:
            return False

        self.logger.info(f"Saving configuration to {config_path}...")

        config_file = Path(config_path)

        # Load existing config or create new
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}

        # Update executor configuration
        if 'executor' not in config:
            config['executor'] = {}

        config['executor']['winpe_entry_ids'] = [self.winpe_guid]
        config['executor']['marker_path'] = r"C:\DKTM\dktm_transition.marker"
        config['executor']['mode'] = "dry-run"  # Safe default
        config['executor']['transition_method'] = "bcd"
        config['executor']['fallback_method'] = "winre"

        # Write configuration
        try:
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            self.logger.info(f"✓ Configuration saved to {config_path}")
            return True
        except Exception as exc:
            self.logger.error(f"✗ Could not save configuration: {exc}")
            return False

    def setup(self, save_config: bool = True, add_to_menu: bool = False) -> bool:
        """Execute complete BCD setup.

        Parameters
        ----------
        save_config : bool
            If True, save GUID to configuration file.
        add_to_menu : bool
            If True, add to permanent boot menu.

        Returns
        -------
        bool
            True if successful.
        """
        steps = [
            ("Checking administrator privileges", self.check_admin),
            ("Verifying WinPE files", self.verify_pe_files),
            ("Creating BCD entry", self.create_bcd_entry),
            ("Configuring ramdisk options", self.configure_ramdisk_options),
            ("Configuring WinPE entry", self.configure_winpe_entry),
            ("Adding to boot menu", lambda: self.add_to_display_order(add_to_menu)),
            ("Verifying configuration", self.verify_configuration),
        ]

        for step_name, step_func in steps:
            self.logger.info(f"\n>>> {step_name}...")
            if not step_func():
                self.logger.error(f"✗ Setup failed at: {step_name}")
                return False

        if save_config:
            self.logger.info("\n>>> Saving configuration...")
            if not self.save_to_config():
                self.logger.warning("Could not save configuration (non-critical)")

        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="DKTM BCD Auto-Configurator")
    parser.add_argument(
        "--pe-path",
        default=r"C:\WinPE",
        help="Path to WinPE installation (default: C:\\WinPE)"
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save GUID to dktm_config.yaml"
    )
    parser.add_argument(
        "--add-to-menu",
        action="store_true",
        help="Add to permanent boot menu"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s"
    )

    # Check platform
    if sys.platform != "win32":
        print("ERROR: This tool must be run on Windows")
        return 1

    # Setup BCD
    configurator = BCDConfigurator(args.pe_path)
    success = configurator.setup(
        save_config=args.save_config,
        add_to_menu=args.add_to_menu
    )

    if success:
        print("\n" + "=" * 60)
        print("  DKTM BCD Setup Successful!")
        print("=" * 60)
        print(f"\n✓ WinPE BCD Entry: {configurator.winpe_guid}")
        if args.save_config:
            print("✓ Configuration saved to dktm_config.yaml")
        print("\n✓ Ready for hot restart!")
        return 0
    else:
        print("\n✗ Setup failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
