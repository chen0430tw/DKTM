#!/usr/bin/env python3
"""
DKTM One-Click Installer
=========================

Automatically sets up DKTM hot restart system.

Usage:
    python install.py              # Full automatic installation
    python install.py --skip-pe    # Skip WinPE build (use existing)

This script:
1. Builds WinPE with DKTM recovery scripts
2. Creates BCD entry
3. Configures DKTM
4. Validates setup

After installation, use:
    python hot_restart.py          # One-click hot restart!
"""

import sys
import os
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Optional


class DKTMInstaller:
    """One-click DKTM installer."""

    def __init__(self, skip_pe: bool = False, transition_method: str = "auto"):
        """Initialize installer.

        Parameters
        ----------
        skip_pe : bool
            If True, skip WinPE build step.
        transition_method : str
            Transition method: "bcd", "winre", or "auto".
        """
        self.skip_pe = skip_pe
        self.transition_method = transition_method
        self.logger = logging.getLogger("dktm.install")
        self.project_root = Path(__file__).parent

    def check_prerequisites(self) -> bool:
        """Check system prerequisites.

        Returns
        -------
        bool
            True if all prerequisites met.
        """
        self.logger.info("Checking prerequisites...")

        # Check platform
        if sys.platform != "win32":
            self.logger.error("✗ This installer requires Windows")
            return False

        # Check admin privileges
        try:
            import ctypes
            if ctypes.windll.shell32.IsUserAnAdmin() == 0:
                self.logger.error("✗ Administrator privileges required")
                self.logger.error("  Please run as Administrator")
                return False
        except Exception:
            self.logger.warning("⚠️  Could not check admin privileges")

        # Check Python version
        if sys.version_info < (3, 7):
            self.logger.error("✗ Python 3.7 or higher required")
            return False

        # Check required Python packages
        # Note: package name for import may differ from pip install name
        required_packages = {
            "numpy": "numpy",
            "yaml": "pyyaml"  # Import as 'yaml', install as 'pyyaml'
        }
        missing = []
        for import_name, pip_name in required_packages.items():
            try:
                __import__(import_name)
            except ImportError:
                missing.append(pip_name)

        if missing:
            self.logger.error(f"✗ Missing Python packages: {', '.join(missing)}")
            self.logger.info("  Install with: pip install " + " ".join(missing))
            return False

        self.logger.info("✓ All prerequisites met")
        return True

    def create_directories(self) -> bool:
        """Create required directories.

        Returns
        -------
        bool
            True if successful.
        """
        self.logger.info("Creating directories...")

        dirs = [
            Path(r"C:\DKTM"),
            Path(r"C:\DKTM\logs"),
        ]

        for dir_path in dirs:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"  Created: {dir_path}")
            except Exception as exc:
                self.logger.error(f"✗ Could not create {dir_path}: {exc}")
                return False

        self.logger.info("✓ Directories created")
        return True

    def build_winpe(self) -> bool:
        """Build WinPE image.

        Returns
        -------
        bool
            True if successful.
        """
        if self.skip_pe:
            self.logger.info("Skipping WinPE build (--skip-pe)")
            # Verify existing WinPE
            if not Path(r"C:\WinPE\sources\boot.wim").exists():
                self.logger.error("✗ No existing WinPE found at C:\\WinPE")
                self.logger.error("  Remove --skip-pe to build automatically")
                return False
            return True

        self.logger.info("Building WinPE...")

        build_script = self.project_root / "tools" / "build_pe.py"
        if not build_script.exists():
            self.logger.error(f"✗ Build script not found: {build_script}")
            return False

        try:
            result = subprocess.run(
                [sys.executable, str(build_script), "--deploy"],
                check=True,
                capture_output=False  # Show output in real-time
            )
            self.logger.info("✓ WinPE built and deployed")
            return True
        except subprocess.CalledProcessError as exc:
            self.logger.error(f"✗ WinPE build failed: {exc}")
            return False

    def setup_bcd(self) -> bool:
        """Setup BCD entry.

        Returns
        -------
        bool
            True if successful.
        """
        self.logger.info("Setting up BCD entry...")

        setup_script = self.project_root / "tools" / "setup_bcd.py"
        if not setup_script.exists():
            self.logger.error(f"✗ Setup script not found: {setup_script}")
            return False

        try:
            result = subprocess.run(
                [sys.executable, str(setup_script), "--save-config"],
                check=True,
                capture_output=False
            )
            self.logger.info("✓ BCD configured")
            return True
        except subprocess.CalledProcessError as exc:
            self.logger.error(f"✗ BCD setup failed: {exc}")
            return False

    def ensure_config(self, transition_method: str) -> bool:
        """Ensure a configuration file exists with transition settings."""
        config_file = self.project_root / "dktm_config.yaml"
        try:
            import yaml
        except ImportError:
            self.logger.error("✗ PyYAML is required to write configuration")
            return False

        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}

        executor = config.setdefault("executor", {})
        executor.setdefault("winpe_entry_ids", [])
        executor.setdefault("marker_path", r"C:\DKTM\dktm_transition.marker")
        executor.setdefault("mode", "dry-run")
        executor["transition_method"] = transition_method
        executor.setdefault("fallback_method", "winre")

        try:
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            self.logger.info("✓ Configuration updated")
            return True
        except Exception as exc:
            self.logger.error(f"✗ Could not save configuration: {exc}")
            return False

    def verify_installation(self) -> bool:
        """Verify installation.

        Returns
        -------
        bool
            True if installation is valid.
        """
        self.logger.info("Verifying installation...")

        # Check WinPE files
        required_files = [
            Path(r"C:\WinPE\sources\boot.wim"),
            Path(r"C:\WinPE\boot\boot.sdi"),
        ]

        for file_path in required_files:
            if not file_path.exists():
                self.logger.error(f"✗ Missing file: {file_path}")
                return False

        # Check configuration
        config_file = self.project_root / "dktm_config.yaml"
        if not config_file.exists():
            self.logger.warning("⚠️  Configuration file not found")
            self.logger.warning("  BCD setup may have failed")
            return False

        # Check BCD entry
        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            executor_config = config.get("executor", {})
            transition_method = executor_config.get("transition_method", "auto")
            winpe_guids = executor_config.get("winpe_entry_ids", [])
            if transition_method != "winre" and not winpe_guids:
                self.logger.error("✗ No WinPE GUID in configuration")
                return False

            if winpe_guids:
                self.logger.info(f"  WinPE GUID: {winpe_guids[0]}")
            self.logger.info(f"  Transition method: {transition_method}")

        except Exception as exc:
            self.logger.error(f"✗ Could not verify configuration: {exc}")
            return False

        self.logger.info("✓ Installation verified")
        return True

    def install(self) -> int:
        """Execute full installation.

        Returns
        -------
        int
            Exit code (0 = success, 1 = failure).
        """
        self.logger.info("╔" + "=" * 58 + "╗")
        self.logger.info("║  DKTM One-Click Installer                             ║")
        self.logger.info("╚" + "=" * 58 + "╝\n")

        steps = [
            ("Checking prerequisites", self.check_prerequisites),
            ("Creating directories", self.create_directories),
            ("Building WinPE", self.build_winpe),
            ("Setting up transition", self.setup_bcd),
            ("Verifying installation", self.verify_installation),
        ]

        for step_name, step_func in steps:
            self.logger.info(f"\n>>> {step_name}...")
            if step_name == "Setting up transition":
                if self.transition_method == "winre":
                    self.logger.info("Skipping BCD setup (transition_method=winre)")
                    if not self.ensure_config("winre"):
                        self.logger.error("\n✗ Installation failed at: Setting up transition")
                        return 1
                    continue
                if self.transition_method == "auto":
                    if step_func():
                        continue
                    self.logger.warning("BCD setup failed; falling back to WinRE")
                    if not self.ensure_config("winre"):
                        self.logger.error("\n✗ Installation failed at: Setting up transition")
                        return 1
                    continue
            if not step_func():
                self.logger.error(f"\n✗ Installation failed at: {step_name}")
                return 1

        self.logger.info("\n" + "=" * 60)
        self.logger.info("  ✅ DKTM Installation Complete!")
        self.logger.info("=" * 60)
        self.logger.info("\nYou can now use:")
        self.logger.info("  python hot_restart.py --dry-run    # Test the system")
        self.logger.info("  python hot_restart.py              # One-click hot restart!")
        self.logger.info("\nConfiguration file: dktm_config.yaml")
        self.logger.info("Logs directory: C:\\DKTM\\logs")
        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DKTM One-Click Installer",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--skip-pe",
        action="store_true",
        help="Skip WinPE build (use existing at C:\\WinPE)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--transition-method",
        choices=["auto", "bcd", "winre"],
        default="auto",
        help="Transition method: auto (default), bcd, or winre"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s"
    )

    # Run installer
    installer = DKTMInstaller(
        skip_pe=args.skip_pe,
        transition_method=args.transition_method
    )
    return installer.install()


if __name__ == "__main__":
    sys.exit(main())
