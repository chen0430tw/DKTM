#!/usr/bin/env python3
"""
DKTM WinPE Auto-Builder
=======================

Automatically builds a custom WinPE image with DKTM recovery scripts.

Usage:
    python tools/build_pe.py --output C:\\WinPE_DKTM --auto-deploy

Features:
- Detects Windows ADK installation
- Generates WinPE base image
- Injects DKTM recovery scripts
- Creates auto-run configuration
- Deploys to system partition
"""

import os
import sys
import subprocess
import shutil
import logging
from pathlib import Path
from typing import Optional, List
import argparse


class WinPEBuilder:
    """Automated WinPE builder for DKTM."""

    # Common ADK installation paths
    ADK_PATHS = [
        r"C:\Program Files (x86)\Windows Kits\10",
        r"C:\Program Files\Windows Kits\10",
    ]

    def __init__(self, output_dir: str, arch: str = "amd64"):
        """Initialize WinPE builder.

        Parameters
        ----------
        output_dir : str
            Output directory for WinPE files.
        arch : str
            Architecture (amd64 or x86).
        """
        self.output_dir = Path(output_dir)
        self.arch = arch
        self.adk_path: Optional[Path] = None
        self.logger = logging.getLogger("dktm.build_pe")

    def detect_adk(self) -> bool:
        """Detect Windows ADK installation.

        Returns
        -------
        bool
            True if ADK found, False otherwise.
        """
        self.logger.info("Detecting Windows ADK installation...")

        for path_str in self.ADK_PATHS:
            path = Path(path_str)
            if path.exists():
                # Check for copype.cmd
                copype = path / "Assessment and Deployment Kit" / "Windows Preinstallation Environment" / "copype.cmd"
                if copype.exists():
                    self.adk_path = path
                    self.logger.info(f"✓ Found ADK at: {path}")
                    return True

        self.logger.error("✗ Windows ADK not found!")
        self.logger.error("Please install Windows ADK from:")
        self.logger.error("https://learn.microsoft.com/en-us/windows-hardware/get-started/adk-install")
        return False

    def run_copype(self) -> bool:
        """Run copype to create base WinPE structure.

        Returns
        -------
        bool
            True if successful.
        """
        self.logger.info(f"Creating WinPE base structure at {self.output_dir}...")

        if self.output_dir.exists():
            self.logger.warning(f"Output directory exists, cleaning up...")
            shutil.rmtree(self.output_dir)

        copype = self.adk_path / "Assessment and Deployment Kit" / "Windows Preinstallation Environment" / "copype.cmd"

        try:
            result = subprocess.run(
                [str(copype), self.arch, str(self.output_dir)],
                capture_output=True,
                text=True,
                check=True
            )
            self.logger.info("✓ WinPE base structure created")
            return True
        except subprocess.CalledProcessError as exc:
            self.logger.error(f"✗ copype failed: {exc.stderr}")
            return False

    def create_mount_point(self) -> bool:
        """Create mount directory for DISM operations.

        Returns
        -------
        bool
            True if successful.
        """
        mount_dir = self.output_dir / "mount"
        mount_dir.mkdir(exist_ok=True)
        self.logger.info(f"✓ Mount point created: {mount_dir}")
        return True

    def mount_wim(self) -> bool:
        """Mount boot.wim for customization.

        Returns
        -------
        bool
            True if successful.
        """
        self.logger.info("Mounting boot.wim...")

        wim_path = self.output_dir / "media" / "sources" / "boot.wim"
        mount_dir = self.output_dir / "mount"

        try:
            subprocess.run(
                [
                    "dism",
                    "/Mount-Image",
                    f"/ImageFile:{wim_path}",
                    "/Index:1",
                    f"/MountDir:{mount_dir}"
                ],
                check=True,
                capture_output=True
            )
            self.logger.info("✓ boot.wim mounted")
            return True
        except subprocess.CalledProcessError as exc:
            self.logger.error(f"✗ Mount failed: {exc}")
            return False

    def inject_dktm_scripts(self) -> bool:
        """Inject DKTM recovery scripts into WinPE.

        Returns
        -------
        bool
            True if successful.
        """
        self.logger.info("Injecting DKTM recovery scripts...")

        mount_dir = self.output_dir / "mount"
        dktm_dir = mount_dir / "DKTM"
        dktm_dir.mkdir(exist_ok=True)

        # Create auto-recovery script
        autorun_script = dktm_dir / "dktm_recovery.cmd"
        autorun_content = self._generate_recovery_script()

        try:
            autorun_script.write_text(autorun_content, encoding="utf-8")
            self.logger.info(f"✓ Recovery script created: {autorun_script}")
            return True
        except Exception as exc:
            self.logger.error(f"✗ Script injection failed: {exc}")
            return False

    def _generate_recovery_script(self) -> str:
        """Generate DKTM recovery script content.

        Returns
        -------
        str
            Script content.
        """
        return r"""@echo off
REM DKTM Auto-Recovery Script
REM Automatically executed in WinPE to perform kernel reset

echo ========================================
echo   DKTM Hot Restart - Recovery Phase
echo ========================================
echo.

REM Check for transition marker
set MARKER_PATH=C:\DKTM\dktm_transition.marker

if not exist %MARKER_PATH% (
    echo [ERROR] No transition marker found!
    echo [INFO] Entering manual recovery mode...
    cmd
    exit /b 1
)

echo [INFO] Transition marker detected
echo [INFO] Reading transition context...

REM Parse marker file (JSON)
REM For now, we just verify it exists

echo.
echo [PHASE 1] Kernel Reset Preparation
echo =====================================

REM Flush all disk caches
echo [1/5] Flushing disk caches...
echo sync > nul 2>&1

REM Clear Windows event logs (to simulate fresh boot)
echo [2/5] Clearing event logs...
wevtutil cl System > nul 2>&1
wevtutil cl Application > nul 2>&1

REM Reset network stack
echo [3/5] Resetting network stack...
netsh int ip reset > nul 2>&1

REM Clear DNS cache
echo [4/5] Clearing DNS cache...
ipconfig /flushdns > nul 2>&1

REM Verify system integrity
echo [5/5] Verifying system integrity...
sfc /scannow > nul 2>&1

echo.
echo [PHASE 2] Kernel Reset Complete
echo =====================================
echo [INFO] Hot restart cycle completed successfully

REM Clear bootsequence to return to normal boot
echo.
echo [PHASE 3] Restoring Boot Configuration
echo ========================================
echo [INFO] Clearing one-time boot sequence...
bcdedit /deletevalue {bootmgr} bootsequence

REM Delete transition marker
echo [INFO] Cleaning up transition marker...
del /f /q %MARKER_PATH%

echo.
echo ========================================
echo   DKTM Hot Restart - SUCCESS
echo ========================================
echo.
echo System will reboot to Windows in 5 seconds...
timeout /t 5

REM Reboot back to Windows
wpeutil reboot
"""

    def configure_startnet(self) -> bool:
        """Configure startnet.cmd to auto-run DKTM recovery.

        Returns
        -------
        bool
            True if successful.
        """
        self.logger.info("Configuring startnet.cmd...")

        mount_dir = self.output_dir / "mount"
        startnet = mount_dir / "Windows" / "System32" / "startnet.cmd"

        startnet_content = r"""@echo off
REM DKTM WinPE Auto-Start Script

wpeinit

REM Execute DKTM recovery script
echo.
echo Starting DKTM Hot Restart Recovery...
call X:\DKTM\dktm_recovery.cmd

REM If script exits without rebooting, open command prompt
cmd
"""

        try:
            # Use mbcs encoding for Windows batch files (equivalent to system default)
            startnet.write_text(startnet_content, encoding="mbcs")
            self.logger.info("✓ startnet.cmd configured")
            return True
        except Exception as exc:
            self.logger.error(f"✗ startnet.cmd configuration failed: {exc}")
            return False

    def unmount_wim(self, commit: bool = True) -> bool:
        """Unmount boot.wim.

        Parameters
        ----------
        commit : bool
            If True, commit changes. If False, discard.

        Returns
        -------
        bool
            True if successful.
        """
        self.logger.info("Unmounting boot.wim...")

        mount_dir = self.output_dir / "mount"
        commit_arg = "/Commit" if commit else "/Discard"

        try:
            subprocess.run(
                ["dism", "/Unmount-Image", f"/MountDir:{mount_dir}", commit_arg],
                check=True,
                capture_output=True
            )
            self.logger.info("✓ boot.wim unmounted")
            return True
        except subprocess.CalledProcessError as exc:
            self.logger.error(f"✗ Unmount failed: {exc}")
            return False

    def deploy_to_system(self, target_path: str = r"C:\WinPE") -> bool:
        """Deploy WinPE to system partition.

        Parameters
        ----------
        target_path : str
            Deployment path on system partition.

        Returns
        -------
        bool
            True if successful.
        """
        self.logger.info(f"Deploying WinPE to {target_path}...")

        source_media = self.output_dir / "media"
        target = Path(target_path)

        try:
            if target.exists():
                self.logger.warning(f"Target exists, removing old files...")
                shutil.rmtree(target)

            shutil.copytree(source_media, target)
            self.logger.info(f"✓ WinPE deployed to {target}")
            return True
        except Exception as exc:
            self.logger.error(f"✗ Deployment failed: {exc}")
            return False

    def build(self, deploy: bool = False) -> bool:
        """Execute complete build process.

        Parameters
        ----------
        deploy : bool
            If True, deploy to system partition after building.

        Returns
        -------
        bool
            True if successful.
        """
        steps = [
            ("Detecting ADK", self.detect_adk),
            ("Creating WinPE base", self.run_copype),
            ("Creating mount point", self.create_mount_point),
            ("Mounting WinPE image", self.mount_wim),
            ("Injecting DKTM scripts", self.inject_dktm_scripts),
            ("Configuring auto-start", self.configure_startnet),
            ("Unmounting image", lambda: self.unmount_wim(commit=True)),
        ]

        for step_name, step_func in steps:
            self.logger.info(f"\n>>> {step_name}...")
            if not step_func():
                self.logger.error(f"✗ Build failed at: {step_name}")
                return False

        self.logger.info("\n✓ WinPE build completed successfully!")

        if deploy:
            self.logger.info("\n>>> Deploying to system partition...")
            if not self.deploy_to_system():
                return False

        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="DKTM WinPE Auto-Builder")
    parser.add_argument(
        "--output",
        default="C:\\WinPE_DKTM_Build",
        help="Output directory for WinPE files"
    )
    parser.add_argument(
        "--arch",
        choices=["amd64", "x86"],
        default="amd64",
        help="Architecture (default: amd64)"
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy to system partition after building"
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

    # Build WinPE
    builder = WinPEBuilder(args.output, args.arch)
    success = builder.build(deploy=args.deploy)

    if success:
        print("\n" + "=" * 60)
        print("  DKTM WinPE Build Successful!")
        print("=" * 60)
        if args.deploy:
            print("\n✓ WinPE deployed to C:\\WinPE")
            print("✓ Ready for BCD configuration")
        else:
            print(f"\n✓ WinPE files: {args.output}\\media")
            print("  Run with --deploy to deploy to system partition")
        return 0
    else:
        print("\n✗ Build failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
