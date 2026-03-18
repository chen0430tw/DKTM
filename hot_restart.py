#!/usr/bin/env python3
"""
DKTM Hot Restart - One-Click Kernel Reset
==========================================

Press one button, automatically:
1. Switches to WinPE
2. Resets Windows kernel
3. Switches back to Windows

Usage:
    python hot_restart.py              # Execute hot restart
    python hot_restart.py --force      # Skip confirmation prompt
    python hot_restart.py --dry-run    # Simulate without executing
"""

import sys
import os
import time
import logging
import argparse
from pathlib import Path

# Add DKTM package to path
sys.path.insert(0, str(Path(__file__).parent))

from dktm import config as dktm_config
from dktm.executor import Executor


class HotRestartOrchestrator:
    """One-click hot restart orchestrator."""

    def __init__(self, config_path=None, force=False, dry_run=False):
        self.force = force
        self.dry_run = dry_run
        self.logger = logging.getLogger("dktm.hot_restart")

        self.config = dktm_config.load_config(config_path)

        if dry_run:
            self.config.setdefault("executor", {})["mode"] = "dry-run"

        self.executor = Executor(self.config.get("executor", {}))

    def prepare_transition(self):
        self.logger.info("\n🔧 Preparing system for transition...")

        self.logger.info("   [1/3] Quiescing services...")
        self.executor.run_command("freeze_services")

        self.logger.info("   [2/3] Flushing I/O buffers...")
        self.executor.run_command("flush_io")
        self.executor.run_command("flush_buffers")

        self.logger.info("   [3/3] Final health check...")
        self.executor.run_command("health_check")

        self.logger.info("✅ System prepared for transition")
        return True

    def execute_hot_restart(self):
        self.logger.info("\n🚀 Executing Hot Restart Sequence...")

        self.logger.info("   [1/4] Handing over control to WinPE...")
        self.executor.run_command("handover_control")

        self.logger.info("   [2/4] Setting one-time boot sequence...")
        self.executor.run_command("commit_transition")

        self.logger.info("   [3/4] Final synchronization...")
        time.sleep(1)

        self.logger.info("   [4/4] Initiating reboot to WinPE...")
        self.logger.info("\n" + "=" * 60)
        self.logger.info("  🔄 System will now reboot into WinPE")
        self.logger.info("  ⚙️  WinPE will perform kernel reset")
        self.logger.info("  🔙 System will automatically return to Windows")
        self.logger.info("=" * 60)

        if not self.dry_run:
            self.logger.info("\nRebooting in 5 seconds...")
            time.sleep(5)

        self.executor.run_command("reboot")
        return True

    def run(self):
        self.logger.info("╔" + "=" * 58 + "╗")
        self.logger.info("║  DKTM Hot Restart - One-Click Kernel Reset            ║")
        self.logger.info("╚" + "=" * 58 + "╝")

        if self.dry_run:
            self.logger.warning("\n⚠️  DRY-RUN MODE - No actual changes will be made\n")

        try:
            if not self.dry_run and not self.force:
                response = input("\n⚠️  Ready to perform hot restart. Continue? [y/N]: ")
                if response.lower() != 'y':
                    self.logger.info("Aborted by user")
                    return 1

            if not self.prepare_transition():
                self.logger.error("❌ Preparation failed")
                return 1

            if not self.execute_hot_restart():
                self.logger.error("❌ Hot restart failed")
                return 1

            if self.dry_run:
                self.logger.info("\n✅ Dry-run completed successfully")
                self.logger.info("   In real mode, system would reboot now")
            else:
                self.logger.info("\n✅ Hot restart sequence initiated")
                self.logger.info("   System rebooting...")

            return 0

        except KeyboardInterrupt:
            self.logger.warning("\n\n⚠️  Interrupted by user")
            return 1
        except Exception as exc:
            self.logger.error(f"\n❌ Unexpected error: {exc}")
            import traceback
            traceback.print_exc()
            return 1


def main():
    parser = argparse.ArgumentParser(
        description="DKTM Hot Restart - One-Click Kernel Reset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hot_restart.py               # Execute hot restart
  python hot_restart.py --dry-run     # Test without rebooting
  python hot_restart.py --force       # Skip confirmation prompt
  python hot_restart.py --config custom.yaml
        """
    )

    parser.add_argument("--config", type=str, default=None, help="Path to configuration file")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without executing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s"
    )

    if sys.platform != "win32" and not args.dry_run:
        print("ERROR: Hot restart requires Windows")
        print("       Use --dry-run to test on other platforms")
        return 1

    orchestrator = HotRestartOrchestrator(
        config_path=args.config,
        force=args.force,
        dry_run=args.dry_run,
    )

    return orchestrator.run()


if __name__ == "__main__":
    sys.exit(main())
