"""
Executor for DKTM plans
-----------------------

This module defines an ``Executor`` class responsible for applying
transition plans produced by the DKTM planner.  Plans are executed
phase by phase, command by command.  In dry‑run mode, the executor
prints each command rather than performing any operation.  In
real‑run mode it dispatches commands to internal handlers or
delegates to platform‑specific operations via ``platform_ops``.

The executor is designed to be extensible: new command names can
either be implemented as methods prefixed with an underscore (e.g.
``_freeze_services``) or forwarded to the ``platform_ops`` module if
they correspond to system‑level actions such as committing a boot
transition or rebooting the machine.
"""

from __future__ import annotations

from typing import Any, Dict, List

import logging

# Import platform_ops using absolute import when running within the dktm package. When
# executed as a script (e.g., via ``python dktm/dktm.py``), the ``dktm`` package name
# resolves to the script module itself and does not expose package modules. In that
# scenario, fall back to importing the root‑level ``platform_ops`` wrapper.
try:
    from dktm import platform_ops  # type: ignore[attr-defined]
except Exception:
    import platform_ops  # type: ignore


class Executor:
    """Execute DKTM plans.

    Parameters
    ----------
    config : dict
        Merged configuration dictionary loaded from config.py.  The
        executor reads the following keys:

        ``mode`` (str): Either ``'dry-run'`` or ``'real-run'``.  In
        dry‑run mode, commands are logged but not executed.  In real
        mode, command handlers perform actual operations or call
        platform_ops.

        ``auto_reboot`` (bool): If true, automatically reboot after
        committing a transition.  Applicable only on Windows when
        committing transitions.

        ``winpe_entry_ids`` (list[str]): List of BCD entry GUIDs for
        WinPE environments.  The first entry will be used for the
        bootsequence when committing a transition on Windows.

        ``marker_path`` (str): Path to a marker file indicating a
        pending transition.  This is created when committing a
        transition and removed on rollback.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.mode: str = config.get("mode", "dry-run")
        self.auto_reboot: bool = config.get("auto_reboot", False)
        self.winpe_entry_ids: List[str] = list(config.get("winpe_entry_ids", []))
        self.marker_path: str = config.get("marker_path", "dktm_transition.marker")
        self.logger = logging.getLogger("dktm.executor")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_plan(self, plan: List[Dict[str, Any]]) -> None:
        """Execute a plan produced by the DKTM planner.

        The plan is a list of phases; each phase is a dictionary with a
        ``commands`` key containing a list of command strings.

        Parameters
        ----------
        plan : list of dict
            Plan as returned by ``plan.build_plan``.
        """
        for phase_idx, phase in enumerate(plan):
            commands: List[str] = phase.get("commands", [])
            self.logger.info(
                "Executing phase %d (%d commands)", phase_idx + 1, len(commands)
            )
            for cmd in commands:
                self.run_command(cmd)

    def run_command(self, command: str) -> None:
        """Execute a single command.

        The method first resolves a handler.  If the handler exists as
        a method on this class (with the name ``_{command}``) it will
        be invoked.  Otherwise, known platform‑level commands are
        delegated to platform_ops.  Unknown commands are simply
        logged.

        Parameters
        ----------
        command : str
            Name of the command to execute.
        """
        self.logger.debug("run_command: %s", command)
        if self.mode != "real-run":
            # Dry run: log and return
            self.logger.info("[dry-run] %s", command)
            return
        # Determine handler
        method_name = f"_{command}"
        if hasattr(self, method_name):
            try:
                getattr(self, method_name)()
            except Exception as exc:
                self.logger.error("Error executing command %s: %s", command, exc)
        else:
            # Delegate to platform_ops if supported
            handled = self._delegate_to_platform(command)
            if not handled:
                self.logger.warning("No handler for command: %s", command)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _delegate_to_platform(self, command: str) -> bool:
        """Try to dispatch a command to platform_ops.

        Returns true if the command was handled.
        """
        # Mapping of command names to platform operations
        if command == "commit_transition":
            platform_ops.commit_transition(
                winpe_entry_ids=self.winpe_entry_ids,
                marker_path=self.marker_path,
                auto_reboot=self.auto_reboot,
            )
            return True
        elif command == "rollback_transition":
            platform_ops.rollback_transition(
                winpe_entry_ids=self.winpe_entry_ids,
                marker_path=self.marker_path,
            )
            return True
        elif command == "handover_control":
            platform_ops.handover_control()
            return True
        elif command == "reboot":
            platform_ops.reboot()
            return True
        return False

    # ------------------------------------------------------------------
    # Example stub implementations
    # ------------------------------------------------------------------
    def _freeze_services(self) -> None:
        """Quiesce running services to prepare for maintenance."""
        # In a real implementation this would stop non‑essential services.
        self.logger.info("Freezing services (stub)")

    def _quiesce_services_tier1(self) -> None:
        """Quiesce first tier of services."""
        self.logger.info("Quiescing tier 1 services (stub)")

    def _quiesce_services_tier2(self) -> None:
        """Quiesce second tier of services."""
        self.logger.info("Quiescing tier 2 services (stub)")

    def _flush_io(self) -> None:
        """Flush pending I/O operations."""
        self.logger.info("Flushing I/O (stub)")

    def _flush_buffers(self) -> None:
        """Flush system buffers."""
        self.logger.info("Flushing buffers (stub)")

    def _quiesce_drivers_soft(self) -> None:
        """Quiesce drivers using soft quiescence."""
        self.logger.info("Quiescing drivers (soft) (stub)")

    def _quiesce_drivers_hard(self) -> None:
        """Quiesce drivers using hard reset."""
        self.logger.info("Quiescing drivers (hard) (stub)")

    def _health_check(self) -> None:
        """Perform a health check on the system."""
        self.logger.info("Performing health check (stub)")

    def _verify_integrity(self) -> None:
        """Verify system integrity."""
        self.logger.info("Verifying system integrity (stub)")

    def _enter_maintenance(self) -> None:
        """Prepare the system for maintenance mode."""
        self.logger.info("Entering maintenance mode (stub)")

    def _exit_maintenance(self) -> None:
        """Exit maintenance mode and resume normal operations."""
        self.logger.info("Exiting maintenance mode (stub)")