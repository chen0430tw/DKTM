"""
plan.py
-------

This module contains simple heuristics for constructing a transition
plan for the Dynamic Kernel Transition Mechanism (DKTM).  Given a
BinaryTwin descriptor, an exploration factor (from the SOSA adapter) and
retina probe results, a plan is built as a list of phase windows.  Each
phase window is a dictionary containing a list of command names that
should be executed together.  The planner is intentionally lightweight
so it can be easily replaced by more sophisticated policies later.

The default strategy is to be conservative when the system is highly
stressed (measured by the retina `E_mean` and low exploration factor)
and more aggressive when conditions appear stable.  If the global edge
mean (E_mean) is high we perform deeper quiescence and a health check
before deciding whether to commit or roll back based on the exploration
factor.  If E_mean is low we perform a lighter sequence and may exit
without committing if the exploration factor is too low.

The commands referenced here must correspond to keys in
``command_dict.COMMANDS``.  If you add or modify commands there you
should update this planner accordingly.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .command_dict import get_group_id

def build_plan(binary_twin: Dict[str, Any], explore_factor: float, retina_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate a simple plan based on binary‑twin features, exploration factor and retina result.

    Parameters
    ----------
    binary_twin : dict
        A dictionary with keys ``x_cont`` (continuous features) and ``b_bits`` (discrete flags) as returned by
        the adapter.  Continuous features are [avg_energy, diversity, size_norm]; discrete bits are
        [high_energy, many_types, big_window].
    explore_factor : float
        A number in [0, 1] indicating whether the system should explore (higher values) or exploit/settle
        (lower values).  It is derived from the SOSA adapter and influences how aggressive the plan is.
    retina_result : dict
        The result of the retina probe, containing at least ``E_mean`` (global edge strength) and
        ``hotspots``.  ``E_mean`` summarises how volatile the current state is.

    Returns
    -------
    list of dict
        A list of phase windows.  Each entry is a dict with a ``commands`` key holding a list of
        command names.  The planner does not attempt to interleave commands at a finer granularity; it
        is the executor's responsibility to sequence them.
    """
    phases: List[Dict[str, Any]] = []

    # Extract features
    E_mean: float = float(retina_result.get("E_mean", 0.0))
    avg_energy, diversity, size_norm = (binary_twin.get("x_cont") or [0.0, 0.0, 0.0])
    high_energy, many_types, big_window = (binary_twin.get("b_bits") or [False, False, False])

    # The first phase always enters a maintenance window
    phases.append({"commands": ["enter_maintenance"]})

    # If the system is under high stress (many edges), quiesce more and perform health checks
    if E_mean > 0.5 or high_energy:
        # Tiered quiescence of services and I/O flush
        phases.append({"commands": ["freeze_services", "quiesce_services_tier1", "flush_io"]})
        phases.append({"commands": ["quiesce_drivers_soft", "health_check"]})
        # Decide whether to commit based on exploration factor
        if explore_factor > 0.5:
            phases.append({"commands": ["commit_transition"]})
        else:
            phases.append({"commands": ["rollback_transition"]})
    else:
        # Low stress – lighter plan
        phases.append({"commands": ["flush_io"]})
        # If there are many distinct action types or big window, still quiesce drivers
        if many_types or big_window:
            phases.append({"commands": ["quiesce_drivers_soft"]})
        # Commit or exit without commit based on explore factor
        if explore_factor > 0.3:
            phases.append({"commands": ["commit_transition"]})
        else:
            phases.append({"commands": ["exit_maintenance"]})

    return phases
