"""Wrapper for backward compatibility.

Re‑export platform operation functions from ``dktm.platform_ops`` so
that they can be imported from the project root when running scripts
outside the package context.
"""

from dktm.platform_ops import commit_transition, rollback_transition, reboot, handover_control  # noqa: F401