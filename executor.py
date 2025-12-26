"""Wrapper for backward compatibility.

This module provides a convenience import for ``Executor`` from
``dktm.executor``.  It allows running scripts that expect to import
``Executor`` from the project root.
"""

from dktm.executor import Executor  # noqa: F401