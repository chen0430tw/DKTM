"""Wrapper for backward compatibility.

This module re‑exports symbols from ``dktm.config`` to allow importing
``config`` from the project root when running scripts outside of the
package context.  It is a thin wrapper around the actual
implementation inside the ``dktm`` package.
"""

from dktm.config import *  # noqa: F401,F403