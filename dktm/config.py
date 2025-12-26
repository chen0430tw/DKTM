"""
Configuration utilities for DKTM
--------------------------------

This module provides helper functions to load and merge configuration
files for the DKTM orchestrator.  Configurations may be supplied in
YAML or JSON format.  A default configuration is provided and will
be merged with any user‑supplied configuration such that missing
values fall back to sensible defaults.

The default configuration covers logging, executor behaviour and
orchestrator parameters (e.g. the number of windows and window
duration).  New keys may be added to this file as the project
evolves.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # YAML parsing will be unavailable if pyyaml is not installed


# ----------------------------------------------------------------------
# Default configuration
# ----------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "dktm": {
        # Number of windows to simulate in the demo.  This parameter is
        # primarily used by the demo function in dktm.py and has no
        # impact on the core algorithm.
        "num_windows": 3,
        # Duration of each window in seconds.  A shorter window length
        # results in more frequent flushes and plans.
        "window_length": 2.0,
    },
    "logging": {
        # Logging level.  One of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
        "level": "INFO",
        # Optional file name to write logs to.  If None, logs are not
        # written to a file.
        "file": None,
    },
    "executor": {
        # Execution mode: 'dry-run' prints actions but does not perform
        # any system changes.  'real-run' invokes handlers defined in
        # executor.py.  Defaults to dry‑run for safety.
        "mode": "dry-run",
        # If true, automatically reboot after committing a transition.
        "auto_reboot": False,
        # A list of BCD entry IDs (strings) for Windows PE environments.
        # When committing a transition on Windows, the first entry in
        # this list will be used as the one‑time boot sequence.  You
        # should update this list in your configuration to match the
        # GUIDs of your WinPE entries.  Leave empty for no action.
        "winpe_entry_ids": [],
        # Path to a marker file used to indicate pending transitions.
        # The marker is created when committing a transition and
        # deleted on rollback.  On Windows this should be an absolute
        # path; on POSIX systems a temporary directory may be used.
        "marker_path": "dktm_transition.marker",
    },
}


def _merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries.

    Values in dictionary ``b`` override those in ``a``.  Nested
    dictionaries are merged recursively.  This helper is used to
    combine the default configuration with a user‑supplied
    configuration.

    Parameters
    ----------
    a : dict
        Base dictionary whose keys will be overridden by ``b``.
    b : dict
        Dictionary with override values.

    Returns
    -------
    dict
        A new dictionary containing the merged values.
    """
    result: Dict[str, Any] = dict(a)
    for key, value in b.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | None) -> Dict[str, Any]:
    """Load a configuration file and merge it with the default config.

    The file may be YAML or JSON.  If the file does not exist or
    cannot be parsed, the default configuration will be returned.

    Parameters
    ----------
    path : str or None
        Path to a YAML or JSON configuration file.  If None, only
        the default configuration is returned.

    Returns
    -------
    dict
        The merged configuration.
    """
    cfg = DEFAULT_CONFIG
    if path is None:
        return cfg
    if not os.path.isfile(path):
        return cfg
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        data: Dict[str, Any]
        # Detect YAML vs JSON by first character
        if text.lstrip().startswith("{"):
            data = json.loads(text)
        else:
            if yaml is None:
                raise RuntimeError(
                    "PyYAML is not installed; cannot parse YAML configuration"
                )
            data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            return cfg
        cfg = _merge_dicts(cfg, data)
    except Exception:
        # On any error, fall back to default config
        return cfg
    return cfg


def dump_default_config(path: str) -> None:
    """Write the default configuration to a file in YAML format.

    Parameters
    ----------
    path : str
        Destination path to write the YAML file.
    """
    # Prefer YAML if available
    try:
        if yaml is not None:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(DEFAULT_CONFIG, f)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONFIG, f, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to write config to {path}: {e}")