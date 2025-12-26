"""
DKTM Orchestrator
==================

This module ties together the various components of the Dynamic Kernel Transition
Mechanism (DKTM) proof-of-concept. It demonstrates how a simple state machine
could gather system observations, feed them into the SparkSeedSOSA model via the
SosaAdapter, probe the state with a "retina" to estimate structural stress,
derive a plan from the Binary‑Twin and explore factor, and finally execute
actions (here we simply print them for demonstration purposes).

The goal of this orchestrator is not to perform any real kernel operations,
but to show how the components interact. It generates synthetic system state
matrices, processes them, and prints a sequence of phases and commands
recommended by the planning module.

Usage:
    python dktm.py

This will run a short demonstration of the DKTM pipeline for a few
iterations. Each iteration represents one "window" of events processed by
SparkSeedSOSA. At the end of each window, the orchestrator flushes the
adapter, derives a plan and prints the results.

Note: To keep dependencies light and avoid external packages, this script
relies on ``numpy`` only. If ``numpy`` is unavailable in the environment,
consider installing it or replacing the synthetic state generation with
simple Python lists.
"""

import time
import random
from typing import List, Dict, Any
import argparse
import logging

# Ensure the ``dktm`` package and its wrappers can be imported when this file
# is executed as a script (e.g., via ``python dktm/dktm.py``).  When
# ``__package__`` is not set, Python treats this file as a top‑level module
# named ``dktm``, which prevents sibling modules (e.g., ``executor``) from
# being found via standard package imports.  We append both the parent
# directory and the package directory to ``sys.path`` so that imports of
# ``executor``, ``config`` and ``platform_ops`` resolve to the correct
# modules.
if __package__ is None or __package__ == "":
    import os
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

try:
    import numpy as np
except ImportError:
    np = None  # Fallback: will generate simple lists instead of arrays

from adapter import SosaAdapter
from retina_probe import retina_probe
from plan import build_plan


def generate_state_matrix(size: int = 10) -> Any:
    """Generate a synthetic system state matrix.

    Parameters
    ----------
    size : int
        The width and height of the square matrix.

    Returns
    -------
    ndarray or list of lists
        A matrix of random floats in the range [0, 1].
    """
    if np is not None:
        return np.random.rand(size, size)
    else:
        return [[random.random() for _ in range(size)] for _ in range(size)]


def flatten_matrix(mat: Any) -> List[float]:
    """Flatten a 2D matrix into a 1D list of floats.

    This helper handles both NumPy arrays and nested Python lists.

    Parameters
    ----------
    mat : ndarray or list of lists
        The matrix to flatten.

    Returns
    -------
    List[float]
        The flattened list.
    """
    if np is not None and isinstance(mat, np.ndarray):
        return mat.flatten().tolist()
    else:
        return [item for row in mat for item in row]


def demo(num_windows: int = 3, window_length: float = 2.0, executor: "Executor | None" = None) -> None:
    """Run a demonstration of the DKTM pipeline.

    The function creates a SosaAdapter with a specified window length,
    repeatedly generates synthetic system states, probes them using the
    retina, accumulates events in the adapter and flushes them once the
    window expires. After each flush, it uses the Binary‑Twin and explore
    factor to build a plan and prints the resulting phases.

    Parameters
    ----------
    num_windows : int
        Number of windows to simulate.
    window_length : float
        Duration of each window in seconds. For demonstration purposes,
        a small value makes the demo run quickly.
    """
    adapter = SosaAdapter(dt_window=window_length)

    for w in range(num_windows):
        print(f"\n=== Window {w + 1}/{num_windows} ===")
        start_time = time.time()
        # We will generate a few samples within each window
        while time.time() - start_time < window_length:
            state = generate_state_matrix()
            # Probe the state to get E_map, E_mean and hotspots
            retina_out = retina_probe(state)
            # Flatten state as observation vector
            obs_vec = flatten_matrix(state)
            # For demonstration, we use a placeholder action. In a real system
            # this might come from a decision engine, a prior plan, or
            # another subsystem. Here we just use 'noop'.
            action = 'noop'
            # Submit the event to the adapter. Internally, events are
            # buffered until the window expires.
            adapter.submit_event(obs_vec, action)
            # Sleep briefly to avoid tight looping
            time.sleep(0.1)
        # When the window duration has elapsed, flush events. This
        # returns BinaryTwin, explore_factor and state distribution.
        flush_result = adapter.flush()
        if flush_result is None:
            # If flush returned None, no events were collected; skip.
            print("No events to process in this window.")
            continue
        bt, explore_factor, state_dist = flush_result
        # Use the retina output from the last state in the window to
        # inform the plan. In a real implementation you might want to
        # aggregate retina outputs over the window.
        # Use the retina result dictionary returned by retina_probe
        # Extract continuous and discrete features from the BinaryTwin dataclass
        binary_twin_dict = {
            "x_cont": list(bt.x_cont) if hasattr(bt, "x_cont") else None,
            "b_bits": list(bt.b_bits) if hasattr(bt, "b_bits") else None,
        }
        retina_dict = retina_out  # already contains E_map, E_mean and hotspots
        # Build a plan based on Binary‑Twin, explore factor and retina
        plan = build_plan(binary_twin_dict, explore_factor, retina_dict)
        # Log summary of window metrics
        E_mean = retina_dict.get("E_mean", 0.0)
        logging.getLogger("dktm").info(
            "BinaryTwin continuous: %s, discrete bits: %s", bt.x_cont, bt.b_bits
        )
        logging.getLogger("dktm").info(
            "Explore factor: %.3f, E_mean: %.3f", explore_factor, E_mean
        )
        # Execute or display the plan
        if executor is None:
            logging.getLogger("dktm").info("Generated plan:")
            for i, phase in enumerate(plan):
                commands = ", ".join(phase["commands"])
                logging.getLogger("dktm").info("  Phase %d: %s", i + 1, commands)
        else:
            executor.run_plan(plan)


def _setup_logging(log_cfg: Dict[str, Any]) -> None:
    """Configure the root logging handler based on the config.

    Parameters
    ----------
    log_cfg : dict
        Logging configuration with 'level' and optional 'file'.
    """
    level_name = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handlers: List[logging.Handler] = []
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    handlers.append(stream_handler)
    log_file = log_cfg.get("file")
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        handlers.append(file_handler)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def main(argv: List[str] | None = None) -> None:
    """Entry point for the DKTM orchestrator when executed as a script."""
    parser = argparse.ArgumentParser(description="DKTM Orchestrator")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML/JSON configuration file"
    )
    parser.add_argument(
        "--dump-default-config",
        type=str,
        dest="dump_cfg",
        help="Dump the default configuration to the specified path and exit",
    )
    parser.add_argument(
        "--mode",
        choices=["dry-run", "real-run"],
        help="Override executor mode (dry-run or real-run)",
    )
    parser.add_argument(
        "--auto-reboot",
        action="store_true",
        help="Set executor.auto_reboot to true (only meaningful on Windows)",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Roll back a committed transition and exit",
    )
    args = parser.parse_args(argv)

    # Defer imports of configuration and executor until runtime to avoid
    # circular import issues when running as a script.  The wrapper
    # modules at the project root forward to the implementations in the
    # dktm package.
    import config as _config
    from executor import Executor  # type: ignore
    import platform_ops as _platform_ops  # type: ignore

    # Handle dumping default config
    if args.dump_cfg:
        _config.dump_default_config(args.dump_cfg)
        print(f"Default configuration written to {args.dump_cfg}")
        return

    # Load configuration and merge overrides
    cfg = _config.load_config(args.config)
    # Merge command‑line overrides
    if args.mode:
        cfg.setdefault("executor", {})["mode"] = args.mode
    if args.auto_reboot:
        cfg.setdefault("executor", {})["auto_reboot"] = True

    # Setup logging
    _setup_logging(cfg.get("logging", {}))

    # If rollback is requested, perform it and exit
    if args.rollback:
        logging.getLogger("dktm").info("Performing rollback of transition")
        _platform_ops.rollback_transition(
            winpe_entry_ids=cfg.get("executor", {}).get("winpe_entry_ids", []),
            marker_path=cfg.get("executor", {}).get("marker_path", "dktm_transition.marker"),
        )
        print("Rollback completed.")
        return

    # Instantiate executor
    executor_cfg = cfg.get("executor", {})
    exec_obj = Executor(executor_cfg)

    # Run the demo with the configured parameters
    demo(
        num_windows=int(cfg.get("dktm", {}).get("num_windows", 3)),
        window_length=float(cfg.get("dktm", {}).get("window_length", 2.0)),
        executor=exec_obj,
    )


if __name__ == '__main__':
    main()