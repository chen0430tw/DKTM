"""
retina_probe.py
-----------------

This module implements a simple retina-like probe for the Dynamic Kernel
Transition Mechanism (DKTM). It provides utilities to convert a system
state representation into a gradient magnitude map (`E_map`) and a
scalar measure (`E_mean`) capturing the overall level of structural
variation in that state.  The hotspots are returned as a list of
coordinates where the gradient exceeds a given threshold.

The intent is to mirror the behaviour of a human retina or edge map
extraction: highlight where changes or boundaries occur and yield a
summary statistic to inform higher level decision making.  In our
simulation the state is represented as a 2‑D matrix of numeric
values (for example, resource usage, a driver dependency matrix or
other system metrics).  More sophisticated implementations could use
multi‑channel tensors or leverage existing vision libraries.  For our
purposes a simple finite difference operator suffices.

Example usage:

    from retina_probe import retina_probe

    state = [[0.1, 0.2, 0.5],
             [0.2, 0.3, 0.7],
             [0.4, 0.4, 0.9]]
    result = retina_probe(state)
    print(result['E_mean'], result['hotspots'])

The result includes the edge map (`E_map`), a global mean of that map
(`E_mean`) and the list of hotspot indices.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

def retina_probe(state_matrix: Sequence[Sequence[float]], *, threshold: float = 0.7) -> Dict[str, Any]:
    """Compute a simple gradient-based retina probe on a 2D state matrix.

    Parameters
    ----------
    state_matrix : Sequence[Sequence[float]]
        A 2‑D array-like structure representing the current system state.  It
        can be a list of lists or a numpy array.  Values should be
        convertible to floats.
    threshold : float, optional
        A threshold between 0 and 1 used to determine which positions in
        the edge map are considered hotspots.  Hotspots are positions
        where the normalized gradient magnitude exceeds this value.

    Returns
    -------
    dict
        A dictionary with keys:

        * ``E_map`` – a 2‑D numpy array of the same shape as
          ``state_matrix`` holding values in [0, 1] corresponding to
          relative gradient magnitudes.  If the input has only one
          element in a row or column, zeros will be returned.
        * ``E_mean`` – the mean of ``E_map``, a scalar summarising the
          overall level of variation.
        * ``hotspots`` – a list of coordinate pairs (row, column) where
          ``E_map`` is strictly greater than ``threshold``.

    Notes
    -----
    The gradient is computed via simple forward differences along both
    axes.  The last row/column are extended by the previous row/column
    respectively to preserve dimensionality.  The gradient magnitude is
    normalised by the 95th percentile of the magnitude to achieve
    robustness to outliers.  If the 95th percentile is zero (all
    entries are equal), the raw magnitude is used directly.
    """
    # Convert input to float ndarray
    arr = np.asarray(state_matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("state_matrix must be a 2‑D array or nested sequence of numbers")

    # Compute finite differences along axis 1 (columns) and axis 0 (rows).
    # We append the last column/row to maintain shape.
    gx = np.diff(arr, axis=1)
    gx = np.concatenate([gx, gx[:, -1:, ...]], axis=1)  # replicate last diff for last column
    gy = np.diff(arr, axis=0)
    gy = np.concatenate([gy, gy[-1:, ...]], axis=0)  # replicate last diff for last row

    # Gradient magnitude
    mag = np.sqrt(gx ** 2 + gy ** 2)

    # Robust normalisation by 95th percentile
    p95 = np.percentile(mag, 95)
    if p95 == 0:
        E_map = mag.copy()
    else:
        E_map = np.clip(mag / p95, 0.0, 1.0)

    # Compute mean value
    E_mean = float(np.mean(E_map))

    # Determine hotspots (positions with high edge magnitude)
    hotspot_indices: List[List[int]] = []
    if threshold is not None:
        mask = E_map > threshold
        hotspot_indices = list(map(lambda idx: [int(idx[0]), int(idx[1])], np.argwhere(mask)))

    return {
        "E_map": E_map,
        "E_mean": E_mean,
        "hotspots": hotspot_indices,
    }

# If run as a script, demonstrate with a random matrix.
if __name__ == "__main__":
    # Simple demonstration
    np.random.seed(0)
    mat = np.random.rand(5, 5)
    result = retina_probe(mat)
    print("E_mean:", result["E_mean"])
    print("Hotspots:", result["hotspots"])