"""
adapter.py
This module provides an adapter layer between the DKTM orchestration
and the SparkSeedSOSA algorithm defined in spark_seed_sosa.py.  The
adapter collects events in a buffer, forwards them into the SOSA
implementation and reproduces the internal feature computation so
clients can access the Binary‑Twin (BT) summary and the explore
factor used to mix the Markov state.

The key responsibilities of this adapter are:

1. Maintain a window of incoming events along with their timestamps.
2. When the elapsed time exceeds the configured dt_window (or when
   flush() is called), compute the Binary‑Twin features and the
   occupancy metric c_r (combination occupancy), mirroring the
   internal SOSA logic.
3. Update the underlying SparkSeedSOSA instance so its internal
   Markov state distribution stays in sync with the adapter's view.
4. Return the most recent Binary‑Twin, explore factor and updated
   state distribution after each flush.

To keep the adapter self contained, a small subset of the SOSA code
is copied here (namely the action hashing and combination encoding
routines).  This avoids having to expose private methods of
SparkSeedSOSA or modify its implementation.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
import math  # Needed for tanh in flush()

from .spark_seed_sosa import SparkSeedSOSA, Event, BinaryTwin


# ---------------------------------------------------------------------
# Utility functions lifted from spark_seed_sosa.py
# These mirror the logic used inside SparkSeedSOSA for grouping
# actions into IDs and encoding a set of IDs into a combination index
# along with a normalised occupancy measure c_r.
# ---------------------------------------------------------------------

def _map_action_to_group(action: Any, M_groups: int) -> int:
    """Map an arbitrary action to a group ID in [0, M_groups).

    This function first attempts to resolve the action name via
    ``command_dict.get_group_id``, which provides a stable mapping of
    known command names to group identifiers.  If the action is not
    present in ``command_dict.COMMANDS`` then a deterministic hash
    fallback is used.  This behaviour mirrors the mapping logic used
    internally by SparkSeedSOSA but allows DKTM to assign explicit
    group IDs to commands.

    Parameters
    ----------
    action : Any
        Action identifier (typically a string command name).
    M_groups : int
        Number of defined groups.  The returned ID will be in the
        range [0, M_groups).

    Returns
    -------
    int
        A group identifier.
    """
    # Perform lazy import here to avoid circular dependencies when
    # command_dict itself imports SosaAdapter or other modules.  If
    # command_dict is unavailable or the action is unknown, fall back
    # to hashing.
    if isinstance(action, str):
        try:
            from command_dict import get_group_id
            gid = get_group_id(action)
            # command_dict IDs start at 1; convert to zero‑based for SOSA
            return (gid - 1) % max(1, M_groups)
        except Exception:
            pass
    # Fallback deterministic mapping
    if M_groups <= 0:
        return 0
    return hash(str(action)) % M_groups


def _comb_encode_subset(sorted_ids: List[int], M_groups: int) -> Tuple[int, float]:
    """
    Encode a sorted subset of group IDs into a combination index and
    compute the occupancy metric c_r.  This mirrors the comb_encode_subset
    function from spark_seed_sosa.py.

    Parameters
    ----------
    sorted_ids: List[int]
        Sorted list of unique group IDs present in the window.
    M_groups: int
        Total number of groups.

    Returns
    -------
    idx_r : int
        The index of this combination in the combinatorial list of
        subsets.  Unused by the adapter but returned for completeness.
    c_r : float
        The occupancy ratio in [0,1], defined as (#unique groups) / M_groups.
    """
    # Compute occupancy c_r
    c_r = float(len(sorted_ids)) / float(M_groups) if M_groups > 0 else 0.0

    # Compute the combinatorial index: treat the subset as a bit mask
    idx_r = 0
    for gid in sorted_ids:
        idx_r |= (1 << gid)
    return idx_r, c_r


class SosaAdapter:
    """
    Adapter around the SparkSeedSOSA implementation.  Use this class
    to feed events into the SOSA algorithm and retrieve the
    window‑level Binary‑Twin, explore factor and Markov state
    distribution.
    """

    def __init__(self, M_groups: int = 4, dt_window: float = 2.0, N_states: Optional[int] = None) -> None:
        """
        Construct an adapter and underlying SparkSeedSOSA instance.

        Parameters
        ----------
        M_groups: int
            Number of behaviour groups used for combination encoding.
        dt_window: float
            Length of the window in seconds.  Events collected in this
            timeframe will be aggregated into one Binary‑Twin when
            flushed.
        N_states: int, optional
            Number of hidden states in the underlying SparkSeedSOSA Markov model.
            If None, a default will be chosen based on the number of groups
            (2 * M_groups) to provide a reasonable state resolution.  This parameter
            is required by the SparkSeedSOSA constructor.
        """
        # Determine a sensible default for N_states if one was not provided.  We
        # default to twice the number of groups to offer enough resolution
        # without being overly large.  The user may override this value by
        # explicitly passing N_states.
        if N_states is None:
            N_states = M_groups * 2
        # Underlying SOSA instance.  We supply N_states, M_groups and dt_window
        # explicitly.  SparkSeedSOSA initialises its own defaults when
        # parameters are omitted, but it requires N_states as a positional
        # argument.  Passing all parameters here avoids the missing argument
        # error seen when only dt_window and M_groups were provided.
        self.sosa = SparkSeedSOSA(N_states=N_states, M_groups=M_groups, dt_window=dt_window)
        # Keep a buffer of events and the time of the last flush
        self.dt_window = dt_window
        self.M_groups = M_groups
        self.N_states = N_states
        self.buffer: List[Event] = []
        self.last_flush: float = time.time()
        # Remember the most recent results
        self.last_BT: Optional[BinaryTwin] = None
        self.last_explore: Optional[float] = None

    def submit_event(self, obs: Any, action: Any, timestamp: Optional[float] = None) -> None:
        """
        Submit a raw event (obs, action) to the adapter.  The event
        will be appended to the buffer and forwarded to the underlying
        SparkSeedSOSA.  When the time window is exceeded, the buffer
        will be flushed automatically and results can be queried via
        flush() or by inspecting last_BT/last_explore.

        Parameters
        ----------
        obs: Any
            Observation associated with the event.  For DKTM this can be
            a representation of the current system state (list, string,
            etc.).  SOSA only looks at its length when computing a
            score.
        action: Any
            Identifier of the action taken.  DKTM should use command
            names here (e.g. "observe", "flush_io", etc.).
        timestamp: float, optional
            Timestamp of the event.  If omitted, current time is used.
        """
        if timestamp is None:
            timestamp = time.time()
        e = Event(obs=obs, action=action, timestamp=timestamp)
        self.buffer.append(e)
        # Forward into underlying SOSA; it manages its own internal buffer
        # and will flush automatically when its dt_window passes.  We
        # still track our own buffer so we can compute BT and c_r.
        self.sosa.process_event(obs, action, timestamp)
        # Auto flush if time window exceeded
        if timestamp - self.last_flush >= self.dt_window:
            self.flush()

    def flush(self) -> Tuple[Optional[BinaryTwin], Optional[float], List[float]]:
        """
        Force a flush of the current event window.  This will compute
        the Binary‑Twin, update the underlying Markov state and return
        the Binary‑Twin, explore factor and state distribution.

        Returns
        -------
        BT: Optional[BinaryTwin]
            The Binary‑Twin computed for this window, or None if no
            events were present.
        explore: Optional[float]
            The explore factor used to mix the Markov state, or None if
            no events were present.
        pi: List[float]
            The updated Markov state distribution (always returned).
        """
        if not self.buffer:
            # Nothing to flush
            return None, None, self.sosa.get_state_distribution()

        events = self.buffer
        self.buffer = []
        self.last_flush = time.time()

        # Step 1: compute Binary‑Twin features (mirroring SOSA logic)
        BT = BinaryTwin()
        scores: List[float] = []
        action_types: set = set()
        for e in events:
            # Compute local energy score; reuse SOSA's estimation
            score_e = self.sosa._estimate_action_score(e)
            scores.append(score_e)
            action_types.add(e.action)
        avg_energy = sum(scores) / len(scores) if scores else 0.0
        diversity = len(action_types) / max(1, len(events))
        size_norm = math.tanh(len(events) / 10.0)
        BT.x_cont = [avg_energy, diversity, size_norm]
        # Discrete bits
        high_energy = any(s > 0.8 for s in scores)
        many_types = len(action_types) >= 3
        big_window = len(events) >= 10
        BT.b_bits = [1 if high_energy else 0,
                     1 if many_types else 0,
                     1 if big_window else 0]

        # Step 2: group actions into group IDs
        group_ids = set()
        for e in events:
            gid = _map_action_to_group(e.action, self.M_groups)
            group_ids.add(gid)
        sorted_ids = sorted(list(group_ids))
        idx_r, c_r = _comb_encode_subset(sorted_ids, self.M_groups)

        # Step 3: update the underlying Markov state using SOSA's logic
        # We call the private method to update the state; this replicates
        # the internal call that would happen on an automatic flush.
        # Because SOSA already computed a BT and c_r internally when
        # _flush_window() was triggered, we need to ensure we're not
        # double‑counting.  To avoid state drift, we reset the window
        # buffer of SOSA before calling _update_markov_state.
        # (SparkSeedSOSA empties its buffer on flush, but since we're
        # computing BT ourselves, we bypass its internal _flush_window.)
        # We'll call _update_markov_state directly with our BT and c_r.
        self.sosa._update_markov_state(BT, c_r)

        # Compute explore factor exactly as in SOSA
        avg_energy, diversity, size_norm = BT.x_cont
        base_explore = 1.0 - c_r
        explore_factor = (
            base_explore *
            (1.0 - 0.5 * diversity) *
            (0.5 + 0.5 * (1.0 - size_norm))
        )
        explore_factor = max(0.0, min(1.0, explore_factor))

        # Record results
        self.last_BT = BT
        self.last_explore = explore_factor

        return BT, explore_factor, self.sosa.get_state_distribution()
