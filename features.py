"""Feature construction: three blocks built from the raw 128 sensor columns.

Decomposes the raw 128 instead of feeding entangled magnitudes:

    pattern   128  which sensors responded relative to each other (gas fingerprint,
                   scale-free: L1 over the 16 sensors, computed per row)
    logscale    1  how strongly the array responded overall
    logconc     1  concentration
                  = 130 columns

A fourth block, logS (sensor state = baseline resistance R0, from S = dR/(ratio-1)),
is computed below but deliberately NOT stacked into X - an earlier iteration
included it and it did not pay for itself. It survives only to produce
``state_fill``.

Descriptor indexing is 0-based here: cube[:, :, 0] is descriptor 1 (dR) and
cube[:, :, 1] is descriptor 2 (ratio) in the 1-based Sensor{s}_Feature{d} naming.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data import FEATURE_COLS, N_DESCRIPTORS, N_SENSORS


def as_sensor_cube(df: pd.DataFrame) -> np.ndarray:
    """(n_rows, 16, 8) view of the feature block."""
    return df[FEATURE_COLS].to_numpy(float).reshape(len(df), N_SENSORS, N_DESCRIPTORS)


def features(df: pd.DataFrame, state_fill: np.ndarray | None = None):
    """Return (X, state_fill).

    ``state_fill`` is the per-sensor median of logS learned on the source frame.
    Because logS is not stacked into X (see the module docstring) it does not
    currently affect the returned features - the plumbing is kept so the block
    can be switched back on without changing either caller.
    """
    cube = as_sensor_cube(df)
    dr, ratio = cube[:, :, 0], cube[:, :, 1]

    l1 = np.abs(cube).sum(axis=1, keepdims=True) + 1e-6  # L1 over the 16 sensors
    pattern = (cube / l1).reshape(len(df), -1)

    logscale = np.log1p(np.abs(dr)).mean(axis=1, keepdims=True)
    logconc = np.log(df["concentration"].to_numpy(float))[:, None]

    with np.errstate(divide="ignore", invalid="ignore"):
        S = dr / (ratio - 1.0)  # = R0, the baseline resistance
        logS = np.log(np.where((dr > 0) & (ratio > 1.02) & (S > 0), S, np.nan))

    if state_fill is None:
        state_fill = np.nanmedian(logS, axis=0)
    logS = np.where(np.isfinite(logS), logS, state_fill)

    X = np.hstack(
        [pattern, logscale, logconc]
    )
    return X, state_fill
