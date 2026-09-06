"""Loading, column conventions, assumption checks and submission writing.

The batch-holdout split lives in ``validate.forward_chaining``: train on batches
< k, validate on batch k, for k = 4..9. One implementation, so everything scores
the same way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
SUBMISSION_DIR = HERE / "output"

ID_COL = "measurement_id"
BATCH_COL = "batch"
CONC_COL = "concentration"
TARGET_COL = "gas_class"

N_SENSORS = 16
N_DESCRIPTORS = 8
FEATURE_COLS = [f"feat_{i}" for i in range(1, N_SENSORS * N_DESCRIPTORS + 1)]

META_COLS_TRAIN = [ID_COL, BATCH_COL, TARGET_COL, CONC_COL]
META_COLS_TEST = [ID_COL, BATCH_COL, CONC_COL]

TRAIN_BATCHES = tuple(range(1, 10))
TEST_BATCH = 10

TRAIN_SHAPE = (10310, 132)
TEST_SHAPE = (3600, 131)

CLASS_NAMES = {
    1: "Ethanol",
    2: "Ethylene",
    3: "Ammonia",
    4: "Acetaldehyde",
    5: "Acetone",
    6: "Toluene",
}
CLASSES = np.array(sorted(CLASS_NAMES))


def load_train(path: Path | None = None) -> pd.DataFrame:
    """Labelled measurements from batches 1-9."""
    return _read_csv(path or DATA_DIR / "train.csv")


def load_test(path: Path | None = None) -> pd.DataFrame:
    """Unlabelled measurements from batch 10."""
    return _read_csv(path or DATA_DIR / "test.csv")


def load_sample_submission(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or DATA_DIR / "sample_submission.csv")


def load_data(data_dir: Path = DATA_DIR, check: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both frames and, by default, assert the assumptions we build on."""
    train = load_train(data_dir / "train.csv")
    test = load_test(data_dir / "test.csv")
    if check:
        check_assumptions(train, test)
    return train, test


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in [ID_COL, BATCH_COL, CONC_COL, *FEATURE_COLS] if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {missing[:5]}...")
    return df


def check_assumptions(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Fail loudly if the data stops looking like what the modelling assumes.

    Cheap insurance against a re-download, a re-export or a merge quietly
    changing the ground under the pipeline.
    """
    # --- structural ---
    assert train.shape == TRAIN_SHAPE, f"train shape changed: {train.shape}"
    assert test.shape == TEST_SHAPE, f"test shape changed: {test.shape}"
    assert train.isna().sum().sum() == 0, "NaNs appeared in train"
    assert test.isna().sum().sum() == 0, "NaNs appeared in test"
    assert set(train.columns) == set(META_COLS_TRAIN + FEATURE_COLS), "train columns changed"
    assert set(test.columns) == set(META_COLS_TEST + FEATURE_COLS), "test columns changed"

    # --- domain: the things that will bite you if they silently change ---
    assert test[BATCH_COL].unique().tolist() == [TEST_BATCH], "test batch assumption broken"
    assert sorted(train[BATCH_COL].unique()) == list(TRAIN_BATCHES), "train batches changed"
    assert sorted(train[TARGET_COL].unique()) == list(CLASSES), "train classes are no longer 1-6"

    class_by_batch = pd.crosstab(train[BATCH_COL], train[TARGET_COL])
    missing_class_6 = class_by_batch.loc[[3, 4, 5], 6].sum()
    assert missing_class_6 == 0, (
        "Class 6 now present in batches 3-5 - re-check the drift assumption, "
        "this is expected to be 0."
    )


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray]:
    """Return (features+meta, y, batch). ``y`` is None for the unlabelled test frame."""
    y = df[TARGET_COL].to_numpy() if TARGET_COL in df.columns else None
    return df.drop(columns=[c for c in (ID_COL, TARGET_COL) if c in df.columns]), y, df[BATCH_COL].to_numpy()


def next_submission_path(stem: str = "submission", suffix: str = ".csv") -> Path:
    """Return the first unused ``submission_NN.csv`` so runs never clobber each other."""
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        candidate = SUBMISSION_DIR / f"{stem}_{n:02d}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def write_submission(ids, preds, path: Path | None = None) -> Path:
    """Write a ``measurement_id,gas_class`` CSV and sanity-check it against the sample.

    With no explicit ``path`` this auto-increments: submission_01.csv, submission_02.csv, ...
    Passing ``--out`` still writes exactly where you point it (and will overwrite).
    """
    path = Path(path) if path else next_submission_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    sub = pd.DataFrame({ID_COL: np.asarray(ids), TARGET_COL: np.asarray(preds).astype(int)})

    sample = load_sample_submission()
    if len(sub) != len(sample):
        raise ValueError(f"submission has {len(sub)} rows, expected {len(sample)}")
    if set(sub[ID_COL]) != set(sample[ID_COL]):
        raise ValueError("submission measurement_ids do not match sample_submission.csv")
    bad = set(sub[TARGET_COL]) - set(CLASS_NAMES)
    if bad:
        raise ValueError(f"submission contains labels outside 1-6: {sorted(bad)}")

    # Keep the sample's row order so the grader sees exactly what it expects.
    sub = sample[[ID_COL]].merge(sub, on=ID_COL, how="left")
    sub.to_csv(path, index=False)
    return path


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Per-batch row counts and class balance - the first thing worth looking at."""
    if TARGET_COL not in df.columns:
        return df.groupby(BATCH_COL).size().rename("n").to_frame()
    counts = df.pivot_table(index=BATCH_COL, columns=TARGET_COL, values=ID_COL, aggfunc="count")
    counts.columns = [CLASS_NAMES[c] for c in counts.columns]
    return counts.fillna(0).astype(int).assign(n=lambda d: d.sum(axis=1))


if __name__ == "__main__":
    train, test = load_data()
    print("Loaded OK:", train.shape, test.shape)
    print(describe(train))
