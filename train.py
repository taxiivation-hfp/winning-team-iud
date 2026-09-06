"""LDA + logistic regression, self-trained onto the target batch.

    python train.py                      # LDA+LR self-training, validate + submit
    python train.py --method baseline    # plain LR, no self-training
    python train.py --no-submit          # validate only
    python train.py --rounds 3           # number of self-training rounds

Nothing here touches a class marginal: no Sinkhorn, no prior correction, no
target class counts. Pseudo-labels are ranked by the model's own posterior and
the final decision is a plain argmax.

Only batches 9 (the drift proxy) and 7 (the sanity fold) are scored, so the
iteration loop stays tight.
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import data
import validate
from features import features
from sklearn.preprocessing import StandardScaler, QuantileTransformer

PROXY_BATCHES = (9, 7)

# EDIT THIS. One entry per self-training round: the fraction of each class's
# predictions to trust that round. Length = number of rounds.
ROUNDS = tuple(np.linspace(0.001, 0.999, 100)) # TRY THIS
# ROUNDS = tuple(np.linspace(0.0025, 0.975, 40)) # TRY THIS
# ROUNDS = (0.3, 0.5, 0.7)

# Bagging: how many bootstrap members and what share of source rows each sees.
# Self-training is bistable - it locks onto one of a few possible labelings of
# the confusable classes, and tiny input changes flip which one. Members that
# see different 80% draws of the source fall into different basins; averaging
# their probabilities keeps the majority basin and cancels the coin flip.
N_BAG = 15
BAG_FRAC = 0.8


def rounds_schedule(n: int | None) -> tuple[float, ...]:
    """ROUNDS as written, unless --rounds asks for a different count.

    With a count, the endpoints of ROUNDS are kept and n values are spread
    evenly between them - handy for a quick sweep without editing the file.
    """
    if n is None:
        return ROUNDS
    if n == 1:
        return (sum(ROUNDS) / len(ROUNDS),)
    return tuple(np.linspace(ROUNDS[0], ROUNDS[-1], n))


# Two members that fail differently: LDA is drift-fragile but well calibrated
# within a batch, LR is the robust one. Averaging beats either alone.
def lda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.0002) # SHRINKAGE 0.0002 WINS


def lr():
    return LogisticRegression(C=0.5, max_iter=5000)
                              


def prepare(df_source, df_target):
    """Features + scaling for a (source, target) pair.

    ``state_fill`` and the scaler are learned on the source only - the one place
    this pipeline could leak, so it is explicit.
    """
    Xs, fill = features(df_source)
    Xt, _ = features(df_target, fill)
    scaler = StandardScaler().fit(Xs)
    return scaler.transform(Xs), scaler.transform(Xt), df_source[data.TARGET_COL].to_numpy()


def self_train(make, Xs, ys, Xt, rounds):
    """Retrain on the target rows the model is most confident about.

    Selection is per predicted class, ranked by the model's own probability.
    With no marginal to correct it each round, whatever skew the model starts
    with gets reinforced - so more rounds is not automatically better here.
    """
    clf = make().fit(Xs, ys)
    P = clf.predict_proba(Xt)
    classes = clf.classes_

    for frac in rounds:
        pred, conf = classes[P.argmax(1)], P.max(1)

        keep = np.zeros(len(Xt), bool)
        for c in classes:
            idx = np.flatnonzero(pred == c)
            if len(idx):
                keep[idx[np.argsort(-conf[idx])[: max(1, int(frac * len(idx)))]]] = True

        clf = make().fit(np.vstack([Xs, Xt[keep]]), np.concatenate([ys, pred[keep]]))
        P = clf.predict_proba(Xt)

    return P, classes


def run(df_source, df_target, method="selftrain", rounds=()):
    """Return (probabilities, classes) for the target frame."""
    Xs, Xt, ys = prepare(df_source, df_target)

    if method == "baseline":
        clf = lr().fit(Xs, ys)
        return clf.predict_proba(Xt), clf.classes_

    rng = np.random.default_rng(0)
    Ps = []
    for _ in range(N_BAG):
        idx = rng.choice(len(Xs), size=int(BAG_FRAC * len(Xs)), replace=False)
        P, classes = self_train(lda, Xs[idx], ys[idx], Xt, rounds)
        Ps.append(P)
    return np.mean(Ps, axis=0), classes  # LDA only - LR member just makes it worse (tested)

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", default="selftrain", choices=["selftrain", "baseline"])
    p.add_argument("--rounds", type=int, default=None, help="override the length of ROUNDS for a quick sweep")
    p.add_argument("--no-submit", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    rounds = rounds_schedule(args.rounds)
    train = data.load_train()
    # from sklearn.covariance import ledoit_wolf
    # Xs, _, ys = prepare(train, data.load_test())
    # means = {c: Xs[ys == c].mean(0) for c in np.unique(ys)}
    # print("ledoit-wolf shrinkage:", ledoit_wolf(Xs - np.array([means[c] for c in ys]))[1])
    # print(f"method={args.method}  rounds={len(rounds)} {tuple(round(f, 2) for f in rounds)}\n")

    for k in PROXY_BATCHES:
        source = train[train[data.BATCH_COL] != k]
        held = train[train[data.BATCH_COL] == k]
        y = held[data.TARGET_COL].to_numpy()

        P, classes = run(source, held, args.method, rounds)
        score = validate.macro_f1(y, classes[P.argmax(1)])
        print(f"held-out batch {k}: macro-F1 {score:.4f}  ({len(held)} rows)")

    if args.no_submit:
        return 0

    test = data.load_test()
    P, classes = run(train, test, args.method, rounds)
    pred = classes[P.argmax(1)]

    path = data.write_submission(test[data.ID_COL], pred, args.out)
    counts = np.bincount(pred, minlength=7)[1:]
    print(f"\nwrote {path}")
    print("predicted counts:", counts.tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())