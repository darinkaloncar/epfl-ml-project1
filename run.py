"""Full pipeline that reproduces submission.csv from the raw csv files (numpy + stdlib only).

  1. read dataset/x_train.csv, x_test.csv, y_train.csv (no cached intermediates)
  2. preprocess()       recoding driven by codebook_values.json, skip-logic fills, dropped
                        columns; row-wise, NaN kept (bin 0 of the trees)
  3. fit_binner()       per-feature histogram bins, fitted on the training set
  4. fit_gbdt()         500 trees on the full training set, seeded
  5. threshold 0.21 on the predicted probability, HAREHAB1 override, write the csv

Usage: python run.py [output.csv]
"""
import sys

import numpy as np

from gbdt import GBDTParams, fit_binner, fit_gbdt, predict_scores, sigmoid
from helpers import create_csv_submission
from implementations import UNORDERED_CATEGORICALS
from process import join_targets, read_csv
from recode_brfss import forced_positive, preprocess

PARAMS = GBDTParams(max_depth=4, learning_rate=0.05, min_hessian=10.0, lambda_leaf=1.0,
                    row_subsample=0.7, col_subsample=0.7, seed=0)
N_TREES = 500       # CV early-stopping optimum was 464 trees on 72% of the data
THRESHOLD = 0.21    # F1-optimal threshold of the out-of-fold CV predictions (same in all 5 folds)


def load(split):
    """Parse the raw dataset/<split>.csv and build its features with preprocess().
    Returns (features, feature names, override mask, Ids)."""
    raw_names, ids, raw = read_csv(split, use_cache=False)
    x, names = preprocess(raw, raw_names)
    return x, names, forced_positive(raw, raw_names), ids


def main(out="submission.csv"):
    x_tr, names, _, ids_tr = load("x_train")
    x_te, names_te, forced_te, ids_te = load("x_test")
    assert names_te == names
    y = (join_targets(ids_tr) > 0).astype(float)
    print(f"features: train {x_tr.shape}, test {x_te.shape}")

    binner = fit_binner(x_tr)
    xb_tr, xb_te = binner.transform(x_tr), binner.transform(x_te)
    is_cat = np.array([n in UNORDERED_CATEGORICALS for n in names])
    trees, base, _ = fit_gbdt(xb_tr, y, is_cat, PARAMS, N_TREES)

    p = sigmoid(predict_scores(trees, base, xb_te))
    pred = np.where((p >= THRESHOLD) | forced_te, 1, -1)
    create_csv_submission(ids_te, pred, out)
    print(f"wrote {out}: {len(pred)} rows, {np.mean(pred == 1):.1%} predicted positive")


if __name__ == "__main__":
    main(*sys.argv[1:])
