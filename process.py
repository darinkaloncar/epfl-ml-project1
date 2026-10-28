# Preprocessing pipeline for the linear models: writes the imputed features to processed/.
# Every data-dependent step (select_columns, fit_caps, fit_impute) is fitted on train only.
import json
import os

import numpy as np

from recode_brfss import (apply_caps, apply_impute, fit_caps, fit_impute, forced_positive,
                          preprocess, select_columns)

DATA_DIR = "dataset"
OUT_DIR = "processed"
CACHE_DIR = "eda_output"


# (header without Id, Ids, values) of dataset/<name>.csv; use_cache keeps a copy in eda_output/
def read_csv(name, use_cache=True):
    path = os.path.join(DATA_DIR, name + ".csv")
    with open(path) as f:
        header = f.readline().strip().split(",")
    cache = os.path.join(CACHE_DIR, name + ".npy")
    if not use_cache:
        x = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float64)
    elif os.path.exists(cache):
        x = np.load(cache)
    else:
        x = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float64)
        np.save(cache, x)
    return header[1:], x[:, 0], x[:, 1:]


# Targets of dataset/y_train.csv, aligned with `ids`
def join_targets(ids):
    yt = np.genfromtxt(os.path.join(DATA_DIR, "y_train.csv"), delimiter=",", skip_header=1)
    order = np.argsort(yt[:, 0])
    pos = np.minimum(np.searchsorted(yt[order, 0], ids), len(order) - 1)
    assert np.array_equal(yt[order[pos], 0], ids), "some Ids of x_train have no target"
    return yt[order[pos], 1]


def write_csv(path, ids, x, names):
    data = np.column_stack([ids, x])
    fmt = ["%d"] + ["%.10g"] * x.shape[1]
    np.savetxt(path, data, delimiter=",", fmt=fmt, header=",".join(["Id"] + names), comments="")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw_names, ids_tr, raw_tr = read_csv("x_train")
    test_names, ids_te, raw_te = read_csv("x_test")
    assert test_names == raw_names, "train and test columns differ"
    y_tr = join_targets(ids_tr)

    force_tr = forced_positive(raw_tr, raw_names)
    force_te = forced_positive(raw_te, raw_names)

    x_tr, names = preprocess(raw_tr, raw_names)
    x_te, names_te = preprocess(raw_te, raw_names)
    assert names_te == names

    # fit on train
    keep = select_columns(x_tr, y_tr, names)
    cols = [names.index(n) for n in keep]
    x_tr, x_te = x_tr[:, cols], x_te[:, cols]
    caps = fit_caps(x_tr, keep)
    x_tr, x_te = apply_caps(x_tr, keep, caps), apply_caps(x_te, keep, caps)
    fills = fit_impute(x_tr, keep)
    x_tr, x_te = apply_impute(x_tr, keep, fills), apply_impute(x_te, keep, fills)
    assert not np.isnan(x_tr).any() and not np.isnan(x_te).any()

    np.save(os.path.join(OUT_DIR, "x_train.npy"), x_tr)
    np.save(os.path.join(OUT_DIR, "x_test.npy"), x_te)
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_tr)
    np.save(os.path.join(OUT_DIR, "ids_train.npy"), ids_tr.astype(np.int64))
    np.save(os.path.join(OUT_DIR, "ids_test.npy"), ids_te.astype(np.int64))
    np.save(os.path.join(OUT_DIR, "forced_positive_train.npy"), force_tr)
    np.save(os.path.join(OUT_DIR, "forced_positive_test.npy"), force_te)
    write_csv(os.path.join(OUT_DIR, "x_train.csv"), ids_tr, x_tr, keep)
    write_csv(os.path.join(OUT_DIR, "x_test.csv"), ids_te, x_te, keep)
    with open(os.path.join(OUT_DIR, "features.txt"), "w") as f:
        f.write("\n".join(keep) + "\n")
    with open(os.path.join(OUT_DIR, "params.json"), "w") as f:
        json.dump({"features": keep, "caps": caps, "fills": fills}, f, indent=1)

    print(f"train {x_tr.shape}, test {x_te.shape}, features {len(keep)}")
    print(f"forced positive: train {force_tr.sum()} (all y=1: {bool(np.all(y_tr[force_tr] == 1))}), "
          f"test {force_te.sum()}")
    print(f"UNSURE_CHRONIC kept: {'UNSURE_CHRONIC' in keep}; "
          f"rate train {100 * x_tr[:, keep.index('UNSURE_CHRONIC')].mean():.2f}%, "
          f"test {100 * x_te[:, keep.index('UNSURE_CHRONIC')].mean():.2f}%")


if __name__ == "__main__":
    main()
