"""Histogram gradient-boosted decision trees for binary classification

Newton boosting on the logistic loss: every round fits a regression tree to the per-sample
gradient g = p - y and Hessian h = p (1 - p) of the loss at the current scores, and each leaf
takes the Newton step -G / (H + lambda_leaf) (G, H = sums of g, h over the leaf).

Features are binned once (fit on the training set): bin 0 holds NaN, bins 1.. the values.
A split sends a set of bins to the left child:
  - ordered features: bins 1..k (a threshold), NaN to whichever side gains more;
  - unordered categoricals: bins sorted by G/H in the node, then a threshold on that order.
Histograms of G and H per (node, feature, bin) come from one np.bincount per level.
"""
from dataclasses import dataclass, field

import numpy as np

N_BINS = 64  # bin 0 = NaN, at most 63 value bins


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


def logistic_loss(y, scores):
    return float(np.mean(np.logaddexp(0, scores) - y * scores))


@dataclass
class Binner:
    thresholds: list  # per feature

    def transform(self, x):
        out = np.zeros(x.shape, dtype=np.uint8)
        for j, t in enumerate(self.thresholds):
            v = x[:, j]
            nan = np.isnan(v)
            out[~nan, j] = 1 + np.searchsorted(t, v[~nan], side="right")
        return out


def fit_binner(x, max_bins=N_BINS - 1):
    """Exact values when a feature has <= max_bins distinct values, quantile bins otherwise."""
    thresholds = []
    for j in range(x.shape[1]):
        v = x[~np.isnan(x[:, j]), j]
        u = np.unique(v)
        if len(u) <= max_bins:
            t = (u[:-1] + u[1:]) / 2
        else:
            t = np.unique(np.quantile(v, np.linspace(0, 1, max_bins + 1)[1:-1]))
        thresholds.append(t)
    return Binner(thresholds)


@dataclass
class Tree:
    feature: np.ndarray    # split feature per node, -1 for a leaf
    left_bins: np.ndarray  # (nodes, N_BINS) bool: bins sent to the left child
    left: np.ndarray       # child indices
    right: np.ndarray
    value: np.ndarray      # leaf values (already scaled by the learning rate)
    depth: int
    gain: np.ndarray = None  # split gain per node (0 for a leaf)

    def predict(self, xb):
        node = np.zeros(len(xb), dtype=np.int64)
        rows = np.arange(len(xb))
        for _ in range(self.depth):
            f = self.feature[node]
            split = f >= 0
            go_left = self.left_bins[node, xb[rows, np.maximum(f, 0)]]
            node = np.where(split, np.where(go_left, self.left[node], self.right[node]), node)
        return self.value[node]


@dataclass
class GBDTParams:
    max_depth: int = 4
    learning_rate: float = 0.1
    lambda_leaf: float = 1.0
    min_hessian: float = 10.0   # minimum sum of h in a child
    min_gain: float = 0.0
    row_subsample: float = 0.7
    col_subsample: float = 0.7
    seed: int = 0


def grow_tree(xb, g, h, is_cat, params, rng):
    """Level-wise growth on the given rows (xb, g, h already subsampled)."""
    n, d = xb.shape
    B, lam = N_BINS, params.lambda_leaf
    feats = np.sort(rng.choice(d, max(1, int(round(params.col_subsample * d))), replace=False))
    xs = xb[:, feats]
    base = (np.arange(len(feats)) * B)[None, :] + xs.astype(np.int64)   # (n, d_s) flat bin index
    cat = is_cat[feats]

    feature, left_bins, left, right, value = [-1], [np.zeros(B, bool)], [-1], [-1], [0.0]
    gains = [0.0]
    node_of_row = np.zeros(n, dtype=np.int64)
    level_nodes = [0]
    for _ in range(params.max_depth):
        if not level_nodes:
            break
        slot = -np.ones(len(feature), dtype=np.int64)
        slot[level_nodes] = np.arange(len(level_nodes))
        s = slot[node_of_row]
        active = s >= 0
        idx = (s[active, None] * (len(feats) * B) + base[active]).ravel()
        size = len(level_nodes) * len(feats) * B
        G = np.bincount(idx, np.repeat(g[active], len(feats)), size).reshape(len(level_nodes), len(feats), B)
        H = np.bincount(idx, np.repeat(h[active], len(feats)), size).reshape(len(level_nodes), len(feats), B)

        # order the value bins: identity for ordered features, by G/H for categoricals
        order = np.broadcast_to(np.arange(1, B), G[:, :, 1:].shape).copy()
        if cat.any():
            ratio = G[:, cat, 1:] / (H[:, cat, 1:] + lam)
            ratio[H[:, cat, 1:] == 0] = np.inf                  # empty bins go last
            order[:, cat] = 1 + np.argsort(ratio, axis=2, kind="stable")
        Gv = np.take_along_axis(G, order, axis=2)
        Hv = np.take_along_axis(H, order, axis=2)
        GL, HL = np.cumsum(Gv, axis=2)[:, :, :-1], np.cumsum(Hv, axis=2)[:, :, :-1]
        Gt, Ht = G.sum(axis=2, keepdims=True), H.sum(axis=2, keepdims=True)
        Gn, Hn = G[:, :, :1], H[:, :, :1]                          # NaN bin

        def gain(gl, hl):
            gr, hr = Gt - gl, Ht - hl
            ok = (hl >= params.min_hessian) & (hr >= params.min_hessian)
            val = 0.5 * (gl**2 / (hl + lam) + gr**2 / (hr + lam) - Gt**2 / (Ht + lam))
            return np.where(ok, val, -np.inf)

        gain_nan_right = gain(GL, HL)
        gain_nan_left = gain(GL + Gn, HL + Hn)
        nan_left = gain_nan_left > gain_nan_right
        best = np.maximum(gain_nan_left, gain_nan_right).reshape(len(level_nodes), -1)
        arg = best.argmax(axis=1)

        next_nodes = []
        new_node_of_row = node_of_row.copy()
        for k, nid in enumerate(level_nodes):
            gk = best[k, arg[k]]
            Gk, Hk = Gt[k, 0, 0], Ht[k, 0, 0]
            if not np.isfinite(gk) or gk <= params.min_gain:
                value[nid] = -Gk / (Hk + lam) * params.learning_rate
                continue
            fj, cut = divmod(arg[k], B - 2)
            mask = np.zeros(B, bool)
            mask[order[k, fj, : cut + 1]] = True
            mask[0] = nan_left[k, fj, cut]
            feature[nid], left_bins[nid], gains[nid] = feats[fj], mask, gk
            for child in (0, 1):
                feature.append(-1); left_bins.append(np.zeros(B, bool))
                left.append(-1); right.append(-1); value.append(0.0); gains.append(0.0)
            left[nid], right[nid] = len(feature) - 2, len(feature) - 1
            rows = node_of_row == nid
            goes_left = mask[xs[rows, fj]]
            new_node_of_row[np.flatnonzero(rows)[goes_left]] = left[nid]
            new_node_of_row[np.flatnonzero(rows)[~goes_left]] = right[nid]
            next_nodes += [left[nid], right[nid]]
        node_of_row = new_node_of_row
        level_nodes = next_nodes

    # leaves of the last level
    if level_nodes:
        Gs = np.bincount(node_of_row, g, len(feature))
        Hs = np.bincount(node_of_row, h, len(feature))
        for nid in level_nodes:
            value[nid] = -Gs[nid] / (Hs[nid] + lam) * params.learning_rate
    return Tree(np.array(feature), np.array(left_bins), np.array(left), np.array(right),
                np.array(value), params.max_depth, np.array(gains))


@dataclass
class GBDT_state:
    scores: np.ndarray      # current raw scores (log-odds) on the training rows
    trees: list = field(default_factory=list)
    base_score: float = 0.0
    rng: np.random.Generator = None


def GBDT_initialize(y, params):
    p = y.mean()
    base = float(np.log(p / (1 - p)))
    return GBDT_state(scores=np.full(len(y), base), base_score=base,
                      rng=np.random.default_rng(params.seed))


def GBDT_update(xb, y, is_cat, params, state):
    """One boosting round: a Newton step in function space, fitted by a tree."""
    p = sigmoid(state.scores)
    g, h = p - y, p * (1 - p)
    rows = np.flatnonzero(state.rng.random(len(y)) < params.row_subsample)
    tree = grow_tree(xb[rows], g[rows], h[rows], is_cat, params, state.rng)
    state.scores = state.scores + tree.predict(xb)
    state.trees.append(tree)
    return state


def predict_scores(trees, base_score, xb):
    s = np.full(len(xb), base_score)
    for t in trees:
        s += t.predict(xb)
    return s


def feature_importance(trees, d):
    """Total split gain per feature over all trees."""
    imp = np.zeros(d)
    for t in trees:
        split = t.feature >= 0
        np.add.at(imp, t.feature[split], t.gain[split])
    return imp


def fit_gbdt(xb, y, is_cat, params, n_rounds, xb_valid=None, y_valid=None, patience=50, verbose=0):
    """Boost for up to n_rounds; with a validation set, stop when its loss hasn't improved for
    `patience` rounds and keep the best number of trees. Returns (trees, base_score, history)."""
    state = GBDT_initialize(y, params)
    valid_scores = None if xb_valid is None else np.full(len(xb_valid), state.base_score)
    history, best, best_k = [], np.inf, 0
    for k in range(n_rounds):
        state = GBDT_update(xb, y, is_cat, params, state)
        if valid_scores is not None:
            valid_scores += state.trees[-1].predict(xb_valid)
            loss = logistic_loss(y_valid, valid_scores)
            history.append(loss)
            if loss < best:
                best, best_k = loss, k + 1
            elif k + 1 - best_k >= patience:
                break
            if verbose and (k + 1) % verbose == 0:
                print(f"round {k + 1}: train {logistic_loss(y, state.scores):.5f} valid {loss:.5f}", flush=True)
    trees = state.trees[:best_k] if valid_scores is not None else state.trees
    return trees, state.base_score, history
