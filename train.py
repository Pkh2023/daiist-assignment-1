"""
Assignment 1 - training pipeline: Premier League home-win probability.

Run with:  uv run python main.py train

Decision supported (see REPORT.md): a betting/trading desk decides, before
kick-off, which home-win bets to place at the bookmaker's odds. The model
outputs P(home win); the bookmaker's odds are the benchmark and set the
profit or loss of every bet.

Steps (all automatic):
  1. Load and clean data/epl_matches.csv; build pre-match features in one
     chronological pass (team ratings only ever see the past).
  2. Choose the rating settings and the indicators with walk-forward
     validation: each season 2012/13..2021/22 is predicted by a model fitted
     on all earlier seasons. Only indicators that help consistently are kept.
  3. Split by season:  train 2001/02-2018/19 | val 2019/20-2021/22 | test 2022/23-latest.
     Tune C (sklearn), the learning rate (PyTorch) and the betting threshold on val.
  4. Refit on train+val and train the SAME logistic regression three ways:
       a) sklearn LogisticRegression (lbfgs)
       b) manual PyTorch: raw tensors, autograd, hand-written gradient step
       c) standard PyTorch: nn.Module + torch.optim.SGD
  5. Score the untouched test seasons against a base-rate baseline and the
     bookmaker; save models + artifacts for app.py.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from features import (
    FEATURE_GROUPS,
    TRAIN_SEASONS,
    Preprocessor,
    RatingConfig,
    build_match_features,
    load_matches,
    split_labels,
)

warnings.filterwarnings("ignore", message="Mean of empty slice")  # promoted-team template on day one

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"
SEED = 42

HALFLIFE_GRID = [5, 10, 20]       # matches of memory for rolling team stats
ELO_K_GRID = [15, 20, 30]         # Elo update speed
CV_SEASONS = list(range(2012, 2022))  # walk-forward folds (train + val seasons only)
MIN_GAIN = 0.0005                 # a group must cut mean log loss by this much...
MIN_FOLDS = 7                     # ...and help in at least 7 of the 10 seasons
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
LR_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]
EPOCHS = 3000
THRESHOLD_GRID = np.round(np.arange(0.40, 0.91, 0.01), 2)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def prob_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "accuracy_at_0.5": float(accuracy_score(y, p > 0.5)),
        "mean_predicted": float(p.mean()),
        "actual_rate": float(y.mean()),
    }


def betting(y: np.ndarray, p: np.ndarray, odds: np.ndarray, threshold: float) -> dict:
    """Stake 1 unit on the home win whenever p > threshold (and odds exist).

    This IS the classification decision: predicted positive = bet placed.
      TP = winning bet  (+ odds - 1)      FP = losing bet (-1)
      FN = home win we skipped (0)        TN = correctly skipped (0)
    """
    has_odds = ~np.isnan(odds)
    bet = (p > threshold) & has_odds
    win = y.astype(bool)
    pnl = np.where(bet & win, odds - 1.0, 0.0) - np.where(bet & ~win, 1.0, 0.0)
    n = int(bet.sum())
    return {
        "threshold": float(threshold), "bets": n,
        "tp": int((bet & win).sum()), "fp": int((bet & ~win).sum()),
        "fn": int((~bet & win & has_odds).sum()), "tn": int((~bet & ~win & has_odds).sum()),
        "pnl_units": float(pnl.sum()), "roi": float(pnl.sum() / n) if n else 0.0,
        "hit_rate": float(win[bet].mean()) if n else float("nan"),
    }


# --------------------------------------------------------------------------
# The three implementations of ONE model: L2-regularised logistic regression
#   sklearn minimises  C * sum_i logloss_i + 0.5 * ||w||^2
#   divide by C*n  ->  mean_i logloss_i + ||w||^2 / (2*C*n)     (same minimiser)
# The PyTorch versions minimise the second form. Bias is not penalised.
# --------------------------------------------------------------------------
def fit_sklearn(X: np.ndarray, y: np.ndarray, C: float) -> LogisticRegression:
    return LogisticRegression(C=C, tol=1e-10, max_iter=10_000).fit(X, y)


def objective(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, C: float) -> float:
    z = X @ w + b
    mean_ll = np.mean(np.logaddexp(0.0, z) - y * z)   # numerically stable log loss
    return float(mean_ll + (w @ w) / (2 * C * len(y)))


def stability_limit(X: np.ndarray, C: float) -> float:
    """Upper bound L on the Hessian of the objective (logistic curvature <= 1/4).
    Gradient descent with lr <= 1/L decreases the loss at every step."""
    n = len(X)
    Xb = np.hstack([X, np.ones((n, 1))])
    H = 0.25 * Xb.T @ Xb / n
    H[:-1, :-1] += np.eye(X.shape[1]) / (C * n)
    return float(np.linalg.eigvalsh(H).max())


def fit_manual_torch(X, y, C, lr, epochs=EPOCHS):
    """Session-5 style: raw tensors, autograd, manual update, manual zero_grad."""
    torch.manual_seed(SEED)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    n, p = Xt.shape
    w = torch.zeros(p, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    history = []
    for _ in range(epochs):
        z = Xt @ w + b
        # log loss written out: softplus(z) - y*z  ==  -[y log s(z) + (1-y) log(1-s(z))]
        loss = torch.mean(torch.nn.functional.softplus(z) - yt * z) + torch.sum(w**2) / (2 * C * n)
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
        history.append(loss.item())
    return {"w": w.detach().clone(), "b": b.detach().clone()}, history


def predict_manual_torch(params, X):
    with torch.no_grad():
        z = torch.tensor(X, dtype=torch.float32) @ params["w"] + params["b"]
        return torch.sigmoid(z).squeeze(1).numpy().astype(np.float64)


class LogisticRegressionNet(torch.nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):          # returns logits; the loss applies the sigmoid
        return self.linear(x)


def fit_standard_torch(X, y, C, lr, epochs=EPOCHS):
    torch.manual_seed(SEED)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    n, p = Xt.shape
    model = LogisticRegressionNet(p)
    torch.nn.init.zeros_(model.linear.weight)   # same start as the manual loop
    torch.nn.init.zeros_(model.linear.bias)
    loss_fn = torch.nn.BCEWithLogitsLoss()      # sigmoid + log loss in one stable op
    # weight_decay wd adds wd*w to the gradient = gradient of (wd/2)||w||^2,
    # so wd = 1/(C*n) reproduces ||w||^2/(2Cn); bias group gets none.
    optimizer = torch.optim.SGD([
        {"params": [model.linear.weight], "weight_decay": 1.0 / (C * n)},
        {"params": [model.linear.bias], "weight_decay": 0.0},
    ], lr=lr)
    history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        bce = loss_fn(model(Xt), yt)
        bce.backward()
        optimizer.step()
        with torch.no_grad():
            history.append(bce.item() + float(model.linear.weight.pow(2).sum()) / (2 * C * n))
    return model, history


def predict_standard_torch(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).squeeze(1).numpy().astype(np.float64)


# --------------------------------------------------------------------------
# Step 2: indicator selection by walk-forward validation
# --------------------------------------------------------------------------
def walk_forward(feats: pd.DataFrame, y: np.ndarray, seasons: np.ndarray, cols: list[str]):
    """Mean log loss over CV_SEASONS, each predicted from all earlier seasons."""
    per_fold = []
    for s in CV_SEASONS:
        tr = (seasons >= min(TRAIN_SEASONS)) & (seasons < s)
        te = seasons == s
        if cols:
            prep = Preprocessor().fit(feats[tr], cols)
            m = LogisticRegression(max_iter=2000).fit(prep.transform(feats[tr]), y[tr])
            p = m.predict_proba(prep.transform(feats[te]))[:, 1]
        else:
            p = np.full(te.sum(), y[tr].mean())
        per_fold.append(log_loss(y[te], p, labels=[0, 1]))
    return float(np.mean(per_fold)), np.array(per_fold)


def select_indicators(df: pd.DataFrame) -> tuple[RatingConfig, list[str], dict]:
    y, seasons = df["home_win"].to_numpy(), df["season"].to_numpy()
    all_cols = [c for cols in FEATURE_GROUPS.values() for c in cols]

    # (a) rating settings, judged with every candidate indicator in the model
    grid_rows, built = [], {}
    for hl in HALFLIFE_GRID:
        for k in ELO_K_GRID:
            feats = build_match_features(df, RatingConfig(halflife=hl, elo_k=k))
            built[(hl, k)] = feats
            grid_rows.append({"halflife_matches": hl, "elo_k": k,
                              "cv_log_loss_all_indicators": walk_forward(feats, y, seasons, all_cols)[0]})
    grid = pd.DataFrame(grid_rows)
    best = grid.loc[grid["cv_log_loss_all_indicators"].idxmin()]
    cfg = RatingConfig(halflife=float(best["halflife_matches"]), elo_k=float(best["elo_k"]))
    feats = built[(int(best["halflife_matches"]), int(best["elo_k"]))]

    # (b) greedy forward selection of indicator groups
    chosen: list[str] = []
    chosen_groups: list[str] = []
    cur, cur_folds = walk_forward(feats, y, seasons, [])
    path = [{"step": 0, "indicator_added": "(none: base rate)", "cv_log_loss": cur,
             "gain": None, "seasons_improved": None, "kept": True}]
    remaining = dict(FEATURE_GROUPS)
    step = 0
    while remaining:
        step += 1
        trials = {g: walk_forward(feats, y, seasons, chosen + cols) for g, cols in remaining.items()}
        for g, (m, folds) in sorted(trials.items(), key=lambda kv: kv[1][0]):
            path.append({"step": step, "indicator_added": g, "cv_log_loss": m, "gain": cur - m,
                         "seasons_improved": int((folds < cur_folds).sum()), "kept": False})
        g_best = min(trials, key=lambda g: trials[g][0])
        m, folds = trials[g_best]
        if cur - m < MIN_GAIN or (folds < cur_folds).sum() < MIN_FOLDS:
            break
        path[-len(trials):] = [dict(r, kept=(r["indicator_added"] == g_best)) for r in path[-len(trials):]]
        chosen += remaining.pop(g_best)
        chosen_groups.append(g_best)
        cur, cur_folds = m, folds

    # (c) for the report: what each indicator ALONE achieves, and leave-one-out from all
    alone = [{"indicator": g, "cv_log_loss_alone": walk_forward(feats, y, seasons, cols)[0]}
             for g, cols in FEATURE_GROUPS.items()]
    full = walk_forward(feats, y, seasons, all_cols)[0]
    loo = [{"indicator_removed": g,
            "cv_log_loss_all_others": walk_forward(feats, y, seasons, [c for c in all_cols if c not in cols])[0]}
           for g, cols in FEATURE_GROUPS.items()]

    pd.DataFrame(path).to_csv(ARTIFACTS / "selection_forward_path.csv", index=False)
    grid.to_csv(ARTIFACTS / "selection_rating_settings.csv", index=False)
    pd.DataFrame(alone).sort_values("cv_log_loss_alone").to_csv(ARTIFACTS / "selection_each_alone.csv", index=False)
    pd.DataFrame(loo).to_csv(ARTIFACTS / "selection_leave_one_out.csv", index=False)
    info = {"halflife_matches": cfg.halflife, "elo_k": cfg.elo_k, "chosen_groups": chosen_groups,
            "chosen_columns": chosen, "cv_log_loss_chosen": cur, "cv_log_loss_all_indicators": full}
    return cfg, chosen, info


# --------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    MODELS.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)

    df = load_matches()
    split = split_labels(df["season"])
    for name in ("burn_in", "train", "val", "test"):
        s = df.loc[split == name, "season"]
        print(f"{name:7s}: {int((split == name).sum()):5d} matches, seasons {s.min()}/{(s.min() + 1) % 100:02d}"
              f" .. {s.max()}/{(s.max() + 1) % 100:02d}")

    # ---- 2. indicator selection ---------------------------------------------
    cfg, cols, sel_info = select_indicators(df)
    print(f"\nRating settings: halflife={cfg.halflife:g} matches, Elo K={cfg.elo_k:g}")
    print(pd.read_csv(ARTIFACTS / "selection_forward_path.csv").round(4).to_string(index=False))
    print(f"-> indicators kept: {sel_info['chosen_groups']}")

    feats = build_match_features(df, cfg)
    y = df["home_win"].to_numpy()
    odds = df["OddHome"].to_numpy(dtype=float)
    book = df["book_p_home"].to_numpy(dtype=float)
    tr, va, te = split == "train", split == "val", split == "test"

    # ---- 3. tuning on train -> val -------------------------------------------
    prep_tv = Preprocessor().fit(feats[tr], cols)
    X_tr, X_va = prep_tv.transform(feats[tr]), prep_tv.transform(feats[va])
    c_rows = []
    for C in C_GRID:
        m = fit_sklearn(X_tr, y[tr], C)
        c_rows.append({"C": C, **prob_metrics(y[va], m.predict_proba(X_va)[:, 1])})
    c_df = pd.DataFrame(c_rows)
    c_df.to_csv(ARTIFACTS / "C_tuning_val.csv", index=False)
    best_C = float(c_df.loc[c_df["log_loss"].idxmin(), "C"])
    print("\nRegularisation C on val:")
    print(c_df[["C", "log_loss", "brier", "auc"]].round(5).to_string(index=False))
    print(f"-> C = {best_C}")

    sk_tv = fit_sklearn(X_tr, y[tr], best_C)
    target_obj = objective(X_tr, y[tr], sk_tv.coef_[0], float(sk_tv.intercept_[0]), best_C)

    # final design matrix (needed now for the learning-rate safety margin)
    fit_mask = tr | va
    prep = Preprocessor().fit(feats[fit_mask], cols)
    X_fit, X_te = prep.transform(feats[fit_mask]), prep.transform(feats[te])
    y_fit = y[fit_mask]
    L = max(stability_limit(X_tr, best_C), stability_limit(X_fit, best_C))

    def steps_to(hist):
        h = np.asarray(hist)
        ok = np.where(np.isfinite(h) & (h <= target_obj * (1 + 1e-6)))[0]
        return float(ok[0] + 1) if len(ok) else np.inf

    lr_rows = []
    for lr in LR_GRID:
        _, hm = fit_manual_torch(X_tr, y[tr], best_C, lr)
        _, hs = fit_standard_torch(X_tr, y[tr], best_C, lr)
        lr_rows.append({"lr": lr, "manual_steps_to_converge": steps_to(hm),
                        "standard_steps_to_converge": steps_to(hs),
                        "manual_final_objective": hm[-1], "safe_lr_le_1_over_L": lr <= 1.0 / L})
    lr_df = pd.DataFrame(lr_rows)
    lr_df.to_csv(ARTIFACTS / "lr_tuning.csv", index=False)
    ok = lr_df[lr_df["safe_lr_le_1_over_L"] & np.isfinite(lr_df["manual_steps_to_converge"])]
    if ok.empty:
        raise RuntimeError("No safe learning rate converged - raise EPOCHS or change LR_GRID.")
    best_lr = float(ok.loc[ok["manual_steps_to_converge"].idxmin(), "lr"])
    print(f"\nLearning rate (curvature bound L={L:.4f}, safe lr <= 1/L = {1 / L:.3f}; "
          f"closed-form objective {target_obj:.6f}):")
    print(lr_df.round(6).to_string(index=False))
    print(f"-> lr = {best_lr}")

    # betting threshold, chosen on val with the train-only model
    p_va = sk_tv.predict_proba(X_va)[:, 1]
    thr_rows = [betting(y[va], p_va, odds[va], t) for t in THRESHOLD_GRID]
    thr_df = pd.DataFrame(thr_rows)
    thr_df.to_csv(ARTIFACTS / "threshold_val.csv", index=False)
    playable = thr_df[thr_df["bets"] >= 30]          # ignore thresholds with almost no bets
    best_t = float(playable.loc[playable["pnl_units"].idxmax(), "threshold"])
    print(f"\nBetting threshold on val (max P&L, >=30 bets): {best_t} "
          f"-> {playable.loc[playable['pnl_units'].idxmax(), 'pnl_units']:.1f} units")

    # ---- 4. final fit on train+val: three implementations ---------------------
    sk = fit_sklearn(X_fit, y_fit, best_C)
    manual, hist_manual = fit_manual_torch(X_fit, y_fit, best_C, best_lr)
    net, hist_std = fit_standard_torch(X_fit, y_fit, best_C, best_lr)
    obj_sk = objective(X_fit, y_fit, sk.coef_[0], float(sk.intercept_[0]), best_C)
    for name, h in (("manual", hist_manual), ("standard", hist_std)):
        if not h[-1] <= obj_sk * (1 + 1e-5):
            raise RuntimeError(f"{name} PyTorch did not reach the optimum: {h[-1]:.7f} vs {obj_sk:.7f}")

    names = ["scikit-learn LogisticRegression", "Manual PyTorch loop", "Standard PyTorch (nn.Module + optim)"]
    p_all = {
        names[0]: sk.predict_proba(prep.transform(feats))[:, 1],
        names[1]: predict_manual_torch(manual, prep.transform(feats)),
        names[2]: predict_standard_torch(net, prep.transform(feats)),
    }

    # ---- 5. test evaluation ---------------------------------------------------
    results = {}
    for n in names:
        results[n] = {**prob_metrics(y[te], p_all[n][te]),
                      "bet_at_val_threshold": betting(y[te], p_all[n][te], odds[te], best_t)}
    base_rate = float(y_fit.mean())
    results["Baseline: home-win rate of train+val"] = prob_metrics(y[te], np.full(te.sum(), base_rate))
    has_book = te & ~np.isnan(book)
    results["Benchmark: bookmaker (Bet365, margin removed)"] = {
        **prob_metrics(y[has_book], book[has_book]),
        "bet_at_val_threshold": betting(y[te], np.nan_to_num(book[te], nan=0.0), odds[te], best_t)}

    w_sk = sk.coef_[0]
    w_man = manual["w"].squeeze(1).numpy().astype(np.float64)
    w_std = net.linear.weight.detach().squeeze(0).numpy().astype(np.float64)
    agreement = {
        "coefficients": {c: {"sklearn": float(a), "manual": float(b), "standard": float(d)}
                         for c, a, b, d in zip(cols, w_sk, w_man, w_std)},
        "intercepts": {"sklearn": float(sk.intercept_[0]), "manual": float(manual["b"]),
                       "standard": float(net.linear.bias.detach())},
        "max_abs_coef_diff_manual_vs_sklearn": float(np.max(np.abs(w_man - w_sk))),
        "max_abs_coef_diff_standard_vs_sklearn": float(np.max(np.abs(w_std - w_sk))),
        "max_abs_prob_diff_manual_vs_sklearn": float(np.max(np.abs(p_all[names[1]] - p_all[names[0]])[te])),
        "max_abs_prob_diff_standard_vs_sklearn": float(np.max(np.abs(p_all[names[2]] - p_all[names[0]])[te])),
        "max_abs_prob_diff_manual_vs_standard": float(np.max(np.abs(p_all[names[1]] - p_all[names[2]])[te])),
        "final_objective": {"sklearn": obj_sk, "manual": hist_manual[-1], "standard": hist_std[-1]},
    }

    # Is the betting result real or luck? Bootstrap the test bets at the
    # val-chosen threshold, and check the "value bet" rule p*odds > 1+edge.
    rng = np.random.default_rng(SEED)
    p_te, y_te, o_te = p_all[names[0]][te], y[te], odds[te]
    bet = (p_te > best_t) & ~np.isnan(o_te)
    pnl = np.where(y_te[bet] == 1, o_te[bet] - 1.0, -1.0)
    boot = np.array([rng.choice(pnl, len(pnl)).sum() for _ in range(5000)])
    edge_rows = []
    for split_name, mask in (("val", va), ("test", te)):
        pm = sk_tv.predict_proba(prep_tv.transform(feats[mask]))[:, 1] if split_name == "val" \
            else p_all[names[0]][mask]
        om, ym = odds[mask], y[mask]
        for edge in (0.0, 0.02, 0.05, 0.10):
            b = (pm * om > 1 + edge) & ~np.isnan(om)
            pl = np.where(ym[b] == 1, om[b] - 1.0, -1.0)
            edge_rows.append({"split": split_name, "edge": edge, "bets": int(b.sum()),
                              "pnl_units": float(pl.sum()), "roi": float(pl.mean()) if b.any() else 0.0})
    pd.DataFrame(edge_rows).to_csv(ARTIFACTS / "value_bet_rule.csv", index=False)
    betting_check = {"test_bets": int(bet.sum()), "test_pnl": float(pnl.sum()),
                     "bootstrap_95ci_pnl": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                     "bootstrap_share_positive": float((boot > 0).mean())}
    print("\nBetting check:", betting_check)
    print(pd.DataFrame(edge_rows).round(3).to_string(index=False))

    print("\nTEST (2022/23 onwards):")
    flat = {k: {m: v for m, v in r.items() if not isinstance(v, dict)} for k, r in results.items()}
    print(pd.DataFrame(flat).T.round(4).to_string())
    print("\nBetting at the val-chosen threshold, test seasons:")
    print(pd.DataFrame({k: r["bet_at_val_threshold"] for k, r in results.items()
                        if "bet_at_val_threshold" in r}).T.round(3).to_string())
    print("\nAgreement:", json.dumps({k: v for k, v in agreement.items() if k != "coefficients"}, indent=2))

    # per-season calibration: does the model's average P(home win) track reality?
    by_season = pd.DataFrame({"season": df["season"], "split": split, "home_win": y,
                              "model_p": p_all[names[0]], "book_p": book}).query("split != 'burn_in'") \
        .groupby(["season", "split"], as_index=False).agg(
            matches=("home_win", "size"), actual_home_win_rate=("home_win", "mean"),
            model_mean_p=("model_p", "mean"), book_mean_p=("book_p", "mean"))
    by_season.to_csv(ARTIFACTS / "by_season.csv", index=False)

    # ---- save -----------------------------------------------------------------
    joblib.dump({"preprocessor": prep, "rating_config": cfg}, MODELS / "preprocessor.joblib")
    joblib.dump(sk, MODELS / "sklearn_logreg.joblib")
    torch.save(manual, MODELS / "manual_torch_params.pt")
    torch.save(net.state_dict(), MODELS / "standard_torch_state.pt")

    pd.DataFrame({"epoch": np.arange(1, EPOCHS + 1), "manual_torch": hist_manual,
                  "standard_torch": hist_std}).to_csv(ARTIFACTS / "loss_curves.csv", index=False)
    keep = ["MatchDate", "season", "HomeTeam", "AwayTeam", "FTHome", "FTAway", "FTResult",
            "OddHome", "OddDraw", "OddAway", "book_p_home", "home_win"]
    out = df[keep].copy()
    out["split"] = split
    out = pd.concat([out, feats], axis=1)  # every candidate indicator, for the dashboard
    out.to_csv(ARTIFACTS / "matches_with_features.csv", index=False)

    meta = {
        "split": {k: {"matches": int((split == k).sum()),
                      "first_season": int(df.loc[split == k, "season"].min()),
                      "last_season": int(df.loc[split == k, "season"].max())}
                  for k in ("burn_in", "train", "val", "test")},
        "selection": sel_info,
        "C": best_C, "learning_rate": best_lr, "curvature_bound_L": L, "epochs": EPOCHS,
        "betting_threshold_from_val": best_t,
        "test_results": results, "agreement": agreement, "betting_check": betting_check,
        "data_quality": {
            "team_name_fixed": "Nottm Forest -> Nott'm Forest",
            "seasons_with_missing_matches": {str(s): int(n) for s, n in
                                             df.groupby("season").size().items() if n != 380 and s < 2026},
            "matches_without_odds": int(np.isnan(odds).sum()),
        },
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved models/ and artifacts/ in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
