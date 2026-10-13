"""
Assignment 1 - Gradio dashboard: Premier League home-win model vs the bookmaker.

Run with:  uv run python main.py app      (after  uv run python main.py train)

Nothing is trained here. At startup the app loads what train.py saved in
models/ (preprocessor, sklearn model, manual-PyTorch weights, nn.Module state
dict), re-scores every match with all three, and reads the tables in
artifacts/ for display.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

from features import FEATURE_GROUPS
from train import (
    LogisticRegressionNet,
    betting,
    predict_manual_torch,
    predict_standard_torch,
    prob_metrics,
)

ROOT = Path(__file__).resolve().parent
MODELS, ARTIFACTS = ROOT / "models", ROOT / "artifacts"

SK, MAN, STD = "scikit-learn LogisticRegression", "Manual PyTorch loop", "Standard PyTorch (nn.Module + optim)"
BOOK = "Bookmaker (Bet365, margin removed)"
MODEL_NAMES = [SK, MAN, STD]
COLORS = {SK: "#2a78d6", MAN: "#eb6834", STD: "#1baf7a", BOOK: "#6b6a65"}
OUTCOME_COLORS = {"Home win": "#2a78d6", "Draw or away win": "#eb6834"}
LAYOUT = dict(template="plotly_white", margin=dict(l=55, r=20, t=55, b=45),
              legend=dict(orientation="h", y=-0.2))


# --------------------------------------------------------------------------
# Load trained artifacts (never retrain)
# --------------------------------------------------------------------------
def load():
    need = ["preprocessor.joblib", "sklearn_logreg.joblib", "manual_torch_params.pt", "standard_torch_state.pt"]
    missing = [f for f in need if not (MODELS / f).exists()]
    if missing:
        raise SystemExit(f"Missing {missing} in models/ - run `uv run python main.py train` first.")
    bundle = joblib.load(MODELS / "preprocessor.joblib")
    prep = bundle["preprocessor"]
    sk = joblib.load(MODELS / "sklearn_logreg.joblib")
    manual = torch.load(MODELS / "manual_torch_params.pt")
    net = LogisticRegressionNet(len(prep.columns))
    net.load_state_dict(torch.load(MODELS / "standard_torch_state.pt"))
    net.eval()
    meta = json.loads((ARTIFACTS / "metrics.json").read_text())
    matches = pd.read_csv(ARTIFACTS / "matches_with_features.csv", parse_dates=["MatchDate"])
    return prep, sk, manual, net, meta, matches


PREP, SK_MODEL, MANUAL, NET, META, M = load()
COLS = PREP.columns
THRESHOLD_DEFAULT = META["betting_threshold_from_val"]


def predict_all(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    X = PREP.transform(frame)
    return {SK: SK_MODEL.predict_proba(X)[:, 1],
            MAN: predict_manual_torch(MANUAL, X),
            STD: predict_standard_torch(NET, X)}


for name, p in predict_all(M).items():   # inference only
    M[name] = p
M[BOOK] = M["book_p_home"]
M["Outcome"] = np.where(M["home_win"] == 1, "Home win", "Draw or away win")
M["label"] = (M["MatchDate"].dt.strftime("%Y-%m-%d") + "  " + M["HomeTeam"] + " v " + M["AwayTeam"]
              + "  (" + M["FTHome"].astype(int).astype(str) + "-" + M["FTAway"].astype(int).astype(str) + ")")
SPLITS = {"Test (2022/23 onwards, never seen in training)": "test",
          "Validation 2019/20-2021/22 (in-sample for the final refit)": "val"}


def subset(split_label: str) -> pd.DataFrame:
    return M[M["split"] == SPLITS[split_label]]


# --------------------------------------------------------------------------
# Tab: compare models
# --------------------------------------------------------------------------
def metrics_table(split_label: str) -> pd.DataFrame:
    d = subset(split_label)
    rows = []
    base = float(M.loc[M["split"].isin(["train", "val"]), "home_win"].mean())
    for name in MODEL_NAMES + [BOOK]:
        ok = d[name].notna()
        m = prob_metrics(d.loc[ok, "home_win"].to_numpy(), d.loc[ok, name].to_numpy())
        rows.append({"Model": name, "Log loss ↓": m["log_loss"], "Brier ↓": m["brier"], "AUC ↑": m["auc"],
                     "Accuracy @0.5": m["accuracy_at_0.5"], "Mean P(home)": m["mean_predicted"]})
    m = prob_metrics(d["home_win"].to_numpy(), np.full(len(d), base))
    rows.append({"Model": "Baseline: constant home-win rate", "Log loss ↓": m["log_loss"], "Brier ↓": m["brier"],
                 "AUC ↑": m["auc"], "Accuracy @0.5": m["accuracy_at_0.5"], "Mean P(home)": m["mean_predicted"]})
    out = pd.DataFrame(rows)
    out.insert(1, "Matches", len(d))
    return out.round(4)


def agreement_table() -> pd.DataFrame:
    a = META["agreement"]
    rows = [{"Parameter": c, "sklearn": v["sklearn"], "Manual PyTorch": v["manual"], "Standard PyTorch": v["standard"]}
            for c, v in a["coefficients"].items()]
    rows.append({"Parameter": "intercept", "sklearn": a["intercepts"]["sklearn"],
                 "Manual PyTorch": a["intercepts"]["manual"], "Standard PyTorch": a["intercepts"]["standard"]})
    rows.append({"Parameter": "final objective", **{k2: a["final_objective"][k1] for k1, k2 in
                 (("sklearn", "sklearn"), ("manual", "Manual PyTorch"), ("standard", "Standard PyTorch"))}})
    t = pd.DataFrame(rows)
    for c in ("sklearn", "Manual PyTorch", "Standard PyTorch"):
        t[c] = t[c].map(lambda v: f"{v:.7f}")
    return t


def calibration_plot(split_label: str, models: list[str]) -> go.Figure:
    """Predicted probability vs what actually happened (the classification
    version of a prediction-vs-actual plot)."""
    d = subset(split_label)
    bins = np.linspace(0, 1, 11)
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfectly calibrated",
                    line=dict(color="#b0afa9", dash="dash", width=1), hoverinfo="skip")
    symbols = {SK: "circle", MAN: "x", STD: "triangle-up", BOOK: "diamond"}
    for name in models:
        ok = d[name].notna()
        b = pd.cut(d.loc[ok, name], bins, include_lowest=True)
        g = d[ok].groupby(b, observed=True).agg(p=(name, "mean"), actual=("home_win", "mean"), n=("home_win", "size"))
        g = g[g["n"] >= 10]
        fig.add_scatter(x=g["p"], y=g["actual"], mode="lines+markers", name=name,
                        marker=dict(size=9, symbol=symbols[name], color=COLORS[name]),
                        line=dict(color=COLORS[name], width=2),
                        customdata=g["n"], hovertemplate="predicted %{x:.2f}<br>actual %{y:.2f}<br>%{customdata} matches")
    fig.update_layout(title="Predicted P(home win) vs actual home-win rate (10% bins, ≥10 matches)",
                      xaxis_title="predicted probability", yaxis_title="actual home-win rate",
                      xaxis_range=[0, 1], yaxis_range=[0, 1], **LAYOUT)
    return fig


def model_vs_book_plot(split_label: str) -> go.Figure:
    d = subset(split_label).dropna(subset=[BOOK])
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="model = bookmaker",
                    line=dict(color="#b0afa9", dash="dash", width=1), hoverinfo="skip")
    for outcome, color in OUTCOME_COLORS.items():
        s = d[d["Outcome"] == outcome]
        fig.add_scatter(x=s[BOOK], y=s[SK], mode="markers", name=outcome,
                        marker=dict(color=color, size=6, opacity=0.55), text=s["label"],
                        hovertemplate="%{text}<br>bookmaker %{x:.2f}<br>model %{y:.2f}<extra></extra>")
    fig.update_layout(title="Our probability vs the bookmaker's, every match", xaxis_title="bookmaker P(home win)",
                      yaxis_title="model P(home win)", **LAYOUT)
    return fig


def season_plot() -> go.Figure:
    s = pd.read_csv(ARTIFACTS / "by_season.csv")
    fig = go.Figure()
    fig.add_bar(x=s["season"], y=s["actual_home_win_rate"], name="actual home-win rate",
                marker_color=[{"train": "#c9d9f2", "val": "#f6c6b0", "test": "#b9e6d3"}[x] for x in s["split"]])
    fig.add_scatter(x=s["season"], y=s["model_mean_p"], name="model's average P(home win)", mode="lines+markers",
                    line=dict(color=COLORS[SK], width=2), marker=dict(size=8))
    fig.add_scatter(x=s["season"], y=s["book_mean_p"], name="bookmaker's average P(home win)", mode="lines+markers",
                    line=dict(color=COLORS[BOOK], width=2, dash="dot"), marker=dict(size=8))
    fig.add_annotation(x=2020, y=0.379, text="2020/21: empty stadiums", showarrow=True, ay=-45)
    fig.update_layout(title="Home advantage by season (bars coloured train / val / test)",
                      xaxis_title="season (start year)", yaxis_title="home-win rate", yaxis_range=[0.3, 0.55], **LAYOUT)
    return fig


# --------------------------------------------------------------------------
# Tab: decision threshold -> confusion matrix -> profit / loss
# --------------------------------------------------------------------------
def threshold_view(threshold: float, model: str, split_label: str):
    d = subset(split_label)
    p = d[model].fillna(0.0).to_numpy()   # no bookmaker price -> never bet
    r = betting(d["home_win"].to_numpy(), p, d["OddHome"].to_numpy(dtype=float), threshold)
    cm = np.array([[r["tp"], r["fn"]], [r["fp"], r["tn"]]])
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Bet on home win (p > t)", "No bet"], y=["Home won", "Home did not win"],
        text=[[f"TP {r['tp']}<br>win +(odds-1)", f"FN {r['fn']}<br>missed, 0"],
              [f"FP {r['fp']}<br>lose −1", f"TN {r['tn']}<br>skipped, 0"]],
        texttemplate="%{text}", colorscale=[[0, "#eef3fb"], [1, "#2a78d6"]], showscale=False))
    fig.update_layout(title=f"Confusion matrix at threshold {threshold:.2f} — {model}",
                      yaxis_autorange="reversed", **{**LAYOUT, "margin": dict(l=130, r=20, t=55, b=45)})
    n_pos = r["tp"] + r["fn"]
    summary = pd.DataFrame([{
        "Bets placed": r["bets"],
        "Hit rate (precision)": round(r["hit_rate"], 3) if r["bets"] else None,
        "Home wins caught (recall)": round(r["tp"] / n_pos, 3) if n_pos else None,
        "Accuracy": round((r["tp"] + r["tn"]) / max(1, cm.sum()), 3),
        "Profit / loss (units, 1 per bet)": round(r["pnl_units"], 2),
        "Return per bet (ROI)": f"{100 * r['roi']:+.1f}%",
    }])
    return fig, summary, pnl_curve(model, split_label, threshold)


def pnl_curve(model: str, split_label: str, threshold: float) -> go.Figure:
    d = subset(split_label)
    ts = np.round(np.arange(0.40, 0.91, 0.01), 2)
    fig = go.Figure()
    for name in [model, BOOK] if model != BOOK else [BOOK]:
        p = d[name].fillna(0.0).to_numpy()
        res = [betting(d["home_win"].to_numpy(), p, d["OddHome"].to_numpy(dtype=float), t) for t in ts]
        fig.add_scatter(x=ts, y=[x["pnl_units"] for x in res], name=f"{name}: profit/loss", mode="lines",
                        line=dict(color=COLORS[name], width=2),
                        customdata=[x["bets"] for x in res], hovertemplate="t=%{x}<br>P&L %{y:.1f}<br>%{customdata} bets")
    fig.add_vline(x=threshold, line_dash="dot", line_color="#2b2b2b")
    fig.add_hline(y=0, line_color="#8a8984", line_width=1)
    fig.update_layout(title="Profit / loss of 'bet 1 unit on the home win when P > threshold'",
                      xaxis_title="threshold", yaxis_title="units won (+) or lost (−)", hovermode="x", **LAYOUT)
    return fig


def value_bet_view(edge: float, split_label: str) -> pd.DataFrame:
    d = subset(split_label).dropna(subset=["OddHome"])
    rows = []
    for name in (SK, BOOK):
        b = d[name] * d["OddHome"] > 1 + edge
        pl = np.where(d.loc[b, "home_win"] == 1, d.loc[b, "OddHome"] - 1, -1.0)
        rows.append({"Rule uses": name, "Bets": int(b.sum()), "Profit / loss": round(float(pl.sum()), 2),
                     "ROI": f"{100 * pl.mean():+.1f}%" if b.any() else "-"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Tab: distributions
# --------------------------------------------------------------------------
ALL_FEATURES = [c for cols in FEATURE_GROUPS.values() for c in cols]
FEATURE_LABELS = {"elo_diff": "Elo difference (home − away)",
                  "sot_net_diff": "Shots-on-target dominance difference",
                  "goals_net_diff": "Goal-difference form difference",
                  "shots_net_diff": "Shots dominance difference",
                  "corners_net_diff": "Corners dominance difference",
                  "form_pts_diff": "Points form difference",
                  "rest_diff": "Rest days difference",
                  "promoted_diff": "Promoted (home − away)"}


def feature_hist(feature: str) -> go.Figure:
    d = M[M["split"].isin(["train", "val", "test"])]
    fig = go.Figure()
    for outcome, color in OUTCOME_COLORS.items():
        fig.add_histogram(x=d.loc[d["Outcome"] == outcome, feature], name=outcome, marker_color=color,
                          opacity=0.55, nbinsx=50, histnorm="probability density")
    used = "used by the model" if feature in COLS else "candidate, NOT selected"
    fig.update_layout(barmode="overlay", title=f"{FEATURE_LABELS.get(feature, feature)} — by outcome ({used})",
                      xaxis_title=feature, yaxis_title="density", **LAYOUT)
    return fig


def prob_hist(split_label: str) -> go.Figure:
    d = subset(split_label)
    fig = go.Figure()
    for outcome, color in OUTCOME_COLORS.items():
        fig.add_histogram(x=d.loc[d["Outcome"] == outcome, SK], name=outcome, marker_color=color,
                          opacity=0.55, nbinsx=30)
    fig.update_layout(barmode="overlay", title="Model's P(home win) split by what actually happened",
                      xaxis_title="predicted probability", yaxis_title="matches", **LAYOUT)
    return fig


def target_plot() -> go.Figure:
    d = M[M["split"].isin(["train", "val", "test"])]
    counts = d.groupby("FTResult").size().reindex(["H", "D", "A"])
    fig = go.Figure(go.Bar(x=["Home win", "Draw", "Away win"], y=counts.to_numpy(),
                           marker_color=["#2a78d6", "#8a8984", "#eb6834"],
                           text=[f"{100 * v / counts.sum():.1f}%" for v in counts], textposition="outside"))
    fig.update_layout(title="Target: home win = 1, draw or away win = 0 (all modelled seasons)",
                      yaxis_title="matches", **LAYOUT)
    return fig


# --------------------------------------------------------------------------
# Tab: training story (read-only artifacts)
# --------------------------------------------------------------------------
def loss_plot() -> go.Figure:
    lc = pd.read_csv(ARTIFACTS / "loss_curves.csv")
    fig = go.Figure()
    fig.add_scatter(x=lc["epoch"], y=lc["manual_torch"], name=MAN, line=dict(color=COLORS[MAN], width=2))
    fig.add_scatter(x=lc["epoch"], y=lc["standard_torch"], name=STD, line=dict(color=COLORS[STD], width=2, dash="dash"))
    fig.add_hline(y=META["agreement"]["final_objective"]["sklearn"], line_dash="dot", line_color=COLORS[SK],
                  annotation_text="sklearn optimum", annotation_position="top right")
    fig.update_layout(title=f"Training objective (lr = {META['learning_rate']}, C = {META['C']})",
                      xaxis_title="full-batch step", yaxis_title="log loss + L2 penalty", xaxis_type="log", **LAYOUT)
    return fig


# --------------------------------------------------------------------------
# Tab: single match and what-if
# --------------------------------------------------------------------------
TEST_LABELS = M.loc[M["split"] == "test"].sort_values("MatchDate", ascending=False)["label"].tolist()


def match_view(label: str) -> pd.DataFrame:
    r = M.loc[M["label"] == label].iloc[0]
    rows = [{"Item": FEATURE_LABELS.get(c, c), "Value": f"{r[c]:.2f}"} for c in COLS]
    for name in MODEL_NAMES + [BOOK]:
        rows.append({"Item": f"P(home win) — {name}", "Value": "-" if pd.isna(r[name]) else f"{r[name]:.3f}"})
    rows.append({"Item": "Bet365 home odds", "Value": "-" if pd.isna(r["OddHome"]) else f"{r['OddHome']:.2f}"})
    rows.append({"Item": "Result", "Value": {"H": "Home win", "D": "Draw", "A": "Away win"}[r["FTResult"]]})
    return pd.DataFrame(rows)


def what_if(*values) -> pd.DataFrame:
    frame = pd.DataFrame([dict(zip(COLS, values))])
    preds = predict_all(frame)
    return pd.DataFrame({"Model": list(preds), "P(home win)": [round(float(v[0]), 4) for v in preds.values()]})


# --------------------------------------------------------------------------
sel = META["selection"]
with gr.Blocks(title="Premier League home-win model") as demo:
    gr.Markdown(
        "# Premier League: P(home win) vs the bookmaker\n"
        f"Logistic regression on **{len(COLS)} pre-match indicators** ({', '.join(sel['chosen_groups'])}), chosen "
        f"by walk-forward validation, trained three ways. Train {META['split']['train']['first_season']}/"
        f"{(META['split']['train']['first_season'] + 1) % 100:02d}–{META['split']['train']['last_season']}/"
        f"{(META['split']['train']['last_season'] + 1) % 100:02d}, val to {META['split']['val']['last_season']}/"
        f"{(META['split']['val']['last_season'] + 1) % 100:02d}, **test {META['split']['test']['first_season']}/"
        f"{(META['split']['test']['first_season'] + 1) % 100:02d} onwards** "
        f"({META['split']['test']['matches']} matches). Models are loaded from `models/`; nothing retrains here."
    )
    split_choice = gr.Radio(list(SPLITS), value=list(SPLITS)[0], label="Evaluate on")

    with gr.Tab("Compare models"):
        m_table = gr.Dataframe(metrics_table(list(SPLITS)[0]), interactive=False, label="Scores (computed live)")
        gr.Markdown("**Do the three implementations agree?** Fitted parameters on the same standardized features:")
        gr.Dataframe(agreement_table(), interactive=False)
        cal_models = gr.CheckboxGroup(MODEL_NAMES + [BOOK], value=MODEL_NAMES + [BOOK], label="Show in calibration plot")
        cal = gr.Plot(calibration_plot(list(SPLITS)[0], MODEL_NAMES + [BOOK]))
        mvb = gr.Plot(model_vs_book_plot(list(SPLITS)[0]))
        gr.Plot(season_plot())

    with gr.Tab("Threshold, confusion matrix & profit"):
        gr.Markdown("Decision rule: **stake 1 unit on the home win when P(home win) > threshold**, at Bet365's odds. "
                    f"Validation picked t = {THRESHOLD_DEFAULT:.2f} (no threshold was profitable on validation).")
        with gr.Row():
            thr = gr.Slider(0.40, 0.90, value=THRESHOLD_DEFAULT, step=0.01, label="Threshold t")
            thr_model = gr.Dropdown(MODEL_NAMES + [BOOK], value=SK, label="Probabilities from")
        with gr.Row():
            cm_plot = gr.Plot()
            pnl_plot = gr.Plot()
        cm_summary = gr.Dataframe(interactive=False)
        gr.Markdown("**Value-bet rule** instead: bet when P × odds > 1 + edge (i.e. we think the odds are too long).")
        edge = gr.Slider(0.0, 0.3, value=0.0, step=0.01, label="Required edge")
        vb = gr.Dataframe(interactive=False)

    with gr.Tab("Distributions"):
        gr.Plot(target_plot())
        feat = gr.Dropdown(ALL_FEATURES, value=COLS[0], label="Indicator (all candidates; the model uses "
                           + ", ".join(COLS) + ")")
        f_plot = gr.Plot(feature_hist(COLS[0]))
        p_hist = gr.Plot(prob_hist(list(SPLITS)[0]))
        feat.change(feature_hist, feat, f_plot)

    with gr.Tab("How indicators were chosen"):
        gr.Markdown(f"Walk-forward validation: seasons 2012/13–2021/22 each predicted from all earlier seasons. "
                    f"An indicator is kept only if it cuts mean log loss by ≥0.0005 **and** helps in ≥7 of 10 seasons. "
                    f"Rating settings chosen: half-life {sel['halflife_matches']:g} matches, Elo K {sel['elo_k']:g}.")
        gr.Dataframe(pd.read_csv(ARTIFACTS / "selection_forward_path.csv").round(4), interactive=False,
                     label="Forward selection path")
        with gr.Row():
            gr.Dataframe(pd.read_csv(ARTIFACTS / "selection_each_alone.csv").round(4), interactive=False,
                         label="Each indicator on its own")
            gr.Dataframe(pd.read_csv(ARTIFACTS / "selection_leave_one_out.csv").round(4), interactive=False,
                         label="All indicators minus one")
        with gr.Row():
            gr.Dataframe(pd.read_csv(ARTIFACTS / "C_tuning_val.csv")[["C", "log_loss", "brier", "auc"]].round(5),
                         interactive=False, label="Regularisation C (validation)")
            gr.Dataframe(pd.read_csv(ARTIFACTS / "lr_tuning.csv").round(6), interactive=False,
                         label=f"Learning rate (safe if ≤ 1/L = {1 / META['curvature_bound_L']:.3f})")
        gr.Plot(loss_plot())

    with gr.Tab("Single match & what-if"):
        pick = gr.Dropdown(TEST_LABELS, value=TEST_LABELS[0], label="Test-season match")
        mv = gr.Dataframe(match_view(TEST_LABELS[0]), interactive=False)
        pick.change(match_view, pick, mv)
        gr.Markdown("What-if: set the indicators yourself (units as in the data).")
        q = M.loc[M["split"] != "burn_in", COLS]
        sliders = [gr.Slider(float(np.floor(q[c].quantile(0.005))), float(np.ceil(q[c].quantile(0.995))),
                             value=0.0, step=0.1 if c != "elo_diff" else 5.0, label=FEATURE_LABELS.get(c, c))
                   for c in COLS]
        wi = gr.Dataframe(what_if(*[0.0] * len(COLS)), interactive=False)
        for s in sliders:
            s.change(what_if, sliders, wi)

    # wiring that depends on the shared split selector
    def refresh_split(split_label, models, t, model, e):
        return (metrics_table(split_label), calibration_plot(split_label, models), model_vs_book_plot(split_label),
                prob_hist(split_label), *threshold_view(t, model, split_label), value_bet_view(e, split_label))

    outputs = [m_table, cal, mvb, p_hist, cm_plot, cm_summary, pnl_plot, vb]
    inputs = [split_choice, cal_models, thr, thr_model, edge]
    split_choice.change(refresh_split, inputs, outputs)
    cal_models.change(calibration_plot, [split_choice, cal_models], cal)
    for ctl in (thr, thr_model):
        ctl.change(threshold_view, [thr, thr_model, split_choice], [cm_plot, cm_summary, pnl_plot])
    edge.change(value_bet_view, [edge, split_choice], vb)
    demo.load(refresh_split, inputs, outputs)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
