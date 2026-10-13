"""
Shared data cleaning and feature engineering for train.py and app.py.

The rule everything here follows: a feature for a match may only use
information that existed BEFORE kick-off. Shots, goals, corners and cards
of a match are its outcome, so they only ever enter the model as part of a
team's *history* (rolling averages updated after the final whistle).

Data: Premier League rows of "Club Football Match Data 2000-2025"
(github.com/xgabora/Club-Football-Match-Data-2000-2025, MIT licence), which
compiles football-data.co.uk results/odds and ClubElo ratings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "data" / "epl_matches.csv"

# Seasons (labelled by the year they start in, Aug -> May)
BURN_IN_SEASON = 2000                 # used only to warm up ratings, never trained on
TRAIN_SEASONS = range(2001, 2019)     # 2001/02 .. 2018/19
VAL_SEASONS = range(2019, 2022)       # 2019/20 .. 2021/22 (includes the empty-stadium season)
TEST_FROM = 2022                      # 2022/23 .. latest rows in the file

TEAM_NAME_FIXES = {"Nottm Forest": "Nott'm Forest"}  # same club spelled two ways in 2024/25

# Per-team statistics tracked as exponentially weighted averages of past matches
TRACKED = {
    "goals": ("FTHome", "FTAway"),
    "sot": ("HomeTarget", "AwayTarget"),      # shots on target
    "shots": ("HomeShots", "AwayShots"),
    "corners": ("HomeCorners", "AwayCorners"),
}


# --------------------------------------------------------------------------
# Loading and cleaning
# --------------------------------------------------------------------------
def load_matches(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, parse_dates=["MatchDate"])
    df = df[df["Division"] == "E0"].copy()
    for col in ("HomeTeam", "AwayTeam"):
        df[col] = df[col].replace(TEAM_NAME_FIXES)

    # Season label: a season runs August -> May (the 2019/20 season was paused
    # and finished in July 2020, which an August cutoff still assigns correctly).
    df["season"] = np.where(df["MatchDate"].dt.month >= 8,
                            df["MatchDate"].dt.year, df["MatchDate"].dt.year - 1)

    # Impossible rows: more shots on target than shots. Total shots is the
    # suspect field (on-target counts are checked against goals), so raise it.
    for h, a in (("HomeShots", "HomeTarget"), ("AwayShots", "AwayTarget")):
        df[h] = np.maximum(df[h], df[a])

    # Bookmaker probabilities (benchmark + betting P&L, NOT model inputs).
    # 1/odds over-sums to ~1.05 (the bookmaker margin): normalise it away.
    inv = 1.0 / df[["OddHome", "OddDraw", "OddAway"]]
    df["book_overround"] = inv.sum(axis=1, min_count=3)
    df["book_p_home"] = inv["OddHome"] / df["book_overround"]

    df["home_win"] = (df["FTResult"] == "H").astype(int)
    df = df.sort_values(["MatchDate", "MatchTime"], kind="stable", na_position="first")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Sequential team ratings (one pass in date order, so only the past is used)
# --------------------------------------------------------------------------
@dataclass
class RatingConfig:
    halflife: float = 10.0       # matches; how fast form averages forget
    elo_k: float = 20.0          # Elo update speed
    elo_home_adv: float = 60.0   # Elo points of home advantage inside the Elo expectation
    elo_init: float = 1500.0


def _ewm(old: float | None, new: float, alpha: float) -> float:
    return new if old is None or np.isnan(old) else old + alpha * (new - old)


def build_match_features(df: pd.DataFrame, cfg: RatingConfig = RatingConfig()) -> pd.DataFrame:
    """Return one row per match with every candidate pre-match feature.

    State per team: Elo rating, EWMA of each tracked stat for and against,
    EWMA of league points, date of last match. Each match reads the state
    first (-> features) and only then updates it with the result.

    Promoted teams have no Premier League history. They inherit the average
    end-of-season state of the teams that were relegated the season before:
    "a promoted side is about as strong as the sides it replaces".
    """
    alpha = 1.0 - 0.5 ** (1.0 / cfg.halflife)
    stat_keys = [f"{k}_{s}" for k in TRACKED for s in ("for", "ag")] + ["pts"]
    state: dict[str, dict] = {}
    rows = []
    prev_season_teams: set[str] = set()
    season_teams: set[str] = set()
    current_season = None

    def blank() -> dict:
        return {"elo": cfg.elo_init, "last": None, **{k: None for k in stat_keys}}

    for r in df.itertuples(index=False):
        if r.season != current_season:                       # season boundary
            if current_season is not None:
                prev_season_teams = season_teams
            new_teams = set(df.loc[df["season"] == r.season, "HomeTeam"]) | \
                set(df.loc[df["season"] == r.season, "AwayTeam"])
            relegated = [t for t in prev_season_teams - new_teams if t in state]
            promoted = [t for t in new_teams - prev_season_teams] if prev_season_teams else []
            if relegated:
                template = {k: np.nanmean([np.nan if state[t][k] is None else state[t][k]
                                           for t in relegated]) for k in ["elo"] + stat_keys}
                for t in promoted:
                    state[t] = {**blank(), **template}
            season_teams = new_teams
            current_season = r.season
            promoted_now = set(promoted)

        h, a = r.HomeTeam, r.AwayTeam
        sh, sa = state.setdefault(h, blank()), state.setdefault(a, blank())

        feat = {
            "elo_home": sh["elo"], "elo_away": sa["elo"],
            "rest_home": np.nan if sh["last"] is None else (r.MatchDate - sh["last"]).days,
            "rest_away": np.nan if sa["last"] is None else (r.MatchDate - sa["last"]).days,
            "promoted_home": float(h in promoted_now), "promoted_away": float(a in promoted_now),
        }
        for k in stat_keys:
            feat[f"{k}_home"] = sh[k]
            feat[f"{k}_away"] = sa[k]
        rows.append(feat)

        # ---- update AFTER the match (this is where the outcome is used) ----
        result = {"H": 1.0, "D": 0.5, "A": 0.0}[r.FTResult]
        expected = 1.0 / (1.0 + 10 ** ((sa["elo"] - sh["elo"] - cfg.elo_home_adv) / 400.0))
        margin = abs(r.FTHome - r.FTAway)
        mult = np.log1p(margin) if margin > 0 else 1.0       # bigger wins move ratings more
        delta = cfg.elo_k * mult * (result - expected)
        sh["elo"] += delta
        sa["elo"] -= delta
        for key, (hcol, acol) in TRACKED.items():
            hv, av = getattr(r, hcol), getattr(r, acol)
            sh[f"{key}_for"] = _ewm(sh[f"{key}_for"], hv, alpha)
            sh[f"{key}_ag"] = _ewm(sh[f"{key}_ag"], av, alpha)
            sa[f"{key}_for"] = _ewm(sa[f"{key}_for"], av, alpha)
            sa[f"{key}_ag"] = _ewm(sa[f"{key}_ag"], hv, alpha)
        sh["pts"] = _ewm(sh["pts"], {"H": 3, "D": 1, "A": 0}[r.FTResult], alpha)
        sa["pts"] = _ewm(sa["pts"], {"A": 3, "D": 1, "H": 0}[r.FTResult], alpha)
        sh["last"] = sa["last"] = r.MatchDate

    raw = pd.DataFrame(rows, index=df.index)
    return add_differences(raw)


def add_differences(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn home/away team states into the matchup features the model sees.

    Each stat becomes ONE number: (home team's net value) - (away team's net
    value), where net = for - against. One coefficient per idea keeps the
    model small and stops it from learning separate, noisy weights for
    'home attack' and 'away defence' that measure the same thing.
    """
    out = pd.DataFrame(index=raw.index)
    out["elo_diff"] = raw["elo_home"] - raw["elo_away"]
    for key in TRACKED:
        net_h = raw[f"{key}_for_home"] - raw[f"{key}_ag_home"]
        net_a = raw[f"{key}_for_away"] - raw[f"{key}_ag_away"]
        out[f"{key}_net_diff"] = net_h - net_a
    out["form_pts_diff"] = raw["pts_home"] - raw["pts_away"]
    # Rest: 2-10 days matters (midweek fixtures); season breaks don't.
    out["rest_diff"] = raw["rest_home"].clip(2, 10) - raw["rest_away"].clip(2, 10)
    out["promoted_diff"] = raw["promoted_home"] - raw["promoted_away"]
    return out


# Feature groups the selection step chooses from (see train.py)
FEATURE_GROUPS = {
    "elo": ["elo_diff"],
    "shots_on_target": ["sot_net_diff"],
    "goals": ["goals_net_diff"],
    "shots": ["shots_net_diff"],
    "corners": ["corners_net_diff"],
    "points_form": ["form_pts_diff"],
    "rest": ["rest_diff"],
    "promoted": ["promoted_diff"],
}


def split_labels(seasons: pd.Series) -> np.ndarray:
    s = seasons.to_numpy()
    return np.select(
        [s == BURN_IN_SEASON, np.isin(s, list(TRAIN_SEASONS)), np.isin(s, list(VAL_SEASONS)), s >= TEST_FROM],
        ["burn_in", "train", "val", "test"], default="unused")


@dataclass
class Preprocessor:
    """Selected columns -> fill missing -> standardise. Fitted on the fitting window only."""
    columns: list[str] = field(default_factory=list)
    medians: dict = field(default_factory=dict)
    scaler: StandardScaler | None = None

    def fit(self, feats: pd.DataFrame, columns: list[str]) -> "Preprocessor":
        self.columns = list(columns)
        self.medians = feats[self.columns].median().to_dict()
        self.scaler = StandardScaler().fit(self._fill(feats))
        return self

    def _fill(self, feats: pd.DataFrame) -> np.ndarray:
        return feats[self.columns].fillna(self.medians).to_numpy(dtype=np.float64)

    def transform(self, feats: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(self._fill(feats))
