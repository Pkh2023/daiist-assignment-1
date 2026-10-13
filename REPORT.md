# Assignment 1 Report

## Dataset

Premier League matches, Aug 2000 – Aug 2026: 9,810 matches × 48 columns, committed as `data/epl_matches.csv` (2.3 MB). It is the E0 (Premier League) subset of [Club Football Match Data 2000-2025](https://github.com/xgabora/Club-Football-Match-Data-2000-2025) (MIT license). The dataset contains results, match statistics and Bet365 odds from football-data.co.uk. One row = one match:

- Before the match: date, home team, away team, odds offered by bookmakers (home/draw/away, over/under 2.5, Asian handicap).
- Goals, shots, shots on target, corners, fouls and cards for each side at the final whistle.
- Columns already made (Elo, last 3/5 games, clusters of "game styles"). I don't use them: I build my own ratings, and the cluster columns are computed from the match's own stats, so they would leak the result.

Why this dataset: there's very little in it that can be used as-is. Each column describing a match is the result of that match. So the only way to predict a match is to engineer the history of each team from previous rows. The whole problem is to get that right (not leak the future). It also has a built-in benchmark: the bookmaker's odds are a professional probability for each match.

What I did, and data quality issues found:

| Problem | Evidence | Decision |
|---|---|---|
| Same club, different name | "Nott'm Forest" (21 games) and "Nottm Forest" (17) in 2024/25 | Merged to one name. Otherwise Forest's ratings would be divided into two "new" teams mid-season. |
| Games missed | 2003/04 and 2004/05 have 335 of 380 games (each team 32–35 games) | Kept the seasons. Ratings just skip the games that aren't there; there is nothing reliable to fill them with. |
| Impossible rows | 2 matches with more shots on target than total shots | Total shots raised to equal shots on target. The model does not take total shots as input, so it doesn't impact predictions. |
| Missing odds | 83 matches in 2001/02 | Only impacts the bookmaker baseline, which is calculated on test, where odds are complete. |
| Goals > shots on target | 49 team games | Own goals (legitimate). Unchanged. |

## Business / real-life framing

Decision supported: prior to each Premier League match, a sports-trading desk decides whether to risk money on a home win at the bookmaker's price. The model's job is to output a calibrated P(home win), and the bookmaker's odds turn every decision into money.

How that framing pushed the pipeline:

1. **Target = home win (1) vs. draw or away win (0).** It is binary, so logistic regression. Home wins represent 45–46% of matches, so the classes are almost balanced. I didn't model draw vs away separately: that would be a three-class model, not what was assigned.
2. **Pre-match information only.** The desk bets before kick-off, so all inputs have to be there before kick-off. Ratings are constructed in a single pass in date order: each game first reads the state of both teams (→ features) and then updates it with the result.
3. **Divided by season, never randomly.** A random split would train on 2024 matches to predict 2019 matches, and the team strength from the future would leak into the past. It would also put two halves of the same season on either side of the split.
   - Burn-in: 2000/01 (ratings warm up, never trained on)
   - Train: 2001/02 – 2018/19 (6,750 games)
   - Validation: 2019/20 – 2021/22 (1,140), used for any tuning decision
   - Test: 2022/23 – Aug 2026 (1,540), scored once

   Validation deliberately includes 2020/21, played in empty stadiums.
4. **Metric: log loss** (lower is better). Log loss penalizes over-confidence. The desk needs probabilities it can compare to odds, not just labels. It is presented with Brier score and AUC.
5. **The threshold is a money decision.** Rule: bet 1 unit on the home win if P > t. The betting ledger is the confusion matrix:

   | | Home won | Home did not win |
   |---|---|---|
   | **Bet placed** | TP: win (odds − 1) | FP: −1 |
   | **No bet** | FN: missed win, 0 | TN: correctly skipped, 0 |

   The business number is P&L in units; t is selected on validation by P&L.

## Data preparation & feature engineering

All of this is in `features.py`.

Each candidate indicator is one "home minus away" number for each match:

| Indicator | How constructed | Idea |
|---|---|---|
| Elo difference | My own Elo: K = 15, a 60-point home advantage inside the expectation, and bigger moves for bigger wins (× log(1 + goal margin)) | Strength from long-term results |
| Shots-on-target dominance | Shots on target (exponentially weighted average, half-life 10 matches) for − against, home team minus away team | Recent underlying performance; less luck-dependent than goals |
| Goals dominance | Same, with goals | Recent results margin |
| Shots dominance | Same, with all shots | Opportunity volume |
| Corners dominance | Same, with corners | Territorial pressure |
| Points form | Weighted average of league points | Recent outcomes |
| Rest days | Days since last match, clipped to 2–10 | Fatigue from midweek matches |
| Promoted | 1 if the team was just promoted | Newcomer effect |

Promoted teams have no Premier League history. Each one inherits the average end-of-season state of the relegated teams: their Elo and all their rolling stats. The logic is "a promoted side is roughly as strong as the sides it replaces". If you gave every team a mid-table rating at the start, then every new team would be a mid-table team, but the 72 promoted sides from 2002/03 to 2025/26 finished 15.7th on average, with 85% ending up in the bottom half.

One number per idea, as a difference. Each indicator is (home team net value) minus (away team net value). A separate "home attack" and "away defence" coefficient would mostly measure the same thing twice and add noise.

Only the most informative indicators remain. Selection employs walk-forward validation: each of the ten seasons 2012/13–2021/22 is predicted by a model based only on the previous seasons. An indicator group is added only if it cuts mean log loss by ≥ 0.0005 and helps in at least 7 of the 10 seasons. The first table below is the greedy forward path; the second is each indicator separately.

| Step | Indicator added | Walk-forward log loss | Gain | Seasons improved |
|---|---|---:|---:|---:|
| 0 | none (base rate) | 0.6880 | | |
| 1 | Elo difference | 0.6101 | 0.0779 | 10/10 |
| 2 | Shots-on-target dominance | 0.6056 | 0.0045 | 10/10 |
| 3 | best remaining: goals | 0.6053 | 0.0004 | 6/10 → stop |

| Indicator | Log loss alone |
|---|---:|
| Elo | 0.6101 |
| Goals | 0.6115 |
| Shots on target | 0.6151 |
| Shots | 0.6177 |
| Points form | 0.6187 |
| Corners | 0.6249 |
| Promoted | 0.6733 |
| Rest | 0.6878 |

What this means:

- Elo + shots on target together are nearly as good as all eight indicators (0.6056 versus 0.6050), using a quarter of the inputs.
- Most candidates help on their own, but are redundant when Elo is available. Goals, points and promotion mostly re-measure strength. Shots on target survive because they capture how a team is actually playing, and results only reflect that with a lag and a lot of luck.
- Rest days have no signal at all (0.6878 versus 0.6880 for the base rate).
- Rating settings: half-life {5, 10, 20} × Elo K {15, 20, 30} were compared in the same way. The results are flat (0.6050–0.6057), so these choices don't matter much. The best was half-life 10, K 15.

Preprocessing: missing values (the very first match of a team in the data) are filled with the fitting-window median, then standardized with a scaler fitted on the fitting window only. Standardization is needed for gradient descent to work with only one learning rate. It also makes the coefficients comparable: one unit = one standard deviation, which is 160 Elo points and 2.9 shots on target per game.

## Modeling: three implementations, one model

Model: L2-regularised logistic regression on the 2 standardized indicators, predicting P(home win). sklearn minimizes `C·Σ log loss + ½‖w‖²`. Dividing by C·n gives `mean log loss + ‖w‖²/(2Cn)`, which has the same minimizer. That is what both PyTorch versions minimize. The bias is never penalized.

- **scikit-learn:** `LogisticRegression(C, tol=1e-10)` (lbfgs solver).
- **Manual PyTorch:** raw `w` and `b` tensors with `requires_grad`, the loss manually coded as `softplus(z) − y·z` (the numerically stable version of log loss), `backward()`, then `w -= lr * w.grad` under `torch.no_grad()`, then zero the grads. Full batch, zero initialization, 3,000 steps.
- **Standard PyTorch:** `nn.Module` wrapper of `nn.Linear(2, 1)`, `BCEWithLogitsLoss`, `torch.optim.SGD` with two parameter groups. Weights get `weight_decay = 1/(C·n)` because the decay SGD adds to the gradient is wd·w, which is the gradient of (wd/2)·‖w‖². Bias gets 0. Same initialization, same number of steps.

Tuning (validation only):

- **C:** {0.001 … 10} → 0.1. Validation log loss is the same (0.6080) from C = 0.1 onward. Regularization doesn't matter much; there is nothing to over-fit with 2 inputs and ~8,000 matches.
- **Learning rate:** the curvature of the log loss is at most ¼ of the covariance of the features, bounding the Hessian by L = 0.465. Gradient descent with lr ≤ 1/L = 2.15 is guaranteed to decrease the loss at every step. I picked lr = 1.0 as the fastest safe learning rate to get to sklearn's optimum (91 manual, 94 standard steps). lr = 3 also converged but is outside the guarantee; lr = 10 diverged.

Test results (2022/23 – Aug 2026, 1,540 matches):

| Model | Log loss ↓ | Brier ↓ | AUC ↑ | Accuracy @ 0.5 |
|---|---:|---:|---:|---:|
| scikit-learn LogisticRegression | 0.6153 | 0.2139 | 0.710 | 65.5% |
| Manual PyTorch loop | 0.6153 | 0.2139 | 0.710 | 65.5% |
| Standard PyTorch (nn.Module + optim) | 0.6153 | 0.2139 | 0.710 | 65.5% |
| Naive baseline: constant 45.8% | 0.6875 | 0.2472 | 0.500 | 55.5% |
| Benchmark: bookmaker (Bet365, margin removed) | 0.6031 | 0.2087 | 0.727 | 66.6% |

Are the three in agreement? Yes, to the float32 level:

| Parameter | sklearn | Manual PyTorch | Standard PyTorch |
|---|---:|---:|---:|
| Elo difference | 0.5380454 | 0.5380448 | 0.5380448 |
| Shots-on-target dominance | 0.3502269 | 0.3502274 | 0.3502275 |
| Intercept | −0.1966018 | −0.1966018 | −0.1966018 |

- Probabilities: the biggest difference in any test match is 2.5 × 10⁻⁷.
- Final objective: 0.6157773 for all 3 of them.
- Why the small differences persist: PyTorch uses float32, sklearn uses float64.
- Safety check: `train.py` will exit with an error if either PyTorch version ends more than 1e-5 above sklearn's optimum.

Reading the coefficients:

- **Elo:** +1 SD of Elo difference (160 points) multiplies the odds of a home win by e^0.538 = 1.71.
- **Shots on target:** +1 SD of shots-on-target dominance (2.9 shots on target per match) multiplies them by 1.42, in addition to Elo.
- **Intercept:** gives the home team about 45% for two evenly matched teams. That's home advantage.

What the comparison with the bookmaker indicates:

- **Gap closed:** 86% of the gap between the naive baseline and the bookmaker (0.6875 → 0.6153, vs 0.6031) is closed with just 2 public indicators. The 0.012 residual is the knowledge the market has and we lack: injuries, lineups, team news.
- **The empty-stadium season (2020/21, in validation):**

  | | Expected / actual home-win rate |
  |---|---:|
  | Real | 37.9% |
  | Model | 46.0% (fixed intercept for home advantage) |
  | Bookmaker | 41.8% (adjusted in part) |

  This is the most clear-cut single example of the market seeing information a model trained on history cannot see.
- **Betting:**
  - *Threshold rule.* On validation no threshold made money. The least bad, with at least 30 bets, was t = 0.80 (−0.9 units). Applied to test, it bet 51 times on heavy favourites and won 48: +4.2 units (+8.3% per bet). Bootstrapping those 51 bets yields a 95% interval of −0.1 to +7.6 units. I see this as luck, not an edge, considering the loss on validation and the fact that t was chosen from a large number of candidates.
  - *Value-bet rule* (bet when P × odds > 1). It loses: −6.9% per bet on 719 test bets, and −1.2% on validation. When the model and the market disagree, the market is usually right.

## Limitations & next steps

1. **No team news.** Line-ups, injuries and suspensions are the market's key advantage. Fix: add availability data (such as minutes lost to injury, weighted by player rating) from a lineup source. That means a new data feed; there is nothing in this file to substitute for it.
2. **Home advantage is a constant.** The model was 8 points off in home-win rate in the empty-stadium season. Fix: let the intercept move over time, e.g. a rolling league-wide home-win rate over the last ~100 matches as an extra indicator, or a crowd attendance flag.
3. **The target discards the draw.** "Not a home win" comprises draws and away wins, which have different odds. Fix: fit two binary logistic models (home win; away win) and get P(draw) as the remainder. Both remain in the correct model class.
4. **Betting results are statistically thin.** 51 test bets don't separate skill from chance. Fix: evaluate the betting rule walk-forward over all seasons since 2012 (≈ 10× more bets) and decide the rule before looking at the test seasons.
5. **One league.** Fix: the same pipeline could be run unchanged on the other 37 divisions in the source data. That would test whether shots on target tell you something everywhere, or just in the Premier League.
6. **Shots on target are a very crude measure of quality.** A tame 35-yard shot is worth the same as a tap-in. Fix: expected goals (xG) data, which weights each chance by its probability of being scored. It's not in this dataset.

## Generative AI use disclosure

I used Claude (Anthropic) as an assistant on this assignment:

- **Dataset and framing:** I chose the Premier League home-win problem and the betting framing. Claude helped merge relevant data accordingly
- **Features:** The choice of indicators was mine. Claude went on to forming these features from the data.
- **Code:** Claude wrote the code in `features.py`, `train.py` and `app.py`: data cleaning, the rating pass, the three training implementations, and the Gradio dashboard.
