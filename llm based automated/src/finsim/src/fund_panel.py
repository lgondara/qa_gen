"""Fund-panel Monte Carlo: replicating the persistence/survivorship
methodology of Plagge et al. (2021), "The Case for Low-Cost Index-Fund
Investing" (Vanguard, UK edition), on synthetic panels with controlled
ground truth.

The paper's Figure 5 ranks active funds into excess-return quintiles over
one five-year window and tabulates their quintile (or death) in the next
five-year window. This module generates synthetic fund panels under
explicit data-generating processes (DGPs) and applies the identical
tabulation, so the observed matrix can be tested against competing
explanations:

    H0        pure luck: zero skill, homogeneous costs
    H_cost    zero skill, heterogeneous persistent expense ratios
    H_tilt    zero skill, persistent factor tilts (partially correlated
              across windows, per the paper's 'portfolio tilts' caveat)
    H_skill   a minority of funds with genuine persistent alpha

Attrition (the merged/liquidated column) is modeled as a performance-
dependent hazard, consistent with the paper's Figure 6 finding that dead
funds trail their benchmarks before closing, plus a baseline hazard for
non-performance-related mergers.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Observed matrix: Plagge et al. (2021), Figure 5 (UK active equity funds;
# ranking window to Dec 2015, evaluation window to Dec 2020). Rows are
# initial quintiles Q1 (top) to Q5; columns are subsequent Q1..Q5, dead.
PAPER_FIGURE_5 = np.array([
    [27.1, 19.9, 17.6, 10.4,  9.3, 15.8],
    [19.5, 17.9, 18.3, 14.6, 11.9, 17.8],
    [11.1, 16.6, 14.3, 18.4, 16.8, 23.0],
    [ 8.4, 12.1, 15.9, 18.4, 18.0, 27.1],
    [10.0, 10.3, 10.3, 14.6, 20.3, 34.4],
])


@dataclass
class DGP:
    """Data-generating process for a two-window fund panel."""
    name: str
    n_funds: int = 2800
    months: int = 60               # per window
    tracking_error: float = 0.06   # annual idiosyncratic vol of excess return
    cost_hetero: bool = True       # lognormal expense ratios vs constant 1.2%
    tilt_sd: float = 0.0           # annual sd of persistent factor tilt
    tilt_corr: float = 0.7         # correlation of tilt across windows
    skill_frac: float = 0.0        # fraction of funds with true alpha
    skill_alpha: float = 0.0       # annual alpha of skilled funds
    death_base: float = 0.03       # non-performance baseline death prob
    death_slope: float = 0.39      # extra death prob for worst performer


H0 = DGP("H0_pure_luck", cost_hetero=False)
H_COST = DGP("H_cost")
H_TILT = DGP("H_tilt", tilt_sd=0.02, tilt_corr=0.7)
H_SKILL = DGP("H_skill", skill_frac=0.10, skill_alpha=0.02)


def simulate_panel(dgp: DGP, seed: int = 0) -> np.ndarray:
    """One synthetic panel -> 5x6 transition matrix (row percentages)."""
    r = np.random.default_rng(seed)
    n, m = dgp.n_funds, dgp.months
    sig = dgp.tracking_error / np.sqrt(12)

    cost = (np.clip(r.lognormal(np.log(0.011), 0.35, n), 0.004, 0.03)
            if dgp.cost_hetero else np.full(n, 0.012))
    t1 = r.normal(0, dgp.tilt_sd, n)
    t2 = (dgp.tilt_corr * t1
          + np.sqrt(max(1 - dgp.tilt_corr**2, 0)) * r.normal(0, dgp.tilt_sd, n))
    alpha = np.zeros(n)
    if dgp.skill_frac > 0:
        alpha[r.random(n) < dgp.skill_frac] = dgp.skill_alpha

    def window(tilt, s):
        rr = np.random.default_rng(s)
        noise = rr.normal(0, sig, (n, m)).mean(1) * 12
        return noise + tilt + alpha - cost

    ex1 = window(t1, seed * 2 + 1)
    ex2 = window(t2, seed * 2 + 2)

    pct = pd.Series(0.5 * ex1 + 0.5 * ex2).rank(pct=True).values
    p_die = np.clip(dgp.death_base + dgp.death_slope * (1 - pct), 0.03, None)
    dead = r.random(n) < p_die

    q1 = 4 - pd.qcut(ex1, 5, labels=False)          # 0 = top quintile
    q2 = np.full(n, 5)                               # 5 = merged/liquidated
    surv = ~dead
    q2[surv] = 4 - pd.qcut(ex2[surv], 5, labels=False)

    mat = np.zeros((5, 6))
    for i in range(5):
        row = q2[q1 == i]
        for j in range(6):
            mat[i, j] = (row == j).mean() * 100
    return mat


def monte_carlo(dgp: DGP, n_reps: int = 100):
    """Replicated panels -> (mean matrix, per-rep Frobenius distances to
    the paper's observed matrix). The distance distribution under each DGP
    gives a Monte Carlo assessment of which DGPs are consistent with the
    observed pattern."""
    mats = np.stack([simulate_panel(dgp, seed=s) for s in range(n_reps)])
    dists = np.sqrt(((mats - PAPER_FIGURE_5) ** 2).sum(axis=(1, 2)))
    return mats.mean(0), dists


def as_frame(mat: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        mat, index=[f"Q{i+1}" for i in range(5)],
        columns=["->Q1", "->Q2", "->Q3", "->Q4", "->Q5", "dead"]).round(1)


def report(n_reps: int = 100) -> None:
    print("Observed (Plagge et al. 2021, Figure 5):")
    print(as_frame(PAPER_FIGURE_5).to_string())
    for dgp in [H0, H_COST, H_TILT, H_SKILL]:
        mean, dists = monte_carlo(dgp, n_reps)
        print(f"\n{dgp.name}  (Frobenius distance to observed: "
              f"{dists.mean():.1f} ± {dists.std():.1f})")
        print(as_frame(mean).to_string())
        print(f"  winner persistence Q1->Q1: {mean[0,0]:.1f}%   "
              f"loser persistence Q5->Q5: {mean[4,4]:.1f}%   "
              f"death gradient: {mean[0,5]:.1f}% -> {mean[4,5]:.1f}%")


if __name__ == "__main__":
    report()
