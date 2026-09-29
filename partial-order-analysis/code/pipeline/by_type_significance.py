#!/usr/bin/env python3
"""
by_type_significance.py
------------------------
Applies the paper's statistical framework (chi-square test,
Wald CI for the risk difference, odds-ratio CI with Haldane-Anscombe
correction, Benjamini-Hochberg FDR) to the by-type pooled-count tables
produced by edge_table_with_counts.py, instead of reporting raw
percentages.

Faithful port of the notebook's `two_proportion_test` / `wald_ci_diff` /
`odds_ratio_ci` / `benjamini_hochberg` functions (Section 2 of
proportional_reasoning_order_analysis.ipynb) -- same math, same
correction, same significance thresholds.

CRITICAL -- never pools across lesson (change3/change4) or order_basis
(first_touch/completion):
  - Each row's chi-square/OR/CI is computed from that row's own counts
    only (no cross-row pooling there either -- matches the notebook's
    per-row test).
  - The BH-FDR correction is run SEPARATELY within each order_basis
    group (first_touch and completion get their own correction), never
    across both at once.
  - This script is run once per lesson's CSV, so change3 and change4
    are never in the same correction or the same output file.
This is stricter than the notebook's original Section 3 (which pooled
FDR across all four dataset/touch_type combos at once) and matches the
"no pooling across dataset or touch_type" discipline adopted later in
the notebook (Sections 13-16, "context-specific odds ratios: no
pooling at all").

Usage:
    python by_type_significance.py output/change3/edge_table_with_counts_by_type_llm.csv output/change3
    python by_type_significance.py output/change4/edge_table_with_counts_by_type_llm.csv output/change4
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def two_proportion_test(count1, total1, count2, total2):
    """Chi-square test of independence for a 2x2 table; equivalent to a
    two-sided two-proportion z-test for 2x2 tables."""
    table = [[count1, total1 - count1], [count2, total2 - count2]]
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    return chi2, p


def wald_ci_diff(count1, total1, count2, total2, alpha=0.05):
    """Wald 95% CI for the difference in two proportions (p1 - p2)."""
    p1, p2 = count1 / total1, count2 / total2
    diff = p1 - p2
    se = np.sqrt(p1 * (1 - p1) / total1 + p2 * (1 - p2) / total2)
    z = stats.norm.ppf(1 - alpha / 2)
    return diff, diff - z * se, diff + z * se


def odds_ratio_ci(count1, total1, count2, total2, alpha=0.05, correction=0.5):
    """Odds ratio (group1 vs group2) with a 95% log-scale CI. Haldane-Anscombe
    0.5 correction applied when any cell of the 2x2 table is zero."""
    a, b = count1, total1 - count1
    c, d = count2, total2 - count2
    if min(a, b, c, d) == 0:
        a, b, c, d = a + correction, b + correction, c + correction, d + correction
    orr = (a * d) / (b * c)
    se_log = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = stats.norm.ppf(1 - alpha / 2)
    log_or = np.log(orr)
    return orr, np.exp(log_or - z * se_log), np.exp(log_or + z * se_log)


def benjamini_hochberg(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    bh = ranked * n / (np.arange(n) + 1)
    bh_min = np.minimum.accumulate(bh[::-1])[::-1]
    bh_min = np.clip(bh_min, 0, 1)
    out = np.empty(n)
    out[order] = bh_min
    return out


def sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def build_results(df: pd.DataFrame, err_label: str) -> pd.DataFrame:
    """err_label = 'A' or 'B' -- matches the _A/_B suffixed columns in
    edge_table_with_counts.py's output."""
    sat_count_col, sat_total_col = f"err_satisfied_{err_label}", f"n_scored_satisfied_{err_label}"
    vio_count_col, vio_total_col = f"err_violated_{err_label}", f"n_scored_violated_{err_label}"
    rows = []
    for _, r in df.iterrows():
        sc, st = r[sat_count_col], r[sat_total_col]
        vc, vt = r[vio_count_col], r[vio_total_col]
        if st == 0 or vt == 0:
            continue  # can't test an empty cell
        chi2, p = two_proportion_test(sc, st, vc, vt)
        diff, lo, hi = wald_ci_diff(sc, st, vc, vt)
        orr, or_lo, or_hi = odds_ratio_ci(sc, st, vc, vt)
        rows.append({
            "edge": r["edge"], "role": r["role"],
            "problem_type": r["problem_type"], "given_set": r["given_set"],
            "order_basis": r["order_basis"], "n_problems": r["n_problems"],
            "error_step": err_label,
            "satisfied_count": sc, "satisfied_total": st, "satisfied_pct": 100 * sc / st,
            "violated_count": vc, "violated_total": vt, "violated_pct": 100 * vc / vt,
            "diff_pp": diff * 100, "diff_ci_low": lo * 100, "diff_ci_high": hi * 100,
            "odds_ratio": orr, "or_ci_low": or_lo, "or_ci_high": or_hi,
            "chi2": chi2, "p_value": p,
        })
    return pd.DataFrame(rows)


def main():
    if len(sys.argv) != 3:
        print("usage: python by_type_significance.py <edge_table_with_counts_by_type_llm.csv> <out_dir>")
        sys.exit(1)
    in_csv, out_dir = sys.argv[1], Path(sys.argv[2])
    df = pd.read_csv(in_csv)

    results = pd.concat([build_results(df, "A"), build_results(df, "B")], ignore_index=True)

    # BH-FDR run SEPARATELY within each order_basis (first_touch / completion) --
    # never pooled across them. This script processes one lesson's CSV per run,
    # so change3/change4 are never pooled here either.
    results["p_adj_fdr"] = np.nan
    for basis, g in results.groupby("order_basis"):
        results.loc[g.index, "p_adj_fdr"] = benjamini_hochberg(g["p_value"].values)
    results["significant"] = results["p_adj_fdr"] < 0.05
    results["sig_stars"] = results["p_adj_fdr"].apply(sig_stars)
    results["direction"] = np.where(results["diff_pp"] > 0,
                                     "higher error when SATISFIED (order hurts)",
                                     "higher error when VIOLATED (order helps)")

    results = results.sort_values(["order_basis", "p_adj_fdr"])

    for basis, g in results.groupby("order_basis"):
        n_sig = int(g["significant"].sum())
        print(f"\n=== {in_csv} | order_basis={basis}: {n_sig} of {len(g)} comparisons "
              f"significant after within-basis BH-FDR (alpha=0.05) ===\n")
        cols = ["edge", "role", "problem_type", "given_set", "n_problems", "error_step",
                "satisfied_pct", "violated_pct", "diff_pp", "odds_ratio", "or_ci_low", "or_ci_high",
                "p_adj_fdr", "sig_stars", "direction"]
        with pd.option_context("display.width", 200, "display.max_rows", None):
            print(g[cols].round(3).to_string(index=False))

    out_path = out_dir / (Path(in_csv).stem + "_significance.csv")
    results.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
