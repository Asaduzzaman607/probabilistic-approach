#!/usr/bin/env python3
"""
demo_synthetic_run.py
----------------------
*** SYNTHETIC DATA -- NOT REAL RESULTS -- FOR CODE DEMONSTRATION ONLY ***

The raw student interaction logs this pipeline is designed to analyze are
not included in this repository (see README.md, "Data availability").
This script exists ONLY to let a reviewer see the analyze -> edge_table_with_counts
-> by_type_significance chain actually execute, using:

  - REAL data: the 34-problem stratified sample and its real per-problem
    classification (problem_type, unknown_slot, given_set), from
    ../sample_data/template_sample_change{3,4}.csv, and the real
    LLM-proposed prerequisite edges for those problems, from
    ../sample_data/prerequisites_change{3,4}_sample.json.

  - FABRICATED data: student attempts. For each sampled problem, this
    script invents N_STUDENTS students, each assigned a uniformly random
    order over the three quantity-entry steps. A student's correctness on
    the "dependent" (unknown) step is then drawn with probability p_ok if
    their random order happens to satisfy ALL of that problem's real
    LLM-recommended edges, and a lower probability if it does not -- i.e.
    the synthetic data is constructed so that following the LLM's
    recommended order helps, by design, purely to produce a
    non-degenerate demo. This has no bearing on whether the effect is
    real in your actual data; the true result is in the paper and in the
    (unshared) real edge_table_with_counts_by_type_llm_significance.csv.

Usage (from code/):
    python3 demo_synthetic_run.py
Writes demo_synthetic_output/{change3,change4}/edge_table_with_counts_by_type_llm_SYNTHETIC.csv
and .../*_significance.csv
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).with_name("pipeline")))

import prerequisite_analysis as PA
from data_loading import load_log_csv
from completion_order_analysis import build_orders_df
from order_effects_analysis import _clean_id
from edge_table_with_counts import edge_role, pooled_counts_for_edge
import by_type_significance as BTS

SAMPLE_DATA = Path(__file__).resolve().parent.parent / "sample_data"
OUT = Path(__file__).with_name("demo_synthetic_output")
STEPS = ["NumeratorQuantity1", "NumeratorQuantity2", "DenominatorQuantity1"]
N_STUDENTS = 10   # fake students per sampled problem
P_OK_COMPLIANT = 0.85     # P(correct on unknown step | order followed the LLM's recommendation)
P_OK_NONCOMPLIANT = 0.35  # P(correct on unknown step | order violated it)
P_OK_GIVEN_STEP = 0.90    # P(correct on a "given" step) -- kept high but not 1.0, to avoid
                          # scipy's zero-cell edge case on very small synthetic samples
SEED = 42

CHANGE_COLUMNS = ["school_id", "prob_num", "student_id", "status", "problem_id",
                  "student_used_code", "eta", "optimal_strategy", "step_sequence",
                  "kc_skills", "problem_order"]

_SHORT_STEP = {"NumeratorQuantity1": "NumQ1", "NumeratorQuantity2": "NumQ2",
               "DenominatorQuantity1": "DenomQ1"}


def _short_rule(edge: str) -> str:
    """'DenominatorQuantity1->NumeratorQuantity1' -> 'DenomQ1->NumQ1', matching
    the paper's Table 6/7 'Rule P -> D' column."""
    a, b = edge.split("->")
    return f"{_SHORT_STEP[a]}->{_SHORT_STEP[b]}"


def _display_problem_type(pt: str) -> str:
    """The raw problem_type value is stored as e.g. 'change_find_amount',
    'prop_find_total' -- reorder to 'find_change_amount', 'find_prop_total'
    for display, matching the paper's Table 6/7 and
    make_figure_coverage_impact.py's labeling convention."""
    tokens = pt.split("_")
    if len(tokens) >= 2 and tokens[1] == "find":
        tokens = [tokens[1], tokens[0]] + tokens[2:]
    return "_".join(tokens)


def load_sample(lesson: str):
    """Real: per-problem classification, already resolved (no metadata needed)."""
    rows = {}
    with open(SAMPLE_DATA / f"template_sample_{lesson}.csv", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["lms-id"]] = row
    return rows


def make_synthetic_log(lesson: str, sample_rows: dict, prereqs: dict, rng: random.Random):
    log_rows = []
    for pid, pairs in prereqs.items():
        if pid not in sample_rows:
            continue
        for i in range(N_STUDENTS):
            order = STEPS[:]
            rng.shuffle(order)
            pos = {s: j for j, s in enumerate(order)}
            complies = all(pos[a] < pos[b] for a, b in pairs)
            p_ok_last = P_OK_COMPLIANT if complies else P_OK_NONCOMPLIANT
            toks = []
            for t, step in enumerate(order):
                is_last = step == order[-1]
                p = p_ok_last if is_last else P_OK_GIVEN_STEP
                outcome = "OK" if rng.random() < p else "ERROR"
                toks.append(f"{step}-Attempt-1-Numeric-{outcome}-2024-01-01T00:00:{t:02d}")
            log_rows.append(["S1", 1, f"{pid}__s{i}", "DONE", pid, "", "", "ER",
                             "\t".join(toks), "", 1])
    return log_rows


def run_lesson(lesson: str):
    print(f"\n{'='*70}\n[{lesson}] SYNTHETIC demo run\n{'='*70}")
    sample_rows = load_sample(lesson)
    prereqs = PA.load_prerequisite_annotations(str(SAMPLE_DATA / f"prerequisites_{lesson}_sample.json"))
    print(f"[{lesson}] {len(sample_rows)} real sampled problems, "
          f"{len(prereqs)} with real LLM prerequisites")

    rng = random.Random(SEED)
    log_rows = make_synthetic_log(lesson, sample_rows, prereqs, rng)
    out_dir = OUT / lesson
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{lesson}-data-SYNTHETIC.csv"
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CHANGE_COLUMNS)
        w.writerows(log_rows)
    print(f"[{lesson}] wrote {len(log_rows)} SYNTHETIC (fabricated) student rows -> {log_path.name}")

    log = load_log_csv(str(log_path), "change")
    odf = build_orders_df(log)

    # group problems by (edge, role, problem_type, given_set), using the REAL
    # per-problem classification already resolved in template_sample_*.csv --
    # same grouping logic edge_table_with_counts.py's main() uses.
    groups: dict[tuple, set] = {}
    for pid, pairs in prereqs.items():
        row = sample_rows.get(pid)
        if row is None:
            continue
        unknown, ptype, gset = row["unknown_slot"], row["problem_type"], row["given_set"]
        seen = set()
        for a, b in pairs:
            edge = f"{a}->{b}"
            if edge in seen:
                continue
            seen.add(edge)
            key = (edge, edge_role(a, b, unknown), ptype, gset)
            groups.setdefault(key, set()).add(pid)

    out_rows = []
    for (edge, role, ptype, gset), pid_set in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2]), str(kv[0][3]))):
        pids = sorted(pid_set)
        a, b = edge.split("->")
        for suffix in ("first_touch", "completion"):
            c = pooled_counts_for_edge(odf, pids, a, b, suffix)
            out_rows.append({"edge": edge, "role": role, "problem_type": ptype,
                             "given_set": gset, "order_basis": suffix,
                             "n_problems": len(pids), **c})

    counts_df = pd.DataFrame(out_rows)
    # drop the small-N degenerate rows (0 errors on BOTH sides for a step) that
    # scipy.stats.chi2_contingency cannot build a valid table from -- an
    # artifact of the thin synthetic sample size, not of real (large-N) data.
    degenerate = ((counts_df["err_satisfied_A"] == 0) & (counts_df["err_violated_A"] == 0)) | \
                 ((counts_df["err_satisfied_B"] == 0) & (counts_df["err_violated_B"] == 0))
    kept = counts_df[~degenerate].reset_index(drop=True)
    print(f"[{lesson}] {len(groups)} (edge, role, problem_type, given_set) groups -> "
          f"{len(counts_df)} rows, {len(kept)} kept ({degenerate.sum()} dropped as "
          f"degenerate small-sample zero-error rows)")

    counts_path = out_dir / "edge_table_with_counts_by_type_llm_SYNTHETIC.csv"
    kept.to_csv(counts_path, index=False)

    if kept.empty:
        print(f"[{lesson}] no rows survived filtering -- try a different SEED.")
        return

    results = pd.concat([BTS.build_results(kept, "A"), BTS.build_results(kept, "B")], ignore_index=True)
    import numpy as np
    results["p_adj_fdr"] = np.nan
    for basis, g in results.groupby("order_basis"):
        results.loc[g.index, "p_adj_fdr"] = BTS.benjamini_hochberg(g["p_value"].values)
    results["significant"] = results["p_adj_fdr"] < 0.05
    results["sig_stars"] = results["p_adj_fdr"].apply(BTS.sig_stars)
    results["direction"] = np.where(results["diff_pp"] > 0,
                                    "higher error when SATISFIED (order hurts)",
                                    "higher error when VIOLATED (order helps)")
    results = results.sort_values(["order_basis", "p_adj_fdr"])

    sig_path = out_dir / "edge_table_with_counts_by_type_llm_SYNTHETIC_significance.csv"
    results.to_csv(sig_path, index=False)
    print(f"[{lesson}] wrote {sig_path.relative_to(Path.cwd()) if sig_path.is_relative_to(Path.cwd()) else sig_path}")

    # display-only formatting to match the paper's Table 6/7 style
    # (saved CSV above keeps the pipeline's real column names/values unchanged)
    display = results.copy()
    display["problem_type"] = display["problem_type"].map(_display_problem_type)
    display["rule"] = display["edge"].map(_short_rule)
    display = display.drop(columns=["edge"])

    cols = ["rule", "role", "problem_type", "given_set", "order_basis", "error_step",
            "satisfied_pct", "violated_pct", "diff_pp", "p_adj_fdr", "sig_stars", "direction"]
    results = display
    with pd.option_context("display.width", 160, "display.max_rows", None):
        print(results[cols].round(2).to_string(index=False))


def main():
    print(__doc__)
    for lesson in ("change3", "change4"):
        run_lesson(lesson)
    print(f"\n{'='*70}\nDone. Output written under {OUT}/ -- SYNTHETIC, not real results.\n{'='*70}")


if __name__ == "__main__":
    main()
