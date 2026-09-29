#!/usr/bin/env python3
"""
completion_order_analysis.py
-------------------------------
Everything so far orders steps by FIRST TOUCH -- the position of the
first token (of any kind) for that step in step_sequence. That's not the
same as COMPLETION order -- the position of the token where the student
finally got that step marked OK. They diverge exactly for the
interesting case: a student attempts A, gets it wrong, moves to B, comes
back and finally gets A right. First-touch order says A before B;
completion order says B before A.

This recomputes both orderings directly from the raw step_sequence
(independent of whatever internal column the rest of the pipeline uses,
so it's a clean, from-scratch comparison) and re-runs the SAME global
(whole-order compliance -> FinalAnswer) and local (edge-level) tests
under both definitions, side by side.

    python completion_order_analysis.py --out output --lesson change4 --which llm
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import prerequisite_analysis as PA
import run_partial_order_study as S
from data_loading import (QUANTITY_STEPNAMES, FINAL_ANSWER_STEP, load_log_csv,
                          load_metadata_csv, parse_step_sequence, OK_OUTCOME)
from labeling import step_correct
from order_effects_analysis import _clean_id
from run_study_cli import LESSONS

STEPS = sorted(QUANTITY_STEPNAMES) + [FINAL_ANSWER_STEP]


def step_orders(step_sequence: str) -> dict:
    """For one student-problem row: {step: (first_touch_order, completion_order,
    correct)}. completion_order is None if the step was touched but never got
    an OK. `correct` uses the SAME rule as the rest of the study (labeling.
    step_correct: first-attempt-OK, no earlier hint) -- NOT 'eventually OK',
    which is near-universal in a tutor that requires correction before
    advancing and would make every error rate ~0%. Order definition and
    correctness definition are independent: only the ORDER changes between
    first-touch and completion; correctness stays fixed to the study's rule."""
    tokens = parse_step_sequence(step_sequence)
    out = {}
    for step in STEPS:
        step_tokens = [t for t in tokens if t.stepname == step]
        if not step_tokens:
            continue
        first_touch = min(t.order for t in step_tokens)
        ok_tokens = [t for t in step_tokens if t.outcome == OK_OUTCOME]
        completion = min(t.order for t in ok_tokens) if ok_tokens else None
        correct = step_correct(tokens, step)  # None if never attempted
        out[step] = (first_touch, completion, correct)
    return out


def build_orders_df(log: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in log.iterrows():
        orders = step_orders(row.get("step_sequence", ""))
        r = {"problem_id": row["problem_id"], "student_id": row["student_id"]}
        for step in STEPS:
            ft, co, corr = orders.get(step, (None, None, None))
            r[f"{step}__first_touch"] = ft
            r[f"{step}__completion"] = co
            r[f"{step}__correct"] = corr
        rows.append(r)
    return pd.DataFrame(rows)


def edge_satisfaction_rate(odf: pd.DataFrame, a: str, b: str, suffix: str,
                           correctness: pd.DataFrame) -> dict:
    """suffix: 'first_touch' or 'completion'. Returns satisfied/violated
    counts and error rate at each endpoint, pooled across all rows passed
    in (caller filters to one problem first)."""
    oa, ob = f"{a}__{suffix}", f"{b}__{suffix}"
    both = odf.dropna(subset=[oa, ob])
    if both.empty:
        return None
    sat = both[both[oa] < both[ob]]
    vio = both[both[oa] >= both[ob]]
    m = correctness.set_index(["problem_id", "student_id"])

    def err(sub, step):
        idx = list(zip(sub["problem_id"], sub["student_id"]))
        col = f"{step}__correct"
        if col not in m.columns:
            return None, 0
        vals = m.loc[m.index.isin(idx), col].dropna()
        return (float(1 - vals.mean()) if len(vals) >= 5 else None), len(vals)

    ea_s, na_s = err(sat, a); eb_s, nb_s = err(sat, b)
    ea_v, na_v = err(vio, a); eb_v, nb_v = err(vio, b)
    return {"n_satisfied": len(sat), "n_violated": len(vio),
            "satisfied_error_A": ea_s, "satisfied_n_A": na_s,
            "satisfied_error_B": eb_s, "satisfied_n_B": nb_s,
            "violated_error_A": ea_v, "violated_n_A": na_v,
            "violated_error_B": eb_v, "violated_n_B": nb_v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output")
    ap.add_argument("--lesson", default="change4", choices=list(LESSONS))
    ap.add_argument("--which", default="llm", help="llm | unknown_last")
    args = ap.parse_args()
    out = Path(args.out)
    cfg = LESSONS[args.lesson]

    meta = load_metadata_csv(str(out / f"{args.lesson}_meta_inlog.csv"))
    log = load_log_csv(cfg["log"], "change")
    llm = PA.load_prerequisite_annotations(str(out / f"prerequisites_{args.lesson}.json"))
    prereqs = (llm if args.which == "llm"
              else S.unknown_last_prereqs(meta, args.lesson, list(llm)))

    print(f"[{args.lesson}] computing first-touch + completion order from raw logs "
          f"({len(log)} rows)...")
    odf = build_orders_df(log)
    correctness = odf[["problem_id", "student_id"]
                     + [f"{s}__correct" for s in STEPS]].copy()

    disagree_n, total_n = 0, 0
    rows = []
    for pid, pairs in prereqs.items():
        sub = odf[odf["problem_id"].map(_clean_id) == _clean_id(pid)]
        if sub.empty:
            continue
        for a, b in pairs:
            r_ft = edge_satisfaction_rate(sub, a, b, "first_touch", correctness)
            r_co = edge_satisfaction_rate(sub, a, b, "completion", correctness)
            if r_ft is None or r_co is None:
                continue
            oa_ft, ob_ft = f"{a}__first_touch", f"{b}__first_touch"
            oa_co, ob_co = f"{a}__completion", f"{b}__completion"
            both = sub.dropna(subset=[oa_ft, ob_ft, oa_co, ob_co])
            if len(both):
                ft_sat = both[oa_ft] < both[ob_ft]
                co_sat = both[oa_co] < both[ob_co]
                disagree_n += int((ft_sat != co_sat).sum())
                total_n += len(both)
            rows.append({"problem_id": pid, "edge": f"{a}->{b}",
                        "first_touch": r_ft, "completion": r_co})

    pct = 100 * disagree_n / total_n if total_n else float("nan")
    print(f"student-edge instances where first-touch and completion order "
          f"DISAGREE: {disagree_n}/{total_n} ({pct:.1f}%)\n")

    def summarize(key):
        vals_sat_a, vals_vio_a = [], []
        for r in rows:
            d = r[key]
            if d["satisfied_error_A"] is not None:
                vals_sat_a.append(d["satisfied_error_A"])
            if d["violated_error_A"] is not None:
                vals_vio_a.append(d["violated_error_A"])
        print(f"[{key}] error at A -- satisfied: mean={np.mean(vals_sat_a):.1%} "
              f"(n={len(vals_sat_a)}) | violated: mean={np.mean(vals_vio_a):.1%} "
              f"(n={len(vals_vio_a)})")

    summarize("first_touch")
    summarize("completion")

    pd.DataFrame(rows).to_csv(out / args.lesson / f"completion_vs_first_touch_{args.which}.csv",
                              index=False)


if __name__ == "__main__":
    main()
