#!/usr/bin/env python3
"""
edge_table_with_counts.py
---------------------------
Same by-(edge, role) breakdown as edge_local_effect.py, but:

  1. Reports RAW POOLED COUNTS (total students, total errors) summed
     across all problems sharing an (edge, role, problem_type, given_set)
     -- not just the mean of each problem's own rate. A mean-of-rates can
     look identical whether it's built from 4 students or 4,000; pooled
     counts can't hide that.
  2. Computes BOTH order definitions (first-touch and completion) side
     by side, using the same students so numbers are directly comparable.
  3. NEW: groups by problem_type + given_set as well as edge + role, so
     an edge that appears in multiple problem types (e.g. the same slot
     pair playing different roles across change_find_amount / prop_find_part)
     is never silently pooled across types that may behave differently.

No LLM calls -- reads the saved prerequisites_*.json and raw logs, same
as the rest of this pipeline.

    python edge_table_with_counts.py --out output --lesson change4 --which llm
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import prerequisite_analysis as PA
import run_partial_order_study as S
from completion_order_analysis import build_orders_df, STEPS
from data_loading import load_log_csv, load_metadata_csv
from order_effects_analysis import _clean_id
from problem_type_stratify import classify_problem, classify_metadata
from run_study_cli import LESSONS


def edge_role(a: str, b: str, unknown: str) -> str:
    if b == unknown:
        return "into_unknown"
    if a == unknown:
        return "out_of_unknown"
    return "between_givens"


def pooled_counts_for_edge(odf: pd.DataFrame, pids: list, a: str, b: str,
                          suffix: str) -> dict:
    """Pool RAW counts across every problem in `pids` that has this exact
    edge, for one order-basis (suffix = 'first_touch' or 'completion')."""
    oa, ob = f"{a}__{suffix}", f"{b}__{suffix}"
    ca, cb = f"{a}__correct", f"{b}__correct"
    tot = {"n_satisfied": 0, "n_violated": 0,
          "err_satisfied_A": 0, "n_scored_satisfied_A": 0,
          "err_violated_A": 0, "n_scored_violated_A": 0,
          "err_satisfied_B": 0, "n_scored_satisfied_B": 0,
          "err_violated_B": 0, "n_scored_violated_B": 0}
    for pid in pids:
        sub = odf[odf["problem_id"].map(_clean_id) == _clean_id(pid)]
        both = sub.dropna(subset=[oa, ob])
        if both.empty:
            continue
        sat = both[both[oa] < both[ob]]
        vio = both[both[oa] >= both[ob]]
        tot["n_satisfied"] += len(sat)
        tot["n_violated"] += len(vio)
        for group, label in ((sat, "satisfied"), (vio, "violated")):
            for col, tag in ((ca, "A"), (cb, "B")):
                vals = group[col].dropna()
                tot[f"n_scored_{label}_{tag}"] += len(vals)
                tot[f"err_{label}_{tag}"] += int((vals == 0).sum())
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output")
    ap.add_argument("--lesson", default="change4", choices=list(LESSONS))
    ap.add_argument("--which", default="llm", help="llm | unknown_last")
    args = ap.parse_args()
    out = Path(args.out)
    cfg = LESSONS[args.lesson]

    meta = load_metadata_csv(str(out / f"{args.lesson}_meta_inlog.csv"))
    by_id = {_clean_id(r["lms-id"]): r for _, r in meta.iterrows()}
    llm = PA.load_prerequisite_annotations(str(out / f"prerequisites_{args.lesson}.json"))
    prereqs = (llm if args.which == "llm"
              else S.unknown_last_prereqs(meta, args.lesson, list(llm)))

    log = load_log_csv(cfg["log"], "change")
    print(f"[{args.lesson}] parsing raw logs for first-touch + completion order "
          f"({len(log)} rows)...")
    odf = build_orders_df(log)

    # --- exact coverage audit: metadata -> in-log -> has >=1 LLM edge -----
    # (replaces any hand-summed estimate of "how many problems are covered")
    covered_pids = {_clean_id(pid) for pid in prereqs if _clean_id(pid) in by_id}
    print(f"\n[{args.lesson}] coverage: {len(by_id)} in-log metadata rows, "
          f"{len(covered_pids)} have >=1 LLM-proposed edge "
          f"({len(by_id) - len(covered_pids)} have none)")

    cls = classify_metadata(meta, args.lesson)
    cls["_cid"] = cls["lms-id"].map(_clean_id)
    cls["has_llm_edge"] = cls["_cid"].isin(covered_pids)
    audit_tbl = (cls.groupby(["problem_type", "given_set"], dropna=False)
                    .agg(n_in_log=("has_llm_edge", "size"),
                         n_with_edge=("has_llm_edge", "sum"))
                    .reset_index()
                    .sort_values(["problem_type", "given_set"]))
    print(f"\n[{args.lesson}] per-type coverage (in-log metadata rows vs. rows with an LLM edge):")
    print(audit_tbl.to_string(index=False))
    print()

    # group problems by (edge, role, problem_type, given_set); dedupe pids
    # defensively so a problem can never be counted twice within one group
    groups: dict[tuple, set] = {}
    for pid, pairs in prereqs.items():
        cid = _clean_id(pid)
        if cid not in by_id:
            continue
        info = classify_problem(by_id[cid], args.lesson)
        unknown = info["unknown_slot"]
        ptype = info["problem_type"]
        gset = info["given_set"]
        seen_edges_this_pid = set()
        for a, b in pairs:
            edge = f"{a}->{b}"
            if edge in seen_edges_this_pid:
                continue  # a problem shouldn't double-contribute the same edge
            seen_edges_this_pid.add(edge)
            key = (edge, edge_role(a, b, unknown), ptype, gset)
            groups.setdefault(key, set()).add(pid)

    rows = []
    for (edge, role, ptype, gset), pid_set in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2]), str(kv[0][3]))):
        pids = sorted(pid_set)
        a, b = edge.split("->")
        for suffix in ("first_touch", "completion"):
            c = pooled_counts_for_edge(odf, pids, a, b, suffix)
            rows.append({"edge": edge, "role": role,
                        "problem_type": ptype, "given_set": gset,
                        "order_basis": suffix,
                        "n_problems": len(pids), **c})

    df = pd.DataFrame(rows)

    def pct(err, n):
        return f"{100*err/n:.1f}% ({err}/{n})" if n else "-"

    print(f"\n=== {args.lesson} | {args.which} | raw pooled counts, by problem type ===\n")
    for (edge, role, ptype, gset), g in df.groupby(["edge", "role", "problem_type", "given_set"], dropna=False):
        print(f"-- {edge}  [{role}]  type={ptype}  given={gset}  ({g['n_problems'].iloc[0]} problems) --")
        for _, r in g.iterrows():
            print(f"  [{r['order_basis']}] "
                  f"satisfied: n={r['n_satisfied']:>6}  "
                  f"err@A={pct(r['err_satisfied_A'], r['n_scored_satisfied_A'])}  "
                  f"err@B={pct(r['err_satisfied_B'], r['n_scored_satisfied_B'])}")
            print(f"  [{r['order_basis']}] "
                  f"violated:  n={r['n_violated']:>6}  "
                  f"err@A={pct(r['err_violated_A'], r['n_scored_violated_A'])}  "
                  f"err@B={pct(r['err_violated_B'], r['n_scored_violated_B'])}")
        print()

    df.to_csv(out / args.lesson / f"edge_table_with_counts_by_type_{args.which}.csv", index=False)


if __name__ == "__main__":
    main()
