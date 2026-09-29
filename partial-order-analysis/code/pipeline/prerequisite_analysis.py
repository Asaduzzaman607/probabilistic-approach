"""
prerequisite_analysis.py
--------------------------
Loads a JSON file of per-problem PARTIAL ORDER annotations over the 3
quantity steps (an expert/curriculum-author's claim about which order
makes pedagogical sense for a given problem -- e.g. "figure out the
denominator relationship before either numerator"), classifies each
student's actual attempt as FOLLOWING or VIOLATING that annotated order,
and compares error rates between the two groups.

Expected JSON format -- a list of objects:
    {
      "lms-id": "ratio_proportion_change3-009",
      "prerequisites": [
        ["DenominatorQuantity1", "NumeratorQuantity1"],
        ["DenominatorQuantity1", "NumeratorQuantity2"],
        ["NumeratorQuantity1", "NumeratorQuantity2"]
      ]
    }
Each pair [A, B] means "A before B". A problem's prerequisites need not
cover all 3 pairs (a genuine PARTIAL order, not necessarily total) -- e.g.
a problem might only specify DenominatorQuantity1 before NumeratorQuantity1
and say nothing about NumeratorQuantity2's position.

METHODOLOGY, matching the rest of this project's partial-order work
(order_effects_analysis.attach_pairwise_precedence): a constraint (A, B)
is only EVALUABLE for a student if they attempted BOTH A and B -- a
partial-attempt student who touched A and B but never touched a third,
unconstrained-here step still gets evaluated on the constraints that
apply to what they did attempt, rather than being dropped entirely (same
robustness reasoning as before: don't throw away partial-attempt data
when the specific comparison you need doesn't require the missing step).

A student's compliance with a problem's WHOLE prerequisite set is:
  - "followed"       -- every EVALUABLE constraint was satisfied (order>=1)
  - "violated"        -- at least one evaluable constraint was broken
  - "not_evaluable"   -- zero of the annotated constraints were evaluable
                          (e.g. only attempted 1 of the 3 steps)
  - not annotated at all (problem_id has no entry in the JSON) -> excluded
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from data_loading import QUANTITY_STEPNAMES
from order_effects_analysis import _clean_id, _repeated_measures_diagnostics

OUTCOME_COLUMNS = [f"{s}__correct" for s in sorted(QUANTITY_STEPNAMES)] + ["FinalAnswer__correct"]


def _acyclic(pairs: list[tuple[str, str]]) -> bool:
    """True iff `pairs` (each (A, B) meaning A-before-B) admits at least
    one consistent total ordering of the stepnames involved -- i.e. no
    contradictory (A,B)+(B,A) pair and no longer cycle. Only ever 3
    possible stepnames here, so brute-force checking every permutation of
    the involved names is simplest and always fast."""
    import itertools as _it
    names = sorted({n for pair in pairs for n in pair})
    if not names:
        return True
    for perm in _it.permutations(names):
        pos = {n: i for i, n in enumerate(perm)}
        if all(pos[a] < pos[b] for a, b in pairs):
            return True
    return False


def load_prerequisite_annotations(path: str) -> dict[str, list[tuple[str, str]]]:
    """Loads the JSON file described in the module docstring. Returns
    {cleaned_problem_id: [(A, B), ...]}. Validates and reports (rather
    than silently accepting) two classes of malformed entries:
      - a stepname outside QUANTITY_STEPNAMES (typo, or annotation
        references a step this pipeline doesn't model) -- raises, since
        silently ignoring a bad pair could quietly change what "followed"
        means for that problem.
      - a genuinely contradictory/cyclic constraint set (e.g. both
        [A,B] and [B,A] listed, or a 3-cycle) -- raises, since no student
        could ever "follow" an unsatisfiable order and the annotation
        itself needs fixing, not the code.
    Problems with an empty "prerequisites" list are kept (they'll simply
    always resolve to "not_evaluable" downstream) rather than dropped, so
    they still show up in coverage diagnostics."""
    path = Path(path)
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        raw = [raw]  # tolerate a single-object file, not just a list

    out: dict[str, list[tuple[str, str]]] = {}
    for entry in raw:
        pid = entry.get("lms-id")
        if pid is None:
            raise ValueError(f"entry missing 'lms-id': {entry}")
        pairs = []
        for pair in entry.get("prerequisites", []):
            if len(pair) != 2:
                raise ValueError(f"problem '{pid}': prerequisite pair must have exactly 2 "
                                  f"elements, got {pair}")
            a, b = pair
            for step in (a, b):
                if step not in QUANTITY_STEPNAMES:
                    raise ValueError(
                        f"problem '{pid}': prerequisite step '{step}' is not one of "
                        f"{sorted(QUANTITY_STEPNAMES)} -- check for a typo in the annotation file."
                    )
            pairs.append((a, b))
        if not _acyclic(pairs):
            raise ValueError(f"problem '{pid}': prerequisite pairs {pairs} are contradictory/"
                              f"cyclic -- no ordering could satisfy all of them simultaneously.")
        out[_clean_id(pid)] = pairs
    return out


def attach_prerequisite_compliance(records: pd.DataFrame,
                                    prereqs: dict[str, list[tuple[str, str]]]) -> pd.DataFrame:
    """Adds columns: prereq_annotated (bool), n_prereqs_total,
    n_prereqs_evaluable, n_prereqs_violated, prereq_compliance (one of
    'followed'/'violated'/'not_evaluable'/None -- None means this
    problem_id has no entry in `prereqs` at all)."""
    out = records.copy()
    cols = ["prereq_annotated", "n_prereqs_total", "n_prereqs_evaluable",
            "n_prereqs_violated", "prereq_compliance"]
    if len(out) == 0:
        for c in cols:
            out[c] = pd.Series(dtype=object)
        return out

    def _compliance(row):
        pid = _clean_id(row["problem_id"])
        pairs = prereqs.get(pid)
        if pairs is None:
            return (False, 0, 0, 0, None)
        n_total = len(pairs)
        n_eval = 0
        n_violated = 0
        for a, b in pairs:
            oa, ob = row.get(f"{a}__first_order"), row.get(f"{b}__first_order")
            if pd.isna(oa) or pd.isna(ob):
                continue
            n_eval += 1
            if not (oa < ob):
                n_violated += 1
        if n_eval == 0:
            status = "not_evaluable"
        elif n_violated == 0:
            status = "followed"
        else:
            status = "violated"
        return (True, n_total, n_eval, n_violated, status)

    computed = out.apply(_compliance, axis=1, result_type="expand")
    computed.columns = cols
    for c in cols:
        out[c] = computed[c]
    out["n_prereqs_total"] = out["n_prereqs_total"].astype(int)
    out["n_prereqs_evaluable"] = out["n_prereqs_evaluable"].astype(int)
    out["n_prereqs_violated"] = out["n_prereqs_violated"].astype(int)
    return out


def _error_columns(g: pd.DataFrame) -> dict:
    row = {}
    known_mask = pd.Series(False, index=g.index)
    any_error_mask = pd.Series(False, index=g.index)
    for col in OUTCOME_COLUMNS:
        vals = g[col].dropna()
        row[f"{col.replace('__correct','')}_error_rate"] = float(1 - vals.mean()) if len(vals) else None
        row[f"{col.replace('__correct','')}_n"] = int(len(vals))
    # any_quantity_error: among the 3 QUANTITY steps only (not FinalAnswer)
    # -- 1 if any known quantity-step correctness for that row is 0, among
    # rows with at least one known quantity-step correctness value.
    quantity_cols = [f"{s}__correct" for s in sorted(QUANTITY_STEPNAMES)]
    has_any_known = g[quantity_cols].notna().any(axis=1)
    any_wrong = (g[quantity_cols] == 0).any(axis=1)
    sub = any_wrong[has_any_known]
    row["any_quantity_error_rate"] = float(sub.mean()) if len(sub) else None
    row["any_quantity_error_n"] = int(len(sub))
    return row


def _compliance_error_table(records: pd.DataFrame, group_col: str | None) -> pd.DataFrame:
    """group_col=None pools across ALL annotated problems into a single
    followed-vs-violated comparison; group_col='problem_id' gives one
    comparison per problem."""
    sub = records[records["prereq_compliance"].isin(["followed", "violated"])]
    group_keys = ["prereq_compliance"] if group_col is None else [group_col, "prereq_compliance"]
    rows = []
    for key, g in sub.groupby(group_keys):
        # pandas groupby(list_of_columns) always yields a tuple key, even
        # for a single-element list -- no special-casing needed here.
        row = dict(zip(group_keys, key))
        row["n_observations"] = len(g)
        row["n_unique_students"] = int(g["student_id"].nunique())
        row["n_problems"] = int(g["problem_id"].nunique())
        row.update(_error_columns(g))
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result):
        result = result.sort_values(group_keys).reset_index(drop=True)
    return result


def prerequisite_error_table(records: pd.DataFrame) -> pd.DataFrame:
    """Per-problem: for followed vs. violated, error rate on each of the
    3 quantity steps, FinalAnswer, and a combined 'any_quantity_error'
    summary. Requires `records` to have gone through
    attach_prerequisite_compliance first."""
    if "prereq_compliance" not in records.columns:
        raise ValueError("records has no 'prereq_compliance' column -- run "
                          "attach_prerequisite_compliance() first")
    return _compliance_error_table(records, group_col="problem_id")


def pooled_prerequisite_error_table(records: pd.DataFrame) -> pd.DataFrame:
    """Same as prerequisite_error_table but pooled across ALL annotated
    problems into one followed-vs-violated comparison (n_problems shows
    how many distinct problems contributed). See
    order_effects_analysis.pooled_ordering_error_table's docstring for
    why n_observations != n_unique_students matters once pooled -- same
    reasoning applies here (a student can be 'followed' on one problem
    and 'violated' on another)."""
    if "prereq_compliance" not in records.columns:
        raise ValueError("records has no 'prereq_compliance' column -- run "
                          "attach_prerequisite_compliance() first")
    return _compliance_error_table(records, group_col=None)


def compliance_coverage_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Per problem_id: how many students landed in each compliance bucket
    (followed / violated / not_evaluable / not annotated) -- a sanity
    check on how much of the population a prerequisite comparison
    actually rests on before trusting its p-value."""
    if "prereq_compliance" not in records.columns:
        raise ValueError("records has no 'prereq_compliance' column -- run "
                          "attach_prerequisite_compliance() first")
    rows = []
    for pid, g in records.groupby("problem_id"):
        col = g["prereq_compliance"]
        rows.append({
            "problem_id": pid,
            "n_followed": int((col == "followed").sum()),
            "n_violated": int((col == "violated").sum()),
            "n_not_evaluable": int((col == "not_evaluable").sum()),
            "n_not_annotated": int(col.isna().sum()),
        })
    return pd.DataFrame(rows)


def _test_compliance(records: pd.DataFrame, outcome_col: str, group_col: str | None,
                      group_value=None) -> dict:
    """2x2 Fisher's exact test: is `outcome_col` (a *_correct column, or
    'any_quantity_error') associated with followed-vs-violated compliance?
    group_col=None + group_value=None -> pooled across all annotated
    problems. group_col='problem_id' + group_value=<id> -> one problem."""
    sub = records[records["prereq_compliance"].isin(["followed", "violated"])]
    if group_col is not None:
        sub = sub[sub[group_col] == group_value]

    if outcome_col == "any_quantity_error":
        quantity_cols = [f"{s}__correct" for s in sorted(QUANTITY_STEPNAMES)]
        has_any_known = sub[quantity_cols].notna().any(axis=1)
        any_wrong = (sub[quantity_cols] == 0).any(axis=1)
        outcome = (~any_wrong).astype(int)  # 1 = no error (i.e. "correct"), for symmetry with *_correct cols
        outcome = outcome.where(has_any_known)
    else:
        outcome = sub[outcome_col]

    eval_sub = pd.DataFrame({"compliance": sub["prereq_compliance"], "outcome": outcome,
                              "student_id": sub["student_id"]}).dropna(subset=["outcome"])
    if eval_sub["compliance"].nunique() < 2 or len(eval_sub) == 0:
        label = group_value if group_col is not None else "pooled"
        return {"scope": label, "outcome": outcome_col, "note": "fewer than 2 compliance groups observed"}

    table = pd.crosstab(eval_sub["compliance"], eval_sub["outcome"].map({1: "correct", 0: "error"}))
    for c in ("correct", "error"):
        if c not in table.columns:
            table[c] = 0
    for r in ("followed", "violated"):
        if r not in table.index:
            table.loc[r] = [0, 0]
    table = table.loc[["followed", "violated"], ["error", "correct"]]

    odds_ratio, p = fisher_exact(table.to_numpy())
    label = group_value if group_col is not None else "pooled"
    # _repeated_measures_diagnostics only suppresses its warning when passed
    # the literal string "problem_id" (correct here: a single problem_id
    # always has 1 observation per student, so the warning would never fire
    # anyway). For the pooled case (group_col=None) we pass a different
    # sentinel so the warning CAN fire when the same student appears under
    # both compliance labels across different problems.
    diag_group_col = "problem_id" if group_col == "problem_id" else "problem_type"
    return {"scope": label, "outcome": outcome_col, "test": "fisher_exact",
            "odds_ratio": odds_ratio, "p_value": p, "table": table,
            **_repeated_measures_diagnostics(eval_sub, diag_group_col)}


def test_prerequisite_compliance(records: pd.DataFrame, problem_id: str,
                                  outcome_col: str = "any_quantity_error") -> dict:
    """Fisher's exact test for ONE problem: followed vs. violated,
    compared on `outcome_col` (default: any_quantity_error; can also be
    'FinalAnswer__correct' or any '{step}__correct' column)."""
    return _test_compliance(records, outcome_col, group_col="problem_id", group_value=problem_id)


def test_pooled_prerequisite_compliance(records: pd.DataFrame,
                                         outcome_col: str = "any_quantity_error") -> dict:
    """Same test, pooled across every annotated problem at once."""
    return _test_compliance(records, outcome_col, group_col=None, group_value=None)
