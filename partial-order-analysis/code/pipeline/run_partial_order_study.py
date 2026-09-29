"""
run_partial_order_study.py
--------------------------
Driver for the whole study, in the order the pieces have to happen:

  step1_select      metadata -> stratified template sample (+ target ids file)
  step2_annotate    problem text -> LLM -> prerequisites.json  (k samples, voted)
  step3_analyze     logs + prerequisites.json -> coverage, error tables, tests

Everything downstream of step2 reuses your existing modules unchanged
(order_effects_analysis, prerequisite_analysis, rules, data_loading).

The baselines in step3 are the point of the whole design: `followed vs
violated` on its own can't distinguish "the LLM found a good order" from
"good students produce canonical orders". Each baseline is just a different
prerequisite dict pushed through the SAME compliance machinery, so the
comparison is apples-to-apples:

  llm         the model's per-problem partial order
  unknown_last  every given quantity before the unknown one (a rule with no
                LLM in it -- if this explains the effect, the LLM adds nothing)
  modal         the most common complete ordering students actually used for
                that problem (compliance here is "did what everyone else did")

If the LLM column doesn't beat these, the finding isn't about LLMs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import order_effects_analysis as OE
import prerequisite_analysis as PA
import rules as R
from data_loading import QUANTITY_STEPNAMES, load_log_csv, load_metadata_csv
from problem_type_stratify import audit, classify_metadata, sample_templates

QUANTITY_LIST = sorted(QUANTITY_STEPNAMES)


# ---------------------------------------------------------------- step 1 ---
def step1_select(meta_path: str, lesson: str, out_dir: str,
                 per_cell: int = 3, min_cell_size: int = 5, seed: int = 0):
    """Stratified sample over (problem_type, unknown_slot, given_set)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_metadata_csv(meta_path)

    grid = audit(meta, lesson)
    grid.to_csv(out_dir / f"template_grid_{lesson}.csv", index=False)

    sample = sample_templates(meta, lesson, per_cell=per_cell,
                              min_cell_size=min_cell_size, seed=seed)
    text_cols = [c for c in ("problem-text", "question-text") if c in meta.columns]
    sample = sample.merge(meta[["lms-id"] + text_cols], on="lms-id", how="left")
    sample.to_csv(out_dir / f"template_sample_{lesson}.csv", index=False)
    (out_dir / f"target_problem_ids_{lesson}.txt").write_text(
        "\n".join(sample["lms-id"].tolist()) + "\n")

    unclassified = classify_metadata(meta, lesson)["problem_type"].isna().sum()
    print(f"[step1] {lesson}: {len(meta)} problems, {len(grid)} cells, "
          f"{len(sample)} sampled, {unclassified} unclassified")
    return sample


# ---------------------------------------------------------------- step 2 ---
PROMPTS = Path(__file__).with_name("llm_partial_order_prompts.md")

SLOT_SEMANTICS = {
    "A": ("For this problem the proportion is\n\n"
          "    change amount / original amount = percent change / 100\n\n"
          "so:\n"
          "    NumeratorQuantity1   = the amount of change (final - original, in magnitude)\n"
          "    DenominatorQuantity1 = the original (baseline) amount\n"
          "    NumeratorQuantity2   = the percent change"),
    "B": ("For this problem the proportion is\n\n"
          "    part / total = percent / 100\n\n"
          "so:\n"
          "    NumeratorQuantity1   = the part (the subset being described)\n"
          "    DenominatorQuantity1 = the total (the whole)\n"
          "    NumeratorQuantity2   = the percent that the part represents of the total"),
}

SYSTEM_TEMPLATE = """You are analyzing how a middle-school student should set up a proportional-reasoning word problem in an intelligent tutor.

The tutor asks the student to fill three entry fields that form the proportion

    NumeratorQuantity1 / DenominatorQuantity1 = NumeratorQuantity2 / 100

{slot_semantics}

Exactly one of the three quantities is not given directly in the problem text; the student still has to enter something in that field, so it is part of the ordering.

Your goal is NOT to solve the problem. Your goal is to determine the order in which a student should identify and enter these three quantities so as to MINIMIZE ERRORS.

Rules:
- Do not assume quantities should be processed in the order they appear in the text. Derive the order from the semantic and mathematical structure.
- Give a PARTIAL order, not a total order. State a constraint [X, Y] ("X must be entered before Y") only when getting X wrong or leaving it undecided would plausibly cause an error in Y. If two fields are independent, state no constraint between them.
- The constraint set must be acyclic.

Return ONLY JSON, no prose, no markdown fences:

{{"lms-id": "<the id you were given>", "prerequisites": [["<slot>", "<slot>"]], "rationale": "<2-3 sentences>"}}

Valid slot names: NumeratorQuantity1, DenominatorQuantity1, NumeratorQuantity2."""

USER_TEMPLATE = """lms-id: {lms_id}

Problem:
{problem_text}

Question asked of the student:
{question_text}"""


def build_prompt(row) -> tuple[str, str]:
    variant = row["prompt_variant"]
    system = SYSTEM_TEMPLATE.format(slot_semantics=SLOT_SEMANTICS[variant])
    user = USER_TEMPLATE.format(lms_id=row["lms-id"],
                                problem_text=row.get("problem-text", ""),
                                question_text=row.get("question-text", "") or "")
    return system, user


def _parse_pairs(text: str) -> list[tuple[str, str]]:
    """Tolerant parse: strip fences, take the outermost {...}."""
    t = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    start, end = t.find("{"), t.rfind("}")
    obj = json.loads(t[start:end + 1])
    pairs = [tuple(p) for p in obj["prerequisites"]]
    bad = [s for p in pairs for s in p if s not in QUANTITY_STEPNAMES]
    if bad:
        raise ValueError(f"invalid slot name(s): {bad}")
    if not PA._acyclic(pairs):
        raise ValueError(f"cyclic constraint set: {pairs}")
    return pairs


def step2_annotate(sample: pd.DataFrame, llm_fn, out_path: str, k: int = 5) -> pd.DataFrame:
    """llm_fn(system, user) -> str. Calls it k times per problem, keeps the
    modal constraint set, writes prerequisites.json, and returns a stability
    table (agreement_rate = share of the k samples matching the mode). Report
    that table -- an order that flips between samples is not teachable."""
    records, stability = [], []
    for _, row in sample.iterrows():
        system, user = build_prompt(row)
        votes, raw_ok = [], 0
        for _ in range(k):
            try:
                pairs = _parse_pairs(llm_fn(system, user))
                votes.append(tuple(sorted(pairs)))
                raw_ok += 1
            except Exception as e:                      # noqa: BLE001
                stability.append({"lms-id": row["lms-id"], "error": str(e)[:120]})
        if not votes:
            continue
        counts = pd.Series(votes).value_counts()
        mode = counts.index[0]
        records.append({"lms-id": row["lms-id"],
                        "prerequisites": [list(p) for p in mode],
                        "prompt_variant": row["prompt_variant"]})
        stability.append({"lms-id": row["lms-id"], "n_valid": raw_ok,
                          "n_distinct": int(counts.size),
                          "agreement_rate": float(counts.iloc[0] / len(votes)),
                          "n_constraints": len(mode)})
    Path(out_path).write_text(json.dumps(records, indent=2))
    PA.load_prerequisite_annotations(out_path)   # validate what we just wrote
    return pd.DataFrame(stability)


# ---------------------------------------------------------------- step 3 ---
def unknown_last_prereqs(meta_df: pd.DataFrame, lesson: str,
                         problem_ids: list[str]) -> dict:
    """Baseline: every given quantity before the unknown one."""
    out = {}
    by_id = {OE._clean_id(r["lms-id"]): r for _, r in meta_df.iterrows()}
    for pid in problem_ids:
        row = by_id.get(OE._clean_id(pid))
        if row is None:
            continue
        unknown = R.determine_unknown_slot(row, lesson)
        if unknown is None:
            continue
        out[OE._clean_id(pid)] = [(s, unknown) for s in QUANTITY_LIST if s != unknown]
    return out


def modal_order_prereqs(records: pd.DataFrame) -> dict:
    """Baseline: the most common COMPLETE ordering students used, expanded
    to its 3 implied pairs."""
    out = {}
    complete = records[records["is_complete"]]
    for pid, g in complete.groupby("problem_id"):
        if g.empty:
            continue
        order = g["ordering"].value_counts().index[0]
        out[OE._clean_id(pid)] = [(order[i], order[j])
                                  for i in range(3) for j in range(i + 1, 3)]
    return out


def bh_fdr(pvals: list[float], alpha: float = 0.05) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    n, order = len(p), np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        prev = min(prev, p[idx] * n / (n - rank))
        q[idx] = prev
    return q.tolist()


def within_student_test(records: pd.DataFrame, outcome_col: str = "any_quantity_error") -> dict:
    """Paired comparison restricted to students observed under BOTH labels
    across different problems -- the cheapest control for the ability
    confound, since each student is their own baseline.

    outcome_col: 'any_quantity_error' (default -- computed from the 3
    quantity slots) or any '{step}__correct' column (e.g.
    'FinalAnswer__correct'), matching what step3_analyze's outcome
    parameter is tested on. Previously this was hardcoded to always use
    any_quantity_error regardless of what step3_analyze was asked to
    test -- fixed so within-student and pooled tests always agree on
    which outcome they're measuring."""
    sub = records[records["prereq_compliance"].isin(["followed", "violated"])].copy()
    if outcome_col == "any_quantity_error":
        qcols = [f"{s}__correct" for s in QUANTITY_LIST]
        sub["_outcome_error"] = (sub[qcols] == 0).any(axis=1).where(sub[qcols].notna().any(axis=1))
    else:
        if outcome_col not in sub.columns:
            return {"note": f"outcome column '{outcome_col}' not present"}
        sub["_outcome_error"] = (sub[outcome_col] == 0).where(sub[outcome_col].notna())
    sub = sub.dropna(subset=["_outcome_error"])
    piv = (sub.groupby(["student_id", "prereq_compliance"])["_outcome_error"]
              .mean().unstack())
    both = piv.dropna()
    if len(both) < 5:
        return {"n_students_both": len(both), "note": "too few students seen under both labels"}
    stat, p = wilcoxon(both["followed"], both["violated"])
    return {"n_students_both": int(len(both)),
            "mean_error_followed": float(both["followed"].mean()),
            "mean_error_violated": float(both["violated"].mean()),
            "wilcoxon_stat": float(stat), "p_value": float(p)}


def step3_analyze(log_path: str, meta_path: str, lesson: str, prereq_json: str,
                  out_dir: str, outcome: str = "any_quantity_error") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = load_log_csv(log_path, "change")
    meta = load_metadata_csv(meta_path)

    records = OE.build_records_all_problems(log, meta, lesson)
    prereq_sets = {
        "llm": PA.load_prerequisite_annotations(prereq_json),
    }
    annotated_ids = list(prereq_sets["llm"].keys())
    prereq_sets["unknown_last"] = unknown_last_prereqs(meta, lesson, annotated_ids)
    prereq_sets["modal"] = {k: v for k, v in modal_order_prereqs(records).items()
                            if k in prereq_sets["llm"]}

    results = {}
    for name, prereqs in prereq_sets.items():
        rec = PA.attach_prerequisite_compliance(records, prereqs)
        coverage = PA.compliance_coverage_summary(rec)
        pooled = PA.pooled_prerequisite_error_table(rec)
        per_problem = PA.prerequisite_error_table(rec)

        tests = []
        for pid in sorted(set(rec.loc[rec["prereq_compliance"].notna(), "problem_id"])):
            r = PA.test_prerequisite_compliance(rec, pid, outcome_col=outcome)
            if "p_value" in r:
                tests.append({"problem_id": pid, "odds_ratio": r["odds_ratio"],
                              "p_value": r["p_value"]})
        tests = pd.DataFrame(tests)
        if len(tests):
            tests["q_value"] = bh_fdr(tests["p_value"].tolist())

        pooled_test = PA.test_pooled_prerequisite_compliance(rec, outcome_col=outcome)
        pooled_test.pop("table", None)

        coverage.to_csv(out_dir / f"coverage_{name}.csv", index=False)
        per_problem.to_csv(out_dir / f"error_table_{name}.csv", index=False)
        tests.to_csv(out_dir / f"tests_{name}.csv", index=False)

        results[name] = {"pooled_error_table": pooled, "pooled_test": pooled_test,
                         "per_problem_tests": tests,
                         "within_student": within_student_test(rec, outcome_col=outcome)}
        print(f"[step3] {name}: pooled p={pooled_test.get('p_value')} "
              f"OR={pooled_test.get('odds_ratio')} "
              f"| n_sig(q<.05)={int((tests['q_value'] < .05).sum()) if len(tests) else 0}"
              f"/{len(tests)} | within-student={results[name]['within_student']}")
    return results
