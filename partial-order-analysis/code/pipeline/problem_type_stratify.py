"""
problem_type_stratify.py
------------------------
(1) Fixes metadata dialects that rules.determine_unknown_slot alone can't
    resolve, and (2) builds the stratified problem sample for the
    LLM-partial-order study.

Three metadata dialects, all producing the same 3 fields
(problem_type / unknown_slot / given_set):

  - standard: quantities are in their named columns
    (ppc-scenario_initial-amount, -final-amount, -change-amount,
    -percent-change / -part-amount, -percent, -total-amount).

  - mix4a (ratio_proportion_mix4a-cms-*, 100 rows): ppc-scenario_initial-amount
    holds a SUB-TYPE TAG ('_total-part' | '_percent-total' | '_part-percent')
    instead of a number; the tag names the two given quantities.

  - change4b (ratio_proportion_change4b-cms-*, 100 rows): when percent-change
    is absent, the final-amount is stored in `benchmark` and the
    change-amount is stored in `ppc-scenario_lisp-name` (normally a name
    string) instead of their usual columns.

`given_set` is canonicalized across all three dialects (e.g. 'percent-total'
and 'percent+total-amount' both become 'percent+total') so semantically
identical given-pairs pool together regardless of which dialect produced
them, rather than being split into separate cells.
"""
from __future__ import annotations

import pandas as pd

import rules as R

MIX4A_TAGS = {"_total-part", "_percent-total", "_part-percent"}

MIX4A_TAG_TO_UNKNOWN = {
    "_total-part":    "NumeratorQuantity2",
    "_percent-total": "NumeratorQuantity1",
    "_part-percent":  "DenominatorQuantity1",
}
MIX4A_TAG_TO_TYPE = {
    "_total-part":    "prop_find_percent",
    "_percent-total": "prop_find_part",
    "_part-percent":  "prop_find_total",
}

RULE_TO_TYPE = {
    "rule_problem_Amount":      "change_find_amount",
    "rule_problem_Percentage":  "change_find_percent",
    "rule_problem_Amount_4":    "change_find_amount",
    "rule_problem_Percentage_4": "change_find_percent",
    "rule_problem_part":        "prop_find_part",
    "rule_problem_per":         "prop_find_percent",
    "rule_problem_tot":         "prop_find_total",
}

PROMPT_VARIANT = {
    "change_find_amount":  "A",
    "change_find_percent": "A",
    "prop_find_part":      "B",
    "prop_find_percent":   "B",
    "prop_find_total":     "B",
}

CHANGE_GIVEN_COLS = ["ppc-scenario_initial-amount", "ppc-scenario_final-amount",
                     "ppc-scenario_change-amount", "ppc-scenario_percent-change"]
PROP_GIVEN_COLS = ["ppc-scenario_part-amount", "ppc-scenario_percent",
                   "ppc-scenario_total-amount"]


def _is_mix4a(meta_row) -> bool:
    return R._gettyp(meta_row, "ppc-scenario_initial-amount") in MIX4A_TAGS


def _is_change4b(meta_row) -> bool:
    lms_id = R._gettyp(meta_row, "lms-id")
    return isinstance(lms_id, str) and "change4b" in lms_id


def _canon_given_set(names) -> str:
    """Normalize given-quantity names to a dialect-independent label:
    strip '-amount'/'-change' suffixes and sort, so 'percent-total' (mix4a)
    and 'percent+total-amount' (standard) both become 'percent+total'."""
    canon = set()
    for n in names:
        n = n.replace("-amount", "").replace("percent-change", "percent")
        canon.add(n)
    return "+".join(sorted(canon))


def classify_problem(meta_row, lesson: str) -> dict:
    """problem_type / unknown_slot / given_set for one metadata row."""
    if lesson == "change4" and _is_mix4a(meta_row):
        tag = R._gettyp(meta_row, "ppc-scenario_initial-amount")
        return {"problem_type": MIX4A_TAG_TO_TYPE[tag],
                "unknown_slot": MIX4A_TAG_TO_UNKNOWN[tag],
                "given_set": _canon_given_set(tag.lstrip("_").split("-")),
                "dialect": "mix4a"}

    rule = R.determine_problem_type(meta_row, lesson)
    ptype = RULE_TO_TYPE.get(rule)

    if lesson == "change4" and _is_change4b(meta_row):
        given = set()
        if R.is_present(R._get(meta_row, "ppc-scenario_initial-amount")):
            given.add("initial-amount")
        if R.is_present(R._get(meta_row, "ppc-scenario_percent-change")):
            given.add("percent-change")
        if R.is_present(R._get(meta_row, "ppc-scenario_final-amount")) or \
           R.is_present(R._get(meta_row, "benchmark")):
            given.add("final-amount")
        if R.is_present(R._get(meta_row, "ppc-scenario_change-amount")) or \
           R.is_present(R._get(meta_row, "ppc-scenario_lisp-name")):
            given.add("change-amount")
        return {"problem_type": ptype,
                "unknown_slot": R.determine_unknown_slot(meta_row, lesson),
                "given_set": _canon_given_set(given),
                "dialect": "change4b"}

    cols = PROP_GIVEN_COLS if (ptype or "").startswith("prop_") else CHANGE_GIVEN_COLS
    givens = [c.replace("ppc-scenario_", "") for c in cols
              if R.is_present(R._get(meta_row, c))]
    return {"problem_type": ptype,
            "unknown_slot": R.determine_unknown_slot(meta_row, lesson),
            "given_set": _canon_given_set(givens),
            "dialect": "standard"}


def classify_metadata(meta_df: pd.DataFrame, lesson: str) -> pd.DataFrame:
    out = pd.DataFrame([classify_problem(r, lesson) for _, r in meta_df.iterrows()])
    out.insert(0, "lms-id", meta_df["lms-id"].values)
    out["prompt_variant"] = out["problem_type"].map(PROMPT_VARIANT)
    return out


def audit(meta_df: pd.DataFrame, lesson: str) -> pd.DataFrame:
    c = classify_metadata(meta_df, lesson)
    return (c.groupby(["problem_type", "unknown_slot", "given_set"], dropna=False)
             .size().rename("n_problems").reset_index()
             .sort_values(["problem_type", "n_problems"], ascending=[True, False]))


def sample_templates(meta_df: pd.DataFrame, lesson: str, per_cell: int = 3,
                     min_cell_size: int = 5, seed: int = 0) -> pd.DataFrame:
    c = classify_metadata(meta_df, lesson)
    c = c[c["problem_type"].notna()]
    keep = []
    for key, g in c.groupby(["problem_type", "unknown_slot", "given_set"], dropna=False):
        if len(g) < min_cell_size:
            continue
        keep.append(g.sample(min(per_cell, len(g)), random_state=seed))
    if not keep:
        return c.head(0)
    return pd.concat(keep, ignore_index=True)


if __name__ == "__main__":
    import sys
    from data_loading import load_metadata_csv
    path, lesson = sys.argv[1], sys.argv[2]
    meta = load_metadata_csv(path)
    print(audit(meta, lesson).to_string(index=False))
    print("\n--- sampled templates ---")
    print(sample_templates(meta, lesson).to_string(index=False))