"""
order_effects_analysis.py
---------------------------
For specific problems (given by problem_id, loaded from a file -- see
load_target_problem_ids), investigates whether the ORDER in which a
student first attempts the three quantity-entry steps

    NumeratorQuantity1, NumeratorQuantity2, DenominatorQuantity1
    (ratio: NumeratorQuantity1 / DenominatorQuantity1 = NumeratorQuantity2 / 100)

is associated with different per-step error rates -- e.g. does making
DenominatorQuantity1 the LAST step you touch (instead of first) change how
often students get it wrong, and does this differ depending on which of
the three is actually the unknown for that problem (rules.determine_unknown_slot)?

Scope, per your last message: PROBLEM-SPECIFIC (comprehension of a
particular problem's text/structure, not pooled across problems) -- so
every table here is keyed by problem_id, built from a caller-supplied list
of problem_ids (load_target_problem_ids), not "all change3/4 problems."

Definitions (matching the rest of this pipeline, nothing new):
  - "first attempted" = the position (StepToken.order, i.e. the token's
    index within that student's step_sequence) of that step's FIRST
    'Attempt' action -- NOT the first hint request, and not attempt
    number (a step can be attempted more than once; we only care about
    when the student first engaged with it).
  - "error" on a step = data_loading._step_correct via labeling.step_correct
    returns 0 (first attempt was not OK, or an earlier detailed hint was
    used) -- the same correctness rule used everywhere else in this
    pipeline. Returns None (excluded, not counted as an error) if the
    student never attempted that step at all.
  - "ordering" for a student on a problem = the tuple of the 3 stepnames
    sorted by first-attempted order. Only defined (a student contributes
    to this analysis) if the student attempted ALL THREE steps at least
    once -- a partial ordering (e.g. only 2 of 3 touched) isn't a
    comparable data point for "does full attempt order matter," so those
    rows are counted separately (`n_partial`) rather than silently merged
    into some ordering bucket.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

from data_loading import parse_step_sequence, QUANTITY_STEPNAMES, FINAL_ANSWER_STEP
from labeling import step_correct
import rules as R

ALL_ORDERINGS = list(itertools.permutations(sorted(QUANTITY_STEPNAMES)))


def _clean_id(x) -> str:
    """Normalize an id for comparison: stringify + strip ordinary AND
    Unicode whitespace (str.strip() already covers \\xa0/non-breaking
    space -- Python treats it as whitespace). Deliberately does NOT do
    anything fancier (case-folding, stripping a trailing '.0', collapsing
    internal whitespace) -- those are exactly the kind of silent, "helpful"
    coercion that can quietly merge two actually-different ids. Use
    diagnose_id_mismatch() to SEE what's different rather than guess-fixing
    it here."""
    return str(x).strip()


def diagnose_id_mismatch(candidate_id, present_ids, max_suggestions: int = 5) -> dict:
    """When `candidate_id` isn't found in `present_ids` (an iterable of the
    actual problem_id/lms-id values from your dataframe) even though it
    looks like it should be, this shows WHY rather than making you guess:
    checks the usual silent-mismatch culprits one at a time (plain
    whitespace, a pandas float-cast '.0' suffix, case, a hidden non-ASCII
    character invisible when printed) and reports which ones would have
    fixed it, plus the closest actual matches by string similarity so you
    can eyeball the real difference (e.g. via repr() showing a stray
    '\\u200b' zero-width space that print() hides)."""
    raw = str(candidate_id)
    present = [str(p) for p in present_ids]
    present_set = set(present)

    checks = {
        "exact_match": raw in present_set,
        "stripped_match": raw.strip() in {p.strip() for p in present},
        "casefold_match": raw.strip().casefold() in {p.strip().casefold() for p in present},
        "float_suffix_match": raw.strip().removesuffix(".0") in
            {p.strip().removesuffix(".0") for p in present},
    }

    import difflib
    close = difflib.get_close_matches(raw, present, n=max_suggestions, cutoff=0.6)

    return {
        "candidate_id": raw,
        "candidate_repr": repr(raw),  # repr() surfaces hidden chars print() would swallow
        "candidate_len": len(raw),
        "checks": checks,
        "any_fix_works": any(checks.values()),
        "closest_matches": [{"value": c, "repr": repr(c), "len": len(c)} for c in close],
        "n_present_total": len(present),
    }


def split_target_ids(target_ids: list[str], sources: dict[str, pd.DataFrame],
                      column: str = "problem_id") -> dict:
    """A single flat problem_id list often mixes ids that only exist in
    change-3 with ids that only exist in change-4 (they're different
    lessons -- a given problem is normally in exactly one of them). This
    splits `target_ids` per source using the SAME whitespace-robust
    _clean_id matching as build_records -- a naive
    `id in set(df['problem_id'])` check (unstripped) can silently drop an
    id that IS genuinely present, which then looks identical to "this id
    doesn't exist in either lesson" and produces a confusing empty-match
    error two functions later.

    Returns {source_name: [matched ids from target_ids, original
    formatting preserved], ..., '__unmatched__': [ids found in NONE of
    the sources]}. An id matching MORE than one source is included in
    every source it matched -- also worth a second look, since change3/
    change4 problem ids aren't expected to collide."""
    cleaned_sets = {name: {_clean_id(p) for p in df[column]} for name, df in sources.items()}
    result: dict[str, list] = {name: [] for name in sources}
    unmatched = []
    matched_in = {}
    for tid in target_ids:
        ct = _clean_id(tid)
        hits = [name for name, cset in cleaned_sets.items() if ct in cset]
        if hits:
            for name in hits:
                result[name].append(tid)
            matched_in[tid] = hits
        else:
            unmatched.append(tid)
    result["__unmatched__"] = unmatched
    result["__matched_in_multiple__"] = {tid: hits for tid, hits in matched_in.items() if len(hits) > 1}
    return result


def load_target_problem_ids(path: str) -> list[str]:
    """Loads a target problem_id list from either:
      - a .csv with a 'problem_id' (or 'lms-id'/'name') column, or
      - a plain .txt with one problem_id per line (blank lines/# comments ignored).
    Returns problem_ids in file order, de-duplicated."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str)
        col = next((c for c in ("problem_id", "lms-id", "name") if c in df.columns), None)
        if col is None:
            raise ValueError(f"{path}: no problem_id/lms-id/name column found (got {list(df.columns)})")
        ids = df[col].dropna().astype(str).str.strip().tolist()
    else:
        ids = [line.strip() for line in path.read_text().splitlines()
               if line.strip() and not line.strip().startswith("#")]
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def student_problem_record(row: pd.Series) -> dict | None:
    """For one (student, problem) log row: returns a dict with the
    first-attempted order of each of the 3 quantity steps and their
    correctness, or None if step_sequence is empty/unparseable. Steps
    never attempted get order=None, correct=None. Also captures
    FinalAnswer correctness (FinalAnswer__correct; no __first_order,
    since it isn't one of the 3 quantity steps orderings are computed
    over) -- the natural downstream outcome for "did violating a
    recommended step order actually hurt the result," not just the
    quantity-entry steps themselves."""
    tokens = parse_step_sequence(row.get("step_sequence", ""))
    if not tokens:
        return None

    out = {"student_id": row["student_id"], "problem_id": row["problem_id"],
           "school_id": row.get("school_id"), "problem_order": row.get("problem_order", np.nan)}
    for stepname in sorted(QUANTITY_STEPNAMES):
        step_tokens = [t for t in tokens if t.stepname == stepname]
        attempts = [t for t in step_tokens if t.action == "Attempt"]
        first_order = min((t.order for t in attempts), default=None)
        out[f"{stepname}__first_order"] = first_order
        out[f"{stepname}__correct"] = step_correct(tokens, stepname)
    out["FinalAnswer__correct"] = step_correct(tokens, FINAL_ANSWER_STEP)
    return out


REQUIRED_RECORD_COLUMNS = ["student_id", "problem_id", "school_id", "problem_order"] + \
    [f"{s}__first_order" for s in sorted(QUANTITY_STEPNAMES)] + \
    [f"{s}__correct" for s in sorted(QUANTITY_STEPNAMES)] + ["FinalAnswer__correct"]


def build_records(df: pd.DataFrame, problem_ids: list[str]) -> pd.DataFrame:
    """df: a change3/4 log dataframe (one row per student x problem
    attempt, with step_sequence -- e.g. change3_merged/change4_merged
    already metadata-joined so unknown_slot can be attached by the
    caller). Filters to `problem_ids` and builds one row per (student,
    problem) with each quantity step's first-attempt order + correctness.

    Matches ids via _clean_id (stringify + strip whitespace) on BOTH sides
    -- df['problem_id'] and the requested list -- since a mismatch is
    often whitespace on the dataframe side that load_target_problem_ids
    already stripped from the FILE side, making a plain '==' comparison
    silently fail even though the ids look identical when printed.

    An EMPTY `problem_ids` is NOT an error -- it's the normal outcome of
    split_target_ids() when every target id happens to belong to the
    other lesson (e.g. ids3=[] because your whole target list is change-4
    problems). Quietly returns an empty (well-formed) table in that case,
    so callers concatenating rec3/rec4 across lessons don't need to
    special-case "no ids for this lesson" themselves.

    Raises ValueError (rather than silently returning an empty table that
    breaks downstream on a confusing pandas empty-DataFrame quirk) only
    when `problem_ids` is NON-empty but NONE of them appear in
    df['problem_id'] -- that combination is a genuine matching problem,
    not a legitimate empty selection. Includes a diagnose_id_mismatch()
    report for the first requested id so you don't have to go dig for the
    cause separately."""
    if "problem_id" not in df.columns:
        raise ValueError(f"df has no 'problem_id' column (columns: {list(df.columns)})")
    if not problem_ids:
        return pd.DataFrame(columns=REQUIRED_RECORD_COLUMNS)
    present_raw = df["problem_id"].tolist()
    present_clean = {_clean_id(p) for p in present_raw}
    requested_clean = {_clean_id(p) for p in problem_ids}
    matched = present_clean & requested_clean
    if not matched:
        first_requested = next(iter(problem_ids), None)
        diag = diagnose_id_mismatch(first_requested, present_raw) if first_requested is not None else None
        raise ValueError(
            f"None of the {len(requested_clean)} requested problem_ids were found in "
            f"df['problem_id'] (checked after stripping whitespace on both sides). "
            f"Requested (sample): {sorted(requested_clean)[:5]}. "
            f"Actually present in df (sample): {sorted(present_clean)[:5]}.\n"
            f"Diagnostic for '{first_requested}': {diag}\n"
            "If 'closest_matches' above shows something that LOOKS identical but isn't, "
            "compare the repr() strings character-by-character -- common causes beyond "
            "whitespace are a pandas float-cast '.0' suffix (id read as a number, not a "
            "string), a hidden non-breaking/zero-width character, or case. Also double-check "
            "you passed the right lesson's dataframe (change3 vs change4)."
        )
    sub = df[df["problem_id"].map(_clean_id).isin(requested_clean)]
    rows = [r for r in (student_problem_record(row) for _, row in sub.iterrows()) if r is not None]
    if not rows:
        # matched on problem_id, but every row's step_sequence was empty/unparseable
        return pd.DataFrame(columns=REQUIRED_RECORD_COLUMNS)
    return pd.DataFrame(rows)


def attach_ordering(records: pd.DataFrame) -> pd.DataFrame:
    """Adds an `ordering` column (tuple of the 3 stepnames, sorted by
    first-attempt order) for rows where all 3 steps were attempted, and an
    `is_complete` boolean flag. Rows with is_complete=False have
    ordering=None (partial -- see module docstring)."""
    out = records.copy()
    if len(out) == 0:
        # DataFrame.apply(axis=1) on an EMPTY frame returns another empty
        # DataFrame, not a Series -- assigning that to a column raises
        # "Cannot set a DataFrame without columns to the column 'ordering'".
        # Handle the 0-row case explicitly instead of hitting that.
        out["ordering"] = pd.Series(dtype=object)
        out["is_complete"] = pd.Series(dtype=bool)
        return out

    def _ordering(row):
        vals = {s: row[f"{s}__first_order"] for s in sorted(QUANTITY_STEPNAMES)}
        if any(pd.isna(v) for v in vals.values()):
            return None
        return tuple(sorted(vals, key=lambda s: vals[s]))

    out["ordering"] = out.apply(_ordering, axis=1)
    out["is_complete"] = out["ordering"].notna()
    return out


def _attach_meta_derived_column(records: pd.DataFrame, meta_df: pd.DataFrame, lesson: str,
                                 id_col: str, out_col: str, derive_fn) -> pd.DataFrame:
    """Shared join logic behind attach_unknown_slot/attach_problem_type:
    for each problem_id in `records`, look up its metadata row in
    `meta_df` (matched via _clean_id on both sides, same whitespace
    reasoning as build_records) and compute `derive_fn(meta_row, lesson)`
    into a new column `out_col`. Raises if EVERY problem_id fails to join
    (near-certainly an id/id_col mismatch, not "no rule applies to any of
    these") rather than silently returning an all-None column."""
    out = records.copy()
    value_by_problem = {}
    for _, meta_row in meta_df.iterrows():
        pid = meta_row.get(id_col)
        if pid is None or (isinstance(pid, float) and np.isnan(pid)):
            continue
        value_by_problem[_clean_id(pid)] = derive_fn(meta_row, lesson)
    if len(out) == 0:
        out[out_col] = pd.Series(dtype=object)
        return out
    out[out_col] = out["problem_id"].map(lambda p: value_by_problem.get(_clean_id(p)))
    unmatched = out.loc[out[out_col].isna(), "problem_id"].unique()
    if len(unmatched) > 0 and len(unmatched) == out["problem_id"].nunique():
        sample = str(unmatched[0])
        diag = diagnose_id_mismatch(sample, meta_df[id_col].tolist())
        raise ValueError(
            f"attach failed for column '{out_col}': NONE of the {len(unmatched)} problem_id(s) "
            f"in `records` matched meta_df['{id_col}'] (checked after stripping whitespace on "
            f"both sides). Diagnostic for '{sample}': {diag}\n"
            f"Double-check id_col='{id_col}' is the right metadata column for this lesson."
        )
    return out


def attach_unknown_slot(records: pd.DataFrame, meta_df: pd.DataFrame, lesson: str,
                         id_col: str = "lms-id") -> pd.DataFrame:
    """Adds an `unknown_slot` column (which of the 3 steps is THE unknown
    for that problem, per rules.determine_unknown_slot) by joining
    `meta_df` on problem_id == meta_df[id_col]. lesson in
    {'change3','change4'}. Matches via _clean_id on both sides (same
    whitespace-mismatch reasoning as build_records)."""
    return _attach_meta_derived_column(records, meta_df, lesson, id_col,
                                        "unknown_slot", R.determine_unknown_slot)


def attach_problem_type(records: pd.DataFrame, meta_df: pd.DataFrame, lesson: str,
                         id_col: str = "lms-id") -> pd.DataFrame:
    """Adds a `problem_type` column -- the NAME of the rule that determined
    this problem's unknown slot (e.g. 'rule_problem_Amount',
    'rule_problem_tot'; see rules.determine_problem_type). This is the
    grouping key for the POOLED (across all problems of the same type)
    analysis functions below, as opposed to the per-problem-id ones."""
    return _attach_meta_derived_column(records, meta_df, lesson, id_col,
                                        "problem_type", R.determine_problem_type)


def build_records_all_problems(df: pd.DataFrame, meta_df: pd.DataFrame, lesson: str,
                                id_col: str = "lms-id") -> pd.DataFrame:
    """Convenience wrapper for the pooled analysis: instead of a caller-
    supplied target problem_id list (build_records + attach_unknown_slot
    for the problem-SPECIFIC analysis), uses EVERY problem_id present in
    `df` and attaches BOTH unknown_slot and problem_type in one call.
    Equivalent to:
        build_records(df, df['problem_id'].unique().tolist())
        -> attach_ordering -> attach_unknown_slot -> attach_problem_type
        -> attach_pairwise_precedence
    """
    all_ids = df["problem_id"].dropna().unique().tolist()
    records = build_records(df, all_ids)
    records = attach_ordering(records)
    records = attach_unknown_slot(records, meta_df, lesson, id_col=id_col)
    records = attach_problem_type(records, meta_df, lesson, id_col=id_col)
    records = attach_pairwise_precedence(records)
    return records


def _ordering_error_table(records: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Shared logic behind per_problem_ordering_error_table (group_col=
    'problem_id') and pooled_ordering_error_table (group_col=
    'problem_type'): for each (group, ordering), n_observations and the
    error rate on each of the 3 steps separately. Only rows with
    is_complete==True contribute -- partial-attempt counts are reported by
    partial_attempt_summary() instead.

    n_observations vs. n_unique_students: for group_col='problem_id' these
    are normally equal (a student attempts a given problem once). For
    group_col='problem_type' they are USUALLY NOT equal -- a student who
    attempted several different problems sharing a type contributes one
    observation PER problem, so n_observations can be several times
    n_unique_students once you sum across a type's rows. Both are reported
    explicitly so this is never silently ambiguous; see
    pooled_ordering_error_table's docstring for why this also matters for
    interpreting the matching significance tests (repeated-measures
    non-independence, not just a bookkeeping detail).

    When pooling by problem_type, also reports n_problems (how many
    distinct problem_ids contributed to that row)."""
    complete = records[records["is_complete"]]
    rows = []
    for (grp, ordering), g in complete.groupby([group_col, "ordering"]):
        row = {group_col: grp, "ordering": " -> ".join(ordering), "n_observations": len(g),
               "n_unique_students": int(g["student_id"].nunique())}
        if group_col != "problem_id":
            row["n_problems"] = int(g["problem_id"].nunique())
        unknown_slot = g["unknown_slot"].iloc[0] if "unknown_slot" in g.columns else None
        row["unknown_slot"] = unknown_slot
        if unknown_slot is not None and unknown_slot in ordering:
            row["unknown_attempted_position"] = ordering.index(unknown_slot) + 1  # 1-based
        else:
            row["unknown_attempted_position"] = None
        for stepname in sorted(QUANTITY_STEPNAMES):
            col = f"{stepname}__correct"
            vals = g[col].dropna()
            row[f"{stepname}_error_rate"] = float(1 - vals.mean()) if len(vals) else None
            row[f"{stepname}_n"] = int(len(vals))
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result):
        result = result.sort_values([group_col, "n_observations"], ascending=[True, False]).reset_index(drop=True)
    return result


def per_problem_ordering_error_table(records: pd.DataFrame) -> pd.DataFrame:
    """The main result table: for each (problem_id, ordering), n_students
    and the error rate on EACH of the 3 steps separately, plus that
    problem's unknown_slot (constant per problem_id, carried through for
    readability) and whether the ordering started with the unknown slot
    or not (a natural summary of 'did they engage with the hard part
    first'). Only rows with is_complete==True (all 3 steps attempted)
    contribute -- partial-attempt counts are reported by
    partial_attempt_summary() instead, not silently dropped without a
    trace."""
    return _ordering_error_table(records, group_col="problem_id")


def pooled_ordering_error_table(records: pd.DataFrame) -> pd.DataFrame:
    """Same as per_problem_ordering_error_table, but pooled ACROSS every
    problem of the same problem_type (see rules.determine_problem_type /
    attach_problem_type / build_records_all_problems) instead of one row
    per specific problem_id -- trades "this exact problem's comprehension"
    for statistical power, on the assumption that problems sharing a
    problem_type share the same ratio structure (what NumeratorQuantity1/
    NumeratorQuantity2/DenominatorQuantity1 actually mean), which the
    rule-based typing is specifically designed to guarantee. Requires
    `records` to already have a `problem_type` column (attach_problem_type
    or build_records_all_problems).

    IMPORTANT: n_observations is NOT the same as n_unique_students here,
    and the gap can be large -- if the same student attempted several
    different problems of this type, they contribute one observation PER
    problem, so summing n_observations across a type's rows can exceed
    the dataset's total student count several times over (check
    n_unique_students, not n_observations, if you want to know how many
    distinct students a result rests on). This also means the matching
    significance tests (test_pooled_*) are treating repeated measurements
    from the same student as independent observations, which they are
    not -- a student who is consistently strong or weak contributes
    correlated data points across all their problems of this type, which
    can make pooled p-values look more significant than the number of
    independent students actually supports (a pseudo-replication /
    repeated-measures problem). Treat pooled p-values as suggestive of
    where to look closer, not as more independent evidence than
    n_unique_students would justify -- a paired/clustered test (e.g.
    clustering standard errors by student_id, or a mixed-effects model
    with a per-student random effect) would be the statistically correct
    next step if this pooled result is going into anything more formal
    than an exploratory pass."""
    if "problem_type" not in records.columns:
        raise ValueError("records has no 'problem_type' column -- run attach_problem_type() "
                          "(or use build_records_all_problems()) first")
    return _ordering_error_table(records, group_col="problem_type")


def partial_attempt_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Per problem_id: how many students had a complete (all 3 attempted)
    vs. partial ordering, so a reader can see how much of the population
    the ordering table actually covers."""
    rows = []
    for pid, g in records.groupby("problem_id"):
        rows.append({"problem_id": pid, "n_total_students": len(g),
                     "n_complete": int(g["is_complete"].sum()),
                     "n_partial": int((~g["is_complete"]).sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# PAIRWISE (partial-order) view -- more robust than the full 3-way ordering
# --------------------------------------------------------------------------
# The functions above require a TOTAL order: all 3 steps attempted, bucketed
# into one of 6 permutations. That's informative but costly in two ways --
# (1) a student who only attempted 2 of the 3 steps is dropped entirely,
# even though "which of those 2 came first" is real precedence information,
# and (2) 6 buckets split even a decent-sized cohort into small, sparse
# groups (see the sparse_warning flags test_step_error_across_orderings
# tends to raise). The functions below instead use a PARTIAL order: for
# each of the 3 unordered pairs of steps, "which of the two was attempted
# first" -- defined whenever BOTH members of that pair were attempted,
# regardless of whether the third step was touched at all. This keeps
# partial-attempt students in the analysis and collapses every comparison
# to 2 groups, which is both more robust (no wasted data) and lets Fisher's
# exact test apply -- exact at any sample size, unlike chi-square's
# large-sample approximation, which is what most single-problem cohorts
# here are too small for anyway.

PAIRS: list[tuple[str, str]] = list(itertools.combinations(sorted(QUANTITY_STEPNAMES), 2))


def attach_pairwise_precedence(records: pd.DataFrame) -> pd.DataFrame:
    """Adds one column per unordered pair, e.g. 'NumeratorQuantity1_vs_NumeratorQuantity2',
    valued '<A>_first' / '<B>_first' / 'tie' / None (None iff either step in
    the pair was never attempted -- unlike `ordering`, this does NOT
    require the THIRD step to have been attempted too)."""
    out = records.copy()
    for a, b in PAIRS:
        oa = out[f"{a}__first_order"]
        ob = out[f"{b}__first_order"]

        def _rel(ra, rb, a=a, b=b):
            if pd.isna(ra) or pd.isna(rb):
                return None
            if ra < rb:
                return f"{a}_first"
            if rb < ra:
                return f"{b}_first"
            return "tie"

        out[f"{a}_vs_{b}"] = [_rel(ra, rb) for ra, rb in zip(oa, ob)]
    return out


def _unknown_vs_given_table(records: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Shared logic behind unknown_vs_given_table (group_col='problem_id')
    and pooled_unknown_vs_given_table (group_col='problem_type'). See
    pooled_ordering_error_table's docstring for why n_observations !=
    n_unique_students matters when group_col='problem_type'."""
    rows = []
    for grp, g in records.groupby(group_col):
        if g["unknown_slot"].isna().all():
            continue
        unknown = g["unknown_slot"].iloc[0]
        givens = [s for s in sorted(QUANTITY_STEPNAMES) if s != unknown]
        for given in givens:
            a, b = sorted([unknown, given])
            col = f"{a}_vs_{b}"
            if col not in g.columns:
                continue
            sub = g[g[col].notna() & (g[col] != "tie")]
            for rel_value, group_label in [(f"{unknown}_first", "unknown_first"),
                                            (f"{given}_first", "given_first")]:
                gg = sub[sub[col] == rel_value]
                if len(gg) == 0:
                    continue
                u_correct = gg[f"{unknown}__correct"].dropna()
                g_correct = gg[f"{given}__correct"].dropna()
                row = {
                    group_col: grp, "unknown_slot": unknown, "given_slot": given,
                    "precedence": group_label, "n_observations": len(gg),
                    "n_unique_students": int(gg["student_id"].nunique()),
                    "unknown_error_rate": float(1 - u_correct.mean()) if len(u_correct) else None,
                    "given_error_rate": float(1 - g_correct.mean()) if len(g_correct) else None,
                }
                if group_col != "problem_id":
                    row["n_problems"] = int(gg["problem_id"].nunique())
                rows.append(row)
    result = pd.DataFrame(rows)
    if len(result):
        result = result.sort_values([group_col, "given_slot", "precedence"]).reset_index(drop=True)
    return result


def unknown_vs_given_table(records: pd.DataFrame) -> pd.DataFrame:
    """For each problem and each of the 2 'given' steps (the ones that
    aren't the unknown), splits students into 'unknown attempted before
    this given' vs. 'given attempted before unknown' -- using ONLY
    students who attempted both of that specific pair, so partial-attempt
    students who touched the unknown + one given (but not the other given)
    still contribute one row here even though they're excluded from
    per_problem_ordering_error_table entirely. Reports error rate on the
    unknown step (the one that actually requires reasoning) under each
    condition, and on the given step too for reference."""
    return _unknown_vs_given_table(records, group_col="problem_id")


def pooled_unknown_vs_given_table(records: pd.DataFrame) -> pd.DataFrame:
    """Same as unknown_vs_given_table, but pooled across all problems of
    the same problem_type -- also reports n_problems (distinct problem_ids
    contributing to that row). Requires a `problem_type` column
    (attach_problem_type or build_records_all_problems)."""
    if "problem_type" not in records.columns:
        raise ValueError("records has no 'problem_type' column -- run attach_problem_type() "
                          "(or use build_records_all_problems()) first")
    return _unknown_vs_given_table(records, group_col="problem_type")


def _repeated_measures_diagnostics(sub: pd.DataFrame, group_col: str) -> dict:
    """Common addition to every test_* result: n_observations vs.
    n_unique_students, plus an explicit warning when group_col=
    'problem_type' and they differ (repeated measurements from the same
    student across multiple problems of that type -- the chi-square/
    Fisher's exact tests here assume independent observations, which
    repeated per-student measurements are not; see
    pooled_ordering_error_table's docstring)."""
    n_obs = len(sub)
    n_students = int(sub["student_id"].nunique()) if "student_id" in sub.columns else None
    out = {"n_observations": n_obs, "n_unique_students": n_students}
    if group_col != "problem_id" and n_students is not None and n_students < n_obs:
        out["repeated_measures_warning"] = (
            f"{n_obs} observations come from only {n_students} unique students -- some "
            "students contributed multiple (correlated) observations across different "
            "problems of this type. This p-value treats them as independent, which "
            "inflates apparent significance; treat it as exploratory, not confirmatory."
        )
    return out


def _test_unknown_vs_given(records: pd.DataFrame, group_col: str, group_value: str,
                            given_slot: str) -> dict:
    """Shared logic behind test_unknown_vs_given (group_col='problem_id')
    and test_pooled_unknown_vs_given (group_col='problem_type"). 2x2
    Fisher's exact test: is error rate on the UNKNOWN step associated with
    whether it was attempted before or after `given_slot`? Exact at any
    sample size (no sparse-cell caveat needed, unlike the chi-square tests
    below). See _repeated_measures_diagnostics -- when pooled, this test's
    "observations" can include multiple per student, which the test does
    not account for."""
    g = records[records[group_col] == group_value]
    if len(g) == 0 or g["unknown_slot"].isna().all():
        return {group_col: group_value, "note": "no records / no known unknown_slot"}
    unknown = g["unknown_slot"].iloc[0]
    if given_slot == unknown:
        return {group_col: group_value, "note": f"'{given_slot}' IS the unknown slot for this group"}
    a, b = sorted([unknown, given_slot])
    col = f"{a}_vs_{b}"
    sub = g[g[col].notna() & (g[col] != "tie")]
    sub = sub.dropna(subset=[f"{unknown}__correct"])
    if len(sub) == 0:
        return {group_col: group_value, "unknown_slot": unknown, "given_slot": given_slot,
                "note": "no students attempted both steps"}

    table = pd.crosstab(sub[col].map(lambda v: "unknown_first" if v == f"{unknown}_first" else "given_first"),
                         sub[f"{unknown}__correct"].map({1: "correct", 0: "error"}))
    for outcome in ("correct", "error"):
        if outcome not in table.columns:
            table[outcome] = 0
    for group in ("unknown_first", "given_first"):
        if group not in table.index:
            table.loc[group] = [0, 0]
    table = table.loc[["unknown_first", "given_first"], ["error", "correct"]]

    odds_ratio, p = fisher_exact(table.to_numpy())
    return {group_col: group_value, "unknown_slot": unknown, "given_slot": given_slot,
            "test": "fisher_exact", "odds_ratio": odds_ratio, "p_value": p, "table": table,
            **_repeated_measures_diagnostics(sub, group_col)}


def test_unknown_vs_given(records: pd.DataFrame, problem_id: str, given_slot: str) -> dict:
    return _test_unknown_vs_given(records, "problem_id", problem_id, given_slot)


def test_pooled_unknown_vs_given(records: pd.DataFrame, problem_type: str, given_slot: str) -> dict:
    """Same test as test_unknown_vs_given, run on data pooled across every
    problem sharing `problem_type` instead of one specific problem_id."""
    return _test_unknown_vs_given(records, "problem_type", problem_type, given_slot)


def _unknown_position_summary(records: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Shared logic behind unknown_slot_position_summary (group_col=
    'problem_id') and pooled_unknown_slot_position_summary (group_col=
    'problem_type'). See pooled_ordering_error_table's docstring for why
    n_observations != n_unique_students matters when group_col=
    'problem_type'."""
    complete = records[records["is_complete"] & records["unknown_slot"].notna()]
    rows = []
    for grp, g in complete.groupby(group_col):
        unknown_slot = g["unknown_slot"].iloc[0]
        pos = g["ordering"].map(lambda o: o.index(unknown_slot) + 1)  # 1-based
        sub = pd.DataFrame({"position": pos, "correct": g[f"{unknown_slot}__correct"],
                             "problem_id": g["problem_id"], "student_id": g["student_id"]
                             }).dropna(subset=["position", "correct"])
        for p in sorted(sub["position"].unique()):
            bucket = sub[sub["position"] == p]
            row = {group_col: grp, "unknown_slot": unknown_slot,
                   "unknown_attempted_position": int(p), "n_observations": len(bucket),
                   "n_unique_students": int(bucket["student_id"].nunique()),
                   "error_rate": float(1 - bucket["correct"].mean())}
            if group_col != "problem_id":
                row["n_problems"] = int(bucket["problem_id"].nunique())
            rows.append(row)
    result = pd.DataFrame(rows)
    if len(result):
        result = result.sort_values([group_col, "unknown_attempted_position"]).reset_index(drop=True)
    return result


def unknown_slot_position_summary(records: pd.DataFrame) -> pd.DataFrame:
    """A more targeted slice of per_problem_ordering_error_table: for each
    problem, does error rate on the UNKNOWN step specifically (the one
    step that actually requires solving, not just transcribing a given)
    depend on whether the student tackled it 1st, 2nd, or 3rd? Collapses
    the 6 full orderings down to 3 buckets (unknown-step position), which
    trades some resolution for a cleaner, more directly interpretable
    number and more students per bucket (helps with the small-cell/
    sparse-table problem test_step_error_across_orderings warns about)."""
    return _unknown_position_summary(records, group_col="problem_id")


def pooled_unknown_slot_position_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Same as unknown_slot_position_summary, but pooled across every
    problem sharing the same problem_type. Requires a `problem_type`
    column (attach_problem_type or build_records_all_problems)."""
    if "problem_type" not in records.columns:
        raise ValueError("records has no 'problem_type' column -- run attach_problem_type() "
                          "(or use build_records_all_problems()) first")
    return _unknown_position_summary(records, group_col="problem_type")


def _test_unknown_position_effect(records: pd.DataFrame, group_col: str, group_value: str,
                                   min_expected_cell: float = 5.0) -> dict:
    """Shared logic behind test_unknown_position_effect (group_col=
    'problem_id') and test_pooled_unknown_position_effect (group_col=
    'problem_type'). See _repeated_measures_diagnostics."""
    complete = records[(records[group_col] == group_value) & records["is_complete"]
                        & records["unknown_slot"].notna()]
    if len(complete) == 0:
        return {group_col: group_value, "note": "no complete records with a known unknown_slot"}
    unknown_slot = complete["unknown_slot"].iloc[0]
    pos = complete["ordering"].map(lambda o: o.index(unknown_slot) + 1)
    correct = complete[f"{unknown_slot}__correct"]
    sub = pd.DataFrame({"position": pos, "correct": correct,
                         "student_id": complete["student_id"]}).dropna(subset=["position", "correct"])
    if sub["position"].nunique() < 2:
        return {group_col: group_value, "unknown_slot": unknown_slot,
                "note": "fewer than 2 positions observed"}

    table = pd.crosstab(sub["position"], sub["correct"].map({1: "correct", 0: "error"}))
    for outcome in ("correct", "error"):
        if outcome not in table.columns:
            table[outcome] = 0
    table = table[["error", "correct"]]
    diag = _repeated_measures_diagnostics(sub, group_col)

    if table.shape[0] == 2:
        odds_ratio, p = fisher_exact(table.to_numpy())
        return {group_col: group_value, "unknown_slot": unknown_slot, "test": "fisher_exact",
                "positions_compared": list(table.index), "odds_ratio": odds_ratio, "p_value": p,
                "table": table, **diag}

    chi2, p, dof, expected = chi2_contingency(table.to_numpy())
    return {group_col: group_value, "unknown_slot": unknown_slot, "test": "chi2_contingency",
            "chi2": chi2, "p_value": p, "dof": dof,
            "sparse_warning": bool((expected < min_expected_cell).any()),
            "min_expected_cell": float(expected.min()), "table": table, **diag}


def test_unknown_position_effect(records: pd.DataFrame, problem_id: str,
                                  min_expected_cell: float = 5.0) -> dict:
    """Same test as test_step_error_across_orderings, but on the collapsed
    3-bucket (unknown attempted 1st/2nd/3rd) view instead of the full 6
    orderings -- usually less sparse, so a more reliable p-value with
    typical classroom-sized cohorts."""
    return _test_unknown_position_effect(records, "problem_id", problem_id, min_expected_cell)


def test_pooled_unknown_position_effect(records: pd.DataFrame, problem_type: str,
                                         min_expected_cell: float = 5.0) -> dict:
    """Same test as test_unknown_position_effect, run on data pooled
    across every problem sharing `problem_type`."""
    return _test_unknown_position_effect(records, "problem_type", problem_type, min_expected_cell)


def _step_error_across_orderings_test(records: pd.DataFrame, group_col: str, group_value: str,
                                       stepname: str, min_expected_cell: float = 5.0) -> dict:
    """Shared logic behind test_step_error_across_orderings (group_col=
    'problem_id') and test_pooled_step_error_across_orderings (group_col=
    'problem_type'). See _repeated_measures_diagnostics."""
    complete = records[(records[group_col] == group_value) & (records["is_complete"])]
    col = f"{stepname}__correct"
    sub = complete[[col, "ordering", "student_id"]].dropna(subset=[col])
    if sub["ordering"].nunique() < 2 or len(sub) == 0:
        return {group_col: group_value, "stepname": stepname, "note": "fewer than 2 orderings observed"}

    table = pd.crosstab(sub["ordering"].map(lambda o: " -> ".join(o)), sub[col].map({1: "correct", 0: "error"}))
    for outcome in ("correct", "error"):
        if outcome not in table.columns:
            table[outcome] = 0
    table = table[["error", "correct"]]
    diag = _repeated_measures_diagnostics(sub, group_col)

    if table.shape[0] == 2 and table.shape[1] == 2:
        odds_ratio, p = fisher_exact(table.to_numpy())
        return {group_col: group_value, "stepname": stepname, "test": "fisher_exact",
                "orderings_compared": list(table.index), "odds_ratio": odds_ratio, "p_value": p,
                "table": table, **diag}

    chi2, p, dof, expected = chi2_contingency(table.to_numpy())
    sparse = bool((expected < min_expected_cell).any())
    return {group_col: group_value, "stepname": stepname, "test": "chi2_contingency",
            "chi2": chi2, "p_value": p, "dof": dof,
            "sparse_warning": sparse, "min_expected_cell": float(expected.min()),
            "table": table, **diag}


def test_step_error_across_orderings(records: pd.DataFrame, problem_id: str, stepname: str,
                                      min_expected_cell: float = 5.0) -> dict:
    """For ONE problem and ONE of the 3 steps: is that step's error rate
    associated with `ordering`? Builds a (n_orderings x 2) contingency
    table (error vs. correct counts per ordering) and runs chi-square
    (or, if any expected cell count < min_expected_cell -- too sparse for
    chi-square's asymptotics -- falls back to nothing-collapsed advice
    rather than silently reporting an unreliable p-value). For exactly 2
    orderings observed with a sparse table, uses Fisher's exact test
    instead, which doesn't have the same sample-size requirement."""
    return _step_error_across_orderings_test(records, "problem_id", problem_id, stepname, min_expected_cell)


def test_pooled_step_error_across_orderings(records: pd.DataFrame, problem_type: str, stepname: str,
                                             min_expected_cell: float = 5.0) -> dict:
    """Same test as test_step_error_across_orderings, run on data pooled
    across every problem sharing `problem_type`."""
    return _step_error_across_orderings_test(records, "problem_type", problem_type, stepname, min_expected_cell)
