"""
labeling.py
-----------
Turns a parsed step_sequence + problem metadata into the set of BKT
"opportunities" (skill, student, problem, correct in {0,1}, order/time) that
everything downstream (filtering, categorization, BKT, CV) consumes.

Design decisions made explicit here (these are judgment calls the spec left
open -- flag if you want them changed):

1. Step-level correctness ("mastery-indicating" observation) for ER/ME
   strategy steps and FinalAnswer:
       correct = 1  iff the FIRST 'Attempt' token for that step has
                    attemptnumber == '1' and outcome == 'OK', AND no
                    HINT_LEVEL_CHANGE token for that same step occurs
                    earlier in the sequence.
       correct = 0  otherwise (multiple attempts before OK, wrong on the
                    first attempt, or hints used before/at the first OK).
   A step the student never touched produces no opportunity at all (it's
   optional, per the spec) -- it is not coded as incorrect.

2. Strategy selection: read from OptionalTask_1 / OptionalTask_2 tokens
   anywhere in the sequence (they're checkbox-style markers, always OK).
   A student can select ER only, ME only, both, or neither.

3. Judgement correctness (ER_Judgement, ME_Judgement) -- only defined for
   change-3/4 problems, one observation of EACH per problem attempt:
       optimal = 'ER' or 'ME'   (from rules.determine_optimal_strategy)
       selected = {'ER'} / {'ME'} / {'ER','ME'} / {}  (from selection markers)

       ER_Judgement correct = 1  iff  selected == {'ER'} and optimal == 'ER'
                                   or  selected != {'ER'} and 'ER' not in selected and optimal == 'ME'
         i.e. simplified: ER_Judgement correct = 1 iff
              (optimal=='ER' and selected=={'ER'}) or (optimal=='ME' and 'ER' not in selected)
       ME_Judgement correct analogous with ER/ME swapped.

       Special case selected == {'ER','ME'} (both): per the spec's explicit
       clarification, this is scored as INCORRECT for *both* judgement
       skills regardless of what was optimal (the two 'optimal==other AND
       X not in selected' branches above already evaluate to False when
       both are in `selected`, so 'both' naturally falls out to (0, 0) --
       no special-casing needed, but it's called out here for clarity).

       Special case selected == {} (neither chosen): the student made no
       selection to judge, and OptionalTask_* tokens are absent entirely.
       We DROP this observation (do not emit an opportunity) rather than
       guessing an implicit judgement, and count these rows so you can see
       how common "neither" is.

4. Skill-selection ("do I use ER or ME") itself: modeled as a mandatory,
   always-observed-correct opportunity when a marker is present (per the
   spec: "it's a selection checkbox, so it would always be OK"). We do NOT
   create a separate BKT skill for the checkbox itself -- only the
   ER_Judgement / ME_Judgement 'was the right box checked' skills, since
   the checkbox-correctness is definitionally trivial (always OK) and
   carries no information.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data_loading import (StepToken, parse_step_sequence, ER_STEPNAMES,
                           ME_STEPNAMES, FINAL_ANSWER_STEP,
                           SELECTION_STEPNAMES, HINT_LEVEL_CHANGE_OUTCOME,
                           OK_OUTCOME)
import rules as R


@dataclass
class Opportunity:
    student_id: str
    school_id: str
    problem_id: str
    lesson: str            # 'prior_ER' | 'prior_ME' | 'change3' | 'change4'
    skill: str              # e.g. 'DenominatorFactor', 'ER_Judgement', 'calculate percent-1'
    correct: int             # 0/1
    problem_order: float | None   # from the log row, for within-lesson ordering
    row_index: int            # original dataframe row index, for tiebreak ordering


def _step_correct(tokens: list[StepToken], stepname: str) -> int | None:
    """Return 1/0 correctness for `stepname` per rule (1) above, or None if
    the student never attempted this step (no opportunity emitted)."""
    step_tokens = [t for t in tokens if t.stepname == stepname]
    if not step_tokens:
        return None

    attempts = [t for t in step_tokens if t.action == "Attempt"]
    first_attempt = next((t for t in attempts if t.attemptnumber == "1"), None)
    if first_attempt is None:
        # Step was touched (e.g. only a hint request) but never actually
        # attempted -- no attempt-1 to judge, so no correctness opportunity.
        return None

    hint_level_changes = [t for t in step_tokens if t.outcome == HINT_LEVEL_CHANGE_OUTCOME]
    earlier_hint = any(h.order < first_attempt.order for h in hint_level_changes)

    if first_attempt.outcome == OK_OUTCOME and not earlier_hint:
        return 1
    return 0


def step_correct(tokens: list[StepToken], stepname: str) -> int | None:
    """Public alias for _step_correct -- for reuse by other modules
    (e.g. order_effects_analysis.py) that need this exact same
    'first-attempt OK, no earlier detailed hint' correctness rule for an
    arbitrary stepname, not just the ones labeling.py itself emits
    opportunities for."""
    return _step_correct(tokens, stepname)


def extract_selection(tokens: list[StepToken]) -> set[str]:
    """Return subset of {'ER','ME'} selected via OptionalTask_1/2 markers."""
    selected = set()
    for t in tokens:
        if t.stepname in SELECTION_STEPNAMES and t.action == "Attempt":
            selected.add(SELECTION_STEPNAMES[t.stepname])
    return selected


NEITHER_SELECTED_COUNT = {"count": 0}


def opportunities_from_row(row: pd.Series, lesson: str, row_index: int,
                            skill_rule_fn=None) -> list[Opportunity]:
    """lesson in {'prior_ER','prior_ME','change3','change4'}.
    skill_rule_fn: rules.determine_skill_change3 / determine_skill_change4,
    required (and only used) for lesson in {'change3','change4'} to map
    FinalAnswer -> a KC skill name."""
    tokens = parse_step_sequence(row.get("step_sequence", ""))
    if not tokens:
        return []

    student_id, school_id, problem_id = row["student_id"], row["school_id"], row["problem_id"]
    problem_order = row.get("problem_order", np.nan)
    out: list[Opportunity] = []

    def add(skill, correct):
        if correct is not None:
            out.append(Opportunity(student_id, school_id, problem_id, lesson,
                                    skill, correct, problem_order, row_index))

    # --- strategy steps (ER/ME), tracked across ALL four lesson sources ---
    relevant_steps = ER_STEPNAMES | ME_STEPNAMES if lesson.startswith("prior") is False \
        else (ER_STEPNAMES if lesson == "prior_ER" else ME_STEPNAMES)
    # NOTE: prior-ER logs should only ever contain ER steps and prior-ME logs
    # only ME steps, but we don't hard-filter on that assumption beyond
    # choosing which stepnames to look for -- if a prior file unexpectedly
    # contains the other family's steps they will simply be ignored here,
    # which is safer than crashing.
    for stepname in relevant_steps:
        add(stepname, _step_correct(tokens, stepname))

    # --- FinalAnswer -> KC skill (change-3/4 only; metadata-dependent) ---
    if lesson in ("change3", "change4") and skill_rule_fn is not None:
        skill_name = skill_rule_fn(row)  # row must carry merged metadata columns
        if skill_name is not None:
            add(skill_name, _step_correct(tokens, FINAL_ANSWER_STEP))
        # else: no rule matched -> can't attribute FinalAnswer to a skill;
        # silently dropping would hide a data problem, so callers should
        # track match-rate via rules.RULE_ERRORS / a separate audit if this
        # matters for their run.

    # --- judgement skills (change-3/4 only) ---
    if lesson in ("change3", "change4"):
        selected = extract_selection(tokens)
        if selected:  # non-empty: at least one strategy chosen
            optimal = R.determine_optimal_strategy(row, lesson)
            er_correct = int((optimal == "ER" and selected == {"ER"}) or
                              (optimal == "ME" and "ER" not in selected))
            me_correct = int((optimal == "ME" and selected == {"ME"}) or
                              (optimal == "ER" and "ME" not in selected))
            add("ER_Judgement", er_correct)
            add("ME_Judgement", me_correct)
        else:
            NEITHER_SELECTED_COUNT["count"] += 1

    return out


def build_opportunities(df: pd.DataFrame, lesson: str, skill_rule_fn=None) -> pd.DataFrame:
    """Vectorized-ish wrapper: apply opportunities_from_row across a whole
    (metadata-merged, for change3/4) log dataframe and return a tidy
    opportunities table."""
    rows: list[Opportunity] = []
    for idx, row in df.iterrows():
        rows.extend(opportunities_from_row(row, lesson, idx, skill_rule_fn))
    if not rows:
        return pd.DataFrame(columns=["student_id", "school_id", "problem_id",
                                      "lesson", "skill", "correct",
                                      "problem_order", "row_index"])
    out = pd.DataFrame([r.__dict__ for r in rows])
    return out
