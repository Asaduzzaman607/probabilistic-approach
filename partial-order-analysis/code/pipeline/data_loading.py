"""
data_loading.py
----------------
Loading + parsing of the raw log CSVs (prior-ER, prior-ME, change-3,
change-4) and the two metadata CSVs, plus the tab-separated step_sequence
mini-language:

    Stepname-action-attemptnumber-type-outcome-timestamp

Stepnames never contain '-', but timestamps typically do (ISO dates), so we
split from the left with a fixed number of fields and let the timestamp
absorb everything left over.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

PRIOR_COLUMNS = [
    "school_id", "prob_num", "student_id", "status", "problem_id",
    "student_used_code", "eta", "kc_skills", "step_sequence", "problem_order",
]

CHANGE_COLUMNS = [
    "school_id", "prob_num", "student_id", "status", "problem_id",
    "student_used_code", "eta", "optimal_strategy", "step_sequence",
    "kc_skills", "problem_order",
]

# Columns explicitly called out as noisy / to be ignored downstream.
IGNORE_COLUMNS = ["optimal_strategy", "eta"]

ER_STEPNAMES = {"DenominatorFactor", "NumeratorFactor", "EquationAnswer"}
ME_STEPNAMES = {"FirstRow1:1", "FirstRow1:2", "FirstRow2:1", "FirstRow2:2",
                 "SecondRow", "ThirdRow"}
# The 3 "enter a quantity" steps that make up a problem's A/B=C/100 ratio
# entry (exactly one is the unknown/variable per problem -- see
# rules.determine_unknown_slot). Distinct from ER_STEPNAMES/ME_STEPNAMES:
# these are entered regardless of which strategy (ER/ME) the student later
# picks to actually solve for the unknown.
QUANTITY_STEPNAMES = {"NumeratorQuantity1", "NumeratorQuantity2", "DenominatorQuantity1"}
FINAL_ANSWER_STEP = "FinalAnswer"
SELECTION_STEPNAMES = {"OptionalTask_1": "ER", "OptionalTask_2": "ME"}

HINT_LEVEL_CHANGE_OUTCOME = "HINT_LEVEL_CHANGE"
INITIAL_HINT_OUTCOME = "INITIAL_HINT"
OK_OUTCOME = "OK"
ERROR_OUTCOME = "ERROR"
JIT_OUTCOME = "JIT"


@dataclass
class StepToken:
    stepname: str
    action: str
    attemptnumber: str
    type_: str
    outcome: str
    timestamp: str
    order: int  # position within the step_sequence (0-indexed), for ordering


def parse_step_sequence(step_sequence: str) -> list[StepToken]:
    """Parse a tab-separated step_sequence string into an ordered list of
    StepToken. Malformed tokens (wrong number of '-'-delimited fields) are
    skipped rather than raising, since a single corrupt token shouldn't
    sink an entire row; callers can inspect `PARSE_WARNINGS` if needed."""
    tokens: list[StepToken] = []
    if not isinstance(step_sequence, str) or step_sequence.strip() == "":
        return tokens

    for i, raw in enumerate(step_sequence.split("\t")):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split("-", 5)  # stepname, action, attempt#, type, outcome, [rest=timestamp]
        if len(parts) < 6:
            PARSE_WARNINGS.append({"token": raw, "reason": "fewer than 6 '-'-fields"})
            continue
        stepname, action, attemptnumber, type_, outcome, timestamp = parts
        tokens.append(StepToken(stepname, action, attemptnumber, type_,
                                 outcome, timestamp, order=i))
    return tokens


PARSE_WARNINGS: list = []


def load_log_csv(path: str, lesson_type: str) -> pd.DataFrame:
    """Load one of the four raw log CSVs.

    lesson_type: one of {'prior', 'change'} -- selects the expected column
    schema (prior-ER / prior-ME share one schema; change-3 / change-4 share
    another). Extra columns beyond the expected schema are kept as-is;
    missing expected columns raise, since that indicates a real schema
    mismatch worth surfacing rather than silently proceeding.
    """
    df = pd.read_csv(path, dtype={"school_id": str, "student_id": str,
                                   "problem_id": str}, low_memory=False)
    expected = PRIOR_COLUMNS if lesson_type == "prior" else CHANGE_COLUMNS
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing expected columns {missing}")
    df = df.drop(columns=[c for c in IGNORE_COLUMNS if c in df.columns])
    df["_source_lesson_type"] = lesson_type
    return df


def load_metadata_csv(path: str) -> pd.DataFrame:
    """Load a metadata CSV, indexed by lms-id (== problem_id in the logs)."""
    df = pd.read_csv(path, dtype=str, low_memory=False)  # read as str; rules.py
    # coerces numerics itself and treats bad strings as absent -- this keeps
    # the loader itself dumb/robust and puts all "what counts as numeric"
    # logic in one place (rules._get).
    if "lms-id" not in df.columns:
        raise ValueError(f"{path}: missing 'lms-id' column required to join to problem_id")
    return df


def attach_metadata(log_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join log rows to their problem metadata via problem_id == lms-id.
    Rows whose problem_id has no metadata match get all-NaN metadata columns
    (rules.py treats NaN as absent, so downstream rules simply won't fire --
    but we flag these rows so they can be audited rather than silently
    dropped)."""
    merged = log_df.merge(meta_df, how="left", left_on="problem_id",
                           right_on="lms-id", suffixes=("", "_meta"))
    merged["_metadata_missing"] = merged["lms-id"].isna()
    return merged
