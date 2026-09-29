"""
rules.py
--------
Robust versions of the skill-mapping / optimal-strategy rules originally
defined in rules.ipynb. Behavior is unchanged from the notebook; the only
additions are:
  * explicit imports (math, numpy)
  * `_get` / `_gettyp` now treat any non-numeric, non-empty string as "absent"
    for the purposes of the numeric rule checks (per the spec: "sometimes
    the columns can have string values that are not numerical in which they
    can be treated as the column is absent"), while still returning the raw
    string for the categorical checks (`_gettyp`, used for problem-class).
  * every rule function is wrapped so that unexpected exceptions (bad row,
    missing column, weird type) turn into `False` / `is_absent` rather than
    crashing a whole batch job. A row that errors is logged by the caller
    (see `labeling.py::apply_rule_safely`) so problems are visible, not
    silently swallowed.
"""
from __future__ import annotations

import math
import numpy as np


# --------------------------------------------------------------------------
# Presence / absence helpers
# --------------------------------------------------------------------------

def is_present(v) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    # numpy nan safety
    try:
        if isinstance(v, (np.floating,)) and np.isnan(v):
            return False
    except Exception:
        pass
    return True


def is_absent(v) -> bool:
    return not is_present(v)


def _is_close_to_integer(x, tol=1e-6):
    return math.isclose(x, round(x), rel_tol=0, abs_tol=tol)


def _gettyp(meta_row, col):
    """Return the raw (stripped) value of a categorical/string column.
    Empty strings -> NaN. Does NOT attempt numeric coercion."""
    v = meta_row.get(col, np.nan)
    if isinstance(v, str):
        v = v.strip()
        if v == "":
            return np.nan
    return v


def _get(meta_row, col):
    """Return a numeric value for `col`, or NaN if the value is missing OR
    is a non-numeric string. This is the robustness fix requested: any
    string that can't be parsed as a float is treated as column-absent for
    the purposes of the numeric scale-checking rules, instead of raising or
    silently comparing incompatible types."""
    v = meta_row.get(col, np.nan)
    if v is None:
        return np.nan
    if isinstance(v, str):
        v = v.strip()
        if v == "":
            return np.nan
        try:
            return float(v)
        except ValueError:
            return np.nan  # <-- robustness: non-numeric string == absent
    if isinstance(v, (int, float, np.integer, np.floating)):
        try:
            if isinstance(v, float) and math.isnan(v):
                return np.nan
        except Exception:
            pass
        return float(v)
    # unknown type (e.g. bool, list) -> treat as absent rather than crash
    return np.nan


def safe_rule(fn):
    """Decorator: any exception inside a rule -> returns False, and stashes
    the exception on a thread-unsafe-but-fine-for-batch module-level list so
    callers can audit which rows caused trouble."""
    def wrapped(meta_row, *a, **kw):
        try:
            return fn(meta_row, *a, **kw)
        except Exception as exc:  # noqa: BLE001
            RULE_ERRORS.append({"rule": fn.__name__, "error": repr(exc),
                                 "row": dict(meta_row) if hasattr(meta_row, "items") else meta_row})
            return False
    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped


RULE_ERRORS: list = []


# --------------------------------------------------------------------------
# Scale-matching primitives (unchanged logic from rules.ipynb)
# --------------------------------------------------------------------------

def check_scale_A_over_B_equals_X_over_100(A, B, tol=1e-6, return_details=False):
    """A/B = X/100 admits an integer-scale relationship between B and 100."""
    info = {"branch": None, "K": None}
    if not (is_present(A) and is_present(B)) or B == 0:
        return (False, info) if return_details else False

    k_a = 100.0 / B
    if k_a > 0 and _is_close_to_integer(k_a, tol):
        K = round(k_a)
        info.update(branch="a", K=K)
        return (True, info) if return_details else True

    k_b = B / 100.0
    if k_b > 0 and _is_close_to_integer(k_b, tol):
        K = round(k_b)
        if K != 0 and _is_close_to_integer(A / K, tol):
            info.update(branch="b", K=K)
            return (True, info) if return_details else True

    return (False, info) if return_details else False


def check_scale_X_over_B_equals_C_over_100(B, C, tol=1e-6, return_details=False):
    """X/B = C/100 admits an integer-scale relationship between B and 100."""
    info = {"branch": None, "K": None}
    if not (is_present(B) and is_present(C)) or B == 0:
        return (False, info) if return_details else False

    k_a = B / 100.0
    if k_a > 0 and _is_close_to_integer(k_a, tol):
        K = round(k_a)
        info.update(branch="a", K=K)
        return (True, info) if return_details else True

    k_b = 100.0 / B
    if k_b > 0 and _is_close_to_integer(k_b, tol):
        K = round(k_b)
        if K != 0 and _is_close_to_integer(C / K, tol):
            info.update(branch="b", K=K)
            return (True, info) if return_details else True

    return (False, info) if return_details else False


def check_scale_A_over_X_equals_B_over_100(change, percent, max_k=1000, tol=1e-9):
    """change / X = percent / 100 -- integer-k search."""
    if not (is_present(change) and is_present(percent)) or percent == 0:
        return False

    for k in range(1, max_k + 1):
        if abs(percent * k - change) < tol:
            return True
        if abs(change * k - percent) < tol:
            return True
    return False


# --------------------------------------------------------------------------
# Change-3 skill mapping (FinalAnswer -> skill)
# --------------------------------------------------------------------------

@safe_rule
def rule_problem_Amount(meta_row) -> bool:
    """percentage-change is not absent."""
    return not is_absent(_get(meta_row, "ppc-scenario_percent-change"))


@safe_rule
def rule_problem_Percentage(meta_row) -> bool:
    """percent-change is absent."""
    return is_absent(_get(meta_row, "ppc-scenario_percent-change"))


CHANGE3_SKILL_RULES = {
    "rule_problem_Amount": ("calculate final amount in context-1", rule_problem_Amount),
    "rule_problem_Percentage": ("calculate percent change in context-1", rule_problem_Percentage),
}


# --------------------------------------------------------------------------
# Change-4 skill mapping (FinalAnswer -> skill)
# --------------------------------------------------------------------------

@safe_rule
def rule_problem_Amount_4(meta_row) -> bool:
    """percentage change is not absent."""
    if _gettyp(meta_row, "problem-class") == "_percentProportion":
        return False
    return not is_absent(_get(meta_row, "ppc-scenario_percent-change"))


@safe_rule
def rule_problem_Percentage_4(meta_row) -> bool:
    """percent-change is absent."""
    if _gettyp(meta_row, "problem-class") == "_percentProportion":
        return False
    return is_absent(_get(meta_row, "ppc-scenario_percent-change"))


@safe_rule
def rule_problem_part(meta_row) -> bool:
    """Find part of percent."""
    if _gettyp(meta_row, "problem-class") == "_percentChange":
        return False
    return is_absent(_get(meta_row, "ppc-scenario_part-amount"))


@safe_rule
def rule_problem_per(meta_row) -> bool:
    """find percent."""
    if _gettyp(meta_row, "problem-class") == "_percentChange":
        return False
    return is_absent(_get(meta_row, "ppc-scenario_percent"))


@safe_rule
def rule_problem_tot(meta_row) -> bool:
    """find total."""
    if _gettyp(meta_row, "problem-class") == "_percentChange":
        return False
    return is_absent(_get(meta_row, "ppc-scenario_total-amount"))


CHANGE4_SKILL_RULES = {
    "rule_problem_Amount_4": ("calculate final amount in context-1", rule_problem_Amount_4),
    "rule_problem_Percentage_4": ("calculate percent change in context-1", rule_problem_Percentage_4),
    "rule_problem_part": ("calculate part-percent-1", rule_problem_part),
    "rule_problem_per": ("calculate percent-1", rule_problem_per),
    "rule_problem_tot": ("calculate total-percent-1", rule_problem_tot),
}


# --------------------------------------------------------------------------
# Optimal-strategy (ER vs ME) rules
# --------------------------------------------------------------------------

@safe_rule
def rule_scale(meta_row) -> bool:
    """Change-3: True -> ER is the optimal strategy for this problem."""
    initial = _get(meta_row, "ppc-scenario_initial-amount")
    final = _get(meta_row, "ppc-scenario_final-amount")
    change = _get(meta_row, "ppc-scenario_change-amount")
    percent = _get(meta_row, "ppc-scenario_percent-change")

    if is_present(change) and is_present(initial):
        if check_scale_A_over_B_equals_X_over_100(change, initial):
            return True

    if is_absent(initial) and is_present(final) and is_present(change):
        if check_scale_A_over_B_equals_X_over_100(change, final):
            return True

    if is_present(percent) and is_present(initial):
        if check_scale_X_over_B_equals_C_over_100(initial, percent):
            return True

    if is_present(initial) and is_present(final) and not is_present(change):
        ch = final - initial
        if check_scale_A_over_B_equals_X_over_100(ch, initial):
            return True

    return False


@safe_rule
def rule_scale_4(meta_row) -> bool:
    """Change-4: True -> ER is the optimal strategy for this problem."""
    if _gettyp(meta_row, "problem-class") == "_percentChange":
        initial = _get(meta_row, "ppc-scenario_total-amount")
        final = _get(meta_row, "ppc-scenario_total-amount")
        change = _get(meta_row, "ppc-scenario_change-amount")
        percent = _get(meta_row, "ppc-scenario_percent-change")
    else:
        initial = _get(meta_row, "ppc-scenario_total-amount")
        final = _get(meta_row, "ppc-scenario_total-amount")
        change = _get(meta_row, "ppc-scenario_part-amount")
        percent = _get(meta_row, "ppc-scenario_percent")

    if is_present(change) and is_present(initial):
        if check_scale_A_over_B_equals_X_over_100(change, initial):
            return True

    if _gettyp(meta_row, "problem-class") == "_percentChange":
        if is_absent(initial) and is_present(final) and is_present(change):
            if check_scale_A_over_B_equals_X_over_100(change, final):
                return True

    if is_present(percent) and is_present(initial):
        if check_scale_X_over_B_equals_C_over_100(initial, percent):
            return True

    if _gettyp(meta_row, "problem-class") == "_percentChange":
        if is_present(initial) and is_present(final) and not is_present(change):
            ch = final - initial
            if check_scale_A_over_B_equals_X_over_100(ch, initial):
                return True

    if _gettyp(meta_row, "problem-class") == "_percentProportion":
        if is_present(change) and is_present(percent) and not is_present(final):
            if check_scale_A_over_X_equals_B_over_100(change, initial):
                return True

    return False


def determine_skill_change3(meta_row) -> str | None:
    """Return the FinalAnswer skill name for a change-3 problem, or None if
    no rule matched (row flagged for manual review by caller)."""
    for _, (skill_name, fn) in CHANGE3_SKILL_RULES.items():
        if fn(meta_row):
            return skill_name
    return None


def determine_skill_change4(meta_row) -> str | None:
    """Return the FinalAnswer skill name for a change-4 problem, or None if
    no rule matched."""
    for _, (skill_name, fn) in CHANGE4_SKILL_RULES.items():
        if fn(meta_row):
            return skill_name
    return None


def determine_optimal_strategy(meta_row, lesson: str) -> str:
    """Return 'ER' or 'ME' for a problem row, given lesson in {'change3','change4'}."""
    if lesson == "change3":
        return "ER" if rule_scale(meta_row) else "ME"
    elif lesson == "change4":
        return "ER" if rule_scale_4(meta_row) else "ME"
    raise ValueError(f"Unknown lesson '{lesson}'")


# --------------------------------------------------------------------------
# Which of the 3 quantity-entry steps is the unknown ("the variable")
# --------------------------------------------------------------------------
# Every change-3/4 problem's ratio has the general form
#     NumeratorQuantity1 / DenominatorQuantity1 = NumeratorQuantity2 / 100
# with the specific quantities depending on problem family:
#   - "Amount"/"Percentage" family (change-3 always; change-4 when NOT
#     _percentProportion):  change-amount / original-amount = percent / 100
#       => NumeratorQuantity1 = change amount, DenominatorQuantity1 =
#          original amount, NumeratorQuantity2 = percent change.
#   - "part/per/tot" family (change-4 only, when NOT _percentChange):
#     part / total = percent / 100
#       => NumeratorQuantity1 = part, DenominatorQuantity1 = total,
#          NumeratorQuantity2 = percent.
# Exactly one of the three is the unknown per problem (the rule that fires
# tells you which) -- this reuses the SAME rule_problem_* functions used
# for skill-tagging above, rather than re-deriving problem type, so there
# is one source of truth for "which rule fires on this problem."

CHANGE3_UNKNOWN_SLOT_RULES = [
    ("rule_problem_Amount", rule_problem_Amount, "NumeratorQuantity1"),   # change-amount unknown
    ("rule_problem_Percentage", rule_problem_Percentage, "NumeratorQuantity2"),  # percent unknown
]

CHANGE4_UNKNOWN_SLOT_RULES = [
    ("rule_problem_Amount_4", rule_problem_Amount_4, "NumeratorQuantity1"),      # change-amount unknown
    ("rule_problem_Percentage_4", rule_problem_Percentage_4, "NumeratorQuantity2"),  # percent unknown
    ("rule_problem_part", rule_problem_part, "NumeratorQuantity1"),      # part unknown
    ("rule_problem_per", rule_problem_per, "NumeratorQuantity2"),        # percent unknown
    ("rule_problem_tot", rule_problem_tot, "DenominatorQuantity1"),      # total unknown
]


def determine_unknown_slot(meta_row, lesson: str) -> str | None:
    """Return which of {'NumeratorQuantity1', 'NumeratorQuantity2',
    'DenominatorQuantity1'} is the unknown/variable step for this problem,
    or None if no rule matched (same 'flag for manual review' contract as
    determine_skill_change3/4). lesson in {'change3', 'change4'}."""
    rules = CHANGE3_UNKNOWN_SLOT_RULES if lesson == "change3" else \
        CHANGE4_UNKNOWN_SLOT_RULES if lesson == "change4" else None
    if rules is None:
        raise ValueError(f"Unknown lesson '{lesson}'")
    for _, fn, slot in rules:
        if fn(meta_row):
            return slot
    return None


def determine_problem_type(meta_row, lesson: str) -> str | None:
    """Return the NAME of the rule that fired (e.g. 'rule_problem_Amount',
    'rule_problem_tot'), or None if no rule matched. This is a finer-
    grained notion of "problem type" than determine_unknown_slot's return
    value: rule_problem_Amount (change3) and rule_problem_part (change4)
    BOTH resolve to unknown_slot='NumeratorQuantity1', but they represent
    different ratio semantics (change-amount unknown vs. part unknown) --
    pooling across problems by rule NAME (not just by which slot ends up
    unknown) keeps that distinction, which matters if the two "types" of
    unknown behave differently even though they land in the same slot."""
    rules = CHANGE3_UNKNOWN_SLOT_RULES if lesson == "change3" else \
        CHANGE4_UNKNOWN_SLOT_RULES if lesson == "change4" else None
    if rules is None:
        raise ValueError(f"Unknown lesson '{lesson}'")
    for rule_name, fn, _ in rules:
        if fn(meta_row):
            return rule_name
    return None
