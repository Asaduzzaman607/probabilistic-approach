"""
Verify the 21 rows in Tables 6 & 7 (and every scatter/lollipop script built
from them) against the ACTUAL raw significance CSVs -- not just against
each other. This is the check that closes the loop: LaTeX <-> ROWS was
already checked (verify_table_self_consistency.py); this checks ROWS <-> source data.

Run this in your own Jupyter environment, where the CSVs actually exist.
Adjust FILES below if your paths differ from the convention used earlier
(../output/change3/..., ../output/change4/...).

What it does, per hardcoded row:
  1. Filters the raw CSV to order_basis=="first_touch", error_step=="B"
     (D, the dependent/primary outcome step), and the row's
     (problem_type, edge, role).
  2. Since some (problem_type, edge, role) combos have MULTIPLE given_set
     variants (this was the earlier given_set-pooling bug's root cause),
     it does NOT assume row order matches -- it looks for a given_set
     instance in the raw CSV whose (satisfied_count, satisfied_total,
     violated_count, violated_total) EXACTLY matches the hardcoded row.
  3. Reports MATCH / NO MATCH / AMBIGUOUS (matches more than one given_set)
     for every row, so you get a definitive yes/no on whether the
     published table numbers are traceable back to the raw data.
"""

import pandas as pd

FILES = {
    "change3": "../output/change3/edge_table_with_counts_by_type_llm_significance.csv",
    "change4": "../output/change4/edge_table_with_counts_by_type_llm_significance.csv",
}
ORDER_BASIS = "first_touch"
ERROR_STEP = "B"  # D = dependent/primary outcome step

# (lesson, problem_type, edge, role, satD_count, satD_total, violD_count, violD_total, theta_D)
ROWS = [
    ("change3", "change_find_amount", "DenomQ1→NumQ1", "into unk.", 6284, 49323, 50945, 198031, -0.86),
    ("change3", "change_find_amount", "NumQ2→NumQ1", "into unk.", 52202, 237971, 5026, 9378, -1.41),
    ("change3", "change_find_percent", "DenomQ1→NumQ2", "into unk.", 1529, 8131, 22171, 64286, -0.82),
    ("change3", "change_find_percent", "NumQ1→NumQ2", "into unk.", 1100, 7713, 22601, 64704, -1.17),
    ("change3", "change_find_percent", "DenomQ1→NumQ1", "betw. givens", 86, 470, 849, 2809, -0.66),
    ("change4", "change_find_amount", "NumQ2→NumQ1", "into unk.", 23415, 181194, 2102, 5299, -1.47),
    ("change4", "change_find_amount", "DenomQ1→NumQ1", "into unk.", 1765, 35909, 23754, 150591, -1.27),
    ("change4", "change_find_percent", "NumQ1→DenomQ1", "betw. givens", 13226, 14923, 1760, 2125, 0.48),
    ("change4", "change_find_percent", "DenomQ1→NumQ2", "into unk.", 178, 577, 805, 4472, 0.71),
    ("change4", "change_find_percent", "NumQ1→NumQ2", "into unk.", 148, 562, 835, 4487, 0.44),
    ("change4", "change_find_percent", "DenomQ1→NumQ2", "into unk.", 531, 5012, 6474, 34728, -0.65),
    ("change4", "change_find_percent", "NumQ1→NumQ2", "into unk.", 263, 4616, 6742, 35126, -1.39),
    ("change4", "prop_find_part", "DenomQ1→NumQ1", "into unk.", 886, 11647, 14547, 34023, -2.21),
    ("change4", "prop_find_part", "NumQ2→NumQ1", "into unk.", 3471, 19460, 8758, 17291, -1.56),
    ("change4", "prop_find_part", "NumQ2→DenomQ1", "betw. givens", 55, 1097, 159, 776, -1.56),
    ("change4", "prop_find_percent", "DenomQ1→NumQ1", "betw. givens", 510, 6637, 8632, 33985, -1.43),
    ("change4", "prop_find_percent", "DenomQ1→NumQ2", "into unk.", 1251, 17448, 1231, 9986, -0.60),
    ("change4", "prop_find_percent", "NumQ1→NumQ2", "into unk.", 1246, 17891, 1236, 9542, -0.69),
    ("change4", "prop_find_total", "NumQ1→DenomQ1", "into unk.", 4841, 31412, 4015, 5423, -2.81),
    ("change4", "prop_find_total", "NumQ2→DenomQ1", "into unk.", 1705, 10484, 2733, 8025, -0.97),
    ("change4", "prop_find_total", "NumQ2→NumQ1", "betw. givens", 8079, 23500, 11573, 23709, -0.60),
]
assert len(ROWS) == 21

dfs = {lesson: pd.read_csv(path) for lesson, path in FILES.items()}

# Translate the abbreviated LaTeX-style labels used in ROWS into the raw
# CSV's actual column values (discovered by inspecting df["role"].unique(),
# df["edge"].unique(), etc. -- they don't match the table's display labels).
ROLE_MAP = {
    "into unk.": "into_unknown",
    "betw. givens": "between_givens",
}
QUANTITY_MAP = {
    "NumQ1": "NumeratorQuantity1",
    "NumQ2": "NumeratorQuantity2",
    "DenomQ1": "DenominatorQuantity1",
}

def translate_edge(edge):
    left, right = edge.split("→")
    return f"{QUANTITY_MAP[left]}->{QUANTITY_MAP[right]}"

n_match, n_ambiguous, n_missing = 0, 0, 0
for lesson, pt, edge_display, role_display, sc, st, vc, vt, theta in ROWS:
    edge = translate_edge(edge_display)
    role = ROLE_MAP[role_display]
    df = dfs[lesson]
    candidates = df[(df["order_basis"] == ORDER_BASIS) &
                     (df["error_step"] == ERROR_STEP) &
                     (df["problem_type"] == pt) &
                     (df["edge"] == edge) &
                     (df["role"] == role)]

    exact = candidates[(candidates["satisfied_count"] == sc) &
                        (candidates["satisfied_total"] == st) &
                        (candidates["violated_count"] == vc) &
                        (candidates["violated_total"] == vt)]

    if len(exact) == 1:
        print(f"MATCH      {lesson:8s} {pt:20s} {edge:16s} {role:14s} "
              f"(given_set={exact.iloc[0].get('given_set', '?')})")
        n_match += 1
    elif len(exact) > 1:
        print(f"AMBIGUOUS  {lesson:8s} {pt:20s} {edge:16s} {role:14s} "
              f"-- {len(exact)} given_set rows have IDENTICAL counts, can't disambiguate")
        n_ambiguous += 1
    else:
        print(f"NO MATCH   {lesson:8s} {pt:20s} {edge:16s} {role:14s} "
              f"-- {len(candidates)} candidate row(s) in raw CSV, none match "
              f"sc={sc} st={st} vc={vc} vt={vt}")
        if len(candidates):
            print("           closest candidates found in raw CSV:")
            print(candidates[["given_set", "satisfied_count", "satisfied_total",
                               "violated_count", "violated_total"]].to_string(index=False))
        n_missing += 1

print()
print(f"{n_match} matched, {n_ambiguous} ambiguous, {n_missing} no match (of 21)")
if n_match == 21:
    print("ALL 21 ROWS TRACE BACK EXACTLY TO THE RAW SIGNIFICANCE CSV.")
