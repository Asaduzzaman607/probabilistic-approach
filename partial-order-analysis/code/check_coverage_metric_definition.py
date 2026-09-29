"""
Figures out what the right denominator is for "% of cases the LLM
recommended this rule," per problem type -- needed to redraw the
coverage/impact plot with circle size = % coverage (per the coauthor's
framing) instead of raw attempt counts.

Run in your own Jupyter environment (needs the real CSVs).

What this checks, per problem_type:
  A) sum of (satisfied_total + violated_total) across EVERY edge/role/
     given_set row under that problem_type -- this double-counts if the
     same student attempt appears under multiple edges (e.g. if a problem
     type has 3 candidate rules and every attempt is scored against all 3),
     so it is only a valid denominator if each attempt maps to exactly ONE
     recommended rule.
  B) same sum, but restricted to order_basis=="first_touch" and
     error_step=="B" only (matching what Tables 6/7 actually use) -- avoids
     double-counting across order_basis/error_step variants at least.
  C) n_problems column (if present) summed per problem_type -- some of the
     earlier files had an n_problems column distinct from satisfied/
     violated counts; if that represents unique problem instances rather
     than attempts, it may be a cleaner denominator.

Look at whether (A) or (B) gives a sensible, stable total per problem type
(e.g. does it match order-of-magnitude expectations, and is it the SAME
number regardless of which edge/rule you pick within that problem type --
if the true denominator is "total attempts for this problem type," it
should NOT vary by edge).
"""

import pandas as pd

FILES = {
    "change3": "../output/change3/edge_table_with_counts_by_type_llm_significance.csv",
    "change4": "../output/change4/edge_table_with_counts_by_type_llm_significance.csv",
}
ORDER_BASIS = "first_touch"
ERROR_STEP = "B"

for lesson, path in FILES.items():
    df = pd.read_csv(path)
    print(f"=== {lesson} ===")
    print(f"columns: {df.columns.tolist()}")
    print()

    df_filtered = df[(df["order_basis"] == ORDER_BASIS) & (df["error_step"] == ERROR_STEP)]

    for pt in sorted(df["problem_type"].unique()):
        sub_all = df[df["problem_type"] == pt]
        sub_filtered = df_filtered[df_filtered["problem_type"] == pt]

        total_A = (sub_all["satisfied_total"] + sub_all["violated_total"]).sum()
        total_B = (sub_filtered["satisfied_total"] + sub_filtered["violated_total"]).sum()

        # per-edge totals within the filtered set, to see whether they vary
        # wildly (a sign attempts are being double counted across edges) or
        # cluster near a consistent problem-type-level total
        per_edge = (sub_filtered.groupby(["edge", "role", "given_set"])
                    .apply(lambda g: g["satisfied_total"].sum() + g["violated_total"].sum()))

        print(f"  {pt}:")
        print(f"    (A) sum over ALL rows (any order_basis/error_step): {total_A:,}")
        print(f"    (B) sum over first_touch + error_step=B rows only:  {total_B:,}")
        if "n_problems" in sub_filtered.columns:
            print(f"    (C) sum of n_problems (first_touch + step B):       {sub_filtered['n_problems'].sum():,}")
        print(f"    per (edge, role, given_set) totals within (B):")
        for k, v in per_edge.items():
            print(f"      {k}: {v:,}")
        print()
