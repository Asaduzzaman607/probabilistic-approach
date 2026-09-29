"""
Lollipop chart, built DIRECTLY from the raw significance CSVs -- no
hardcoded counts. Only the 21 IDENTIFYING KEYS (problem_type, edge, role,
given_set) are hardcoded, taken from the confirmed match against your CSV
(verify_table_against_raw_data.py's output) -- the actual counts, percentages,
and rule weight are all read fresh from the file every time this runs.

Run this in your own Jupyter environment (needs the real CSVs).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

FILES = {
    "change3": "../output/change3/edge_table_with_counts_by_type_llm_significance.csv",
    "change4": "../output/change4/edge_table_with_counts_by_type_llm_significance.csv",
}
ORDER_BASIS = "first_touch"
ERROR_STEP = "B"  # D = dependent/primary outcome step

# The 21 identifying keys for Tables 6 & 7 -- confirmed via
# verify_table_against_raw_data.py to each match exactly one row in the raw CSV.
# (lesson, problem_type, edge, role, given_set)
KEYS = [
    ("change3", "change_find_amount", "DenominatorQuantity1->NumeratorQuantity1", "into_unknown", "initial+percent"),
    ("change3", "change_find_amount", "NumeratorQuantity2->NumeratorQuantity1", "into_unknown", "initial+percent"),
    ("change3", "change_find_percent", "DenominatorQuantity1->NumeratorQuantity2", "into_unknown", "change+initial"),
    ("change3", "change_find_percent", "NumeratorQuantity1->NumeratorQuantity2", "into_unknown", "change+initial"),
    ("change3", "change_find_percent", "DenominatorQuantity1->NumeratorQuantity1", "between_givens", "change+initial"),
    ("change4", "change_find_amount", "NumeratorQuantity2->NumeratorQuantity1", "into_unknown", "initial+percent"),
    ("change4", "change_find_amount", "DenominatorQuantity1->NumeratorQuantity1", "into_unknown", "initial+percent"),
    ("change4", "change_find_percent", "NumeratorQuantity1->DenominatorQuantity1", "between_givens", "change+final"),
    ("change4", "change_find_percent", "DenominatorQuantity1->NumeratorQuantity2", "into_unknown", "change+final"),
    ("change4", "change_find_percent", "NumeratorQuantity1->NumeratorQuantity2", "into_unknown", "change+final"),
    ("change4", "change_find_percent", "DenominatorQuantity1->NumeratorQuantity2", "into_unknown", "change+initial"),
    ("change4", "change_find_percent", "NumeratorQuantity1->NumeratorQuantity2", "into_unknown", "change+initial"),
    ("change4", "prop_find_part", "DenominatorQuantity1->NumeratorQuantity1", "into_unknown", "percent+total"),
    ("change4", "prop_find_part", "NumeratorQuantity2->NumeratorQuantity1", "into_unknown", "percent+total"),
    ("change4", "prop_find_part", "NumeratorQuantity2->DenominatorQuantity1", "between_givens", "percent+total"),
    ("change4", "prop_find_percent", "DenominatorQuantity1->NumeratorQuantity1", "between_givens", "part+total"),
    ("change4", "prop_find_percent", "DenominatorQuantity1->NumeratorQuantity2", "into_unknown", "part+total"),
    ("change4", "prop_find_percent", "NumeratorQuantity1->NumeratorQuantity2", "into_unknown", "part+total"),
    ("change4", "prop_find_total", "NumeratorQuantity1->DenominatorQuantity1", "into_unknown", "part+percent"),
    ("change4", "prop_find_total", "NumeratorQuantity2->DenominatorQuantity1", "into_unknown", "part+percent"),
    ("change4", "prop_find_total", "NumeratorQuantity2->NumeratorQuantity1", "between_givens", "part+percent"),
]
assert len(KEYS) == 21

# short display names for plot labels (reverses the CSV's long quantity names)
SHORT = {
    "NumeratorQuantity1": "NumQ1", "NumeratorQuantity2": "NumQ2",
    "DenominatorQuantity1": "DenomQ1",
}
def short_edge(edge):
    left, right = edge.split("->")
    return f"{SHORT[left]}→{SHORT[right]}"
ROLE_SHORT = {"into_unknown": "into unk.", "between_givens": "betw. givens"}

def display_problem_type(pt):
    """CSV's problem_type strings are e.g. 'change_find_amount', 'prop_find_total'
    -- reorder to 'find_change_amount', 'find_prop_total' for display, matching
    make_figure_coverage_impact.py's labeling convention."""
    tokens = pt.split("_")
    if len(tokens) >= 2 and tokens[1] == "find":
        tokens = [tokens[1], tokens[0]] + tokens[2:]
    return "_".join(tokens)

dfs = {lesson: pd.read_csv(path) for lesson, path in FILES.items()}

records = []
for lesson, pt, edge, role, given_set in KEYS:
    df = dfs[lesson]
    row = df[(df["order_basis"] == ORDER_BASIS) & (df["error_step"] == ERROR_STEP) &
             (df["problem_type"] == pt) & (df["edge"] == edge) &
             (df["role"] == role) & (df["given_set"] == given_set)]
    assert len(row) == 1, f"expected exactly 1 row for {lesson}/{pt}/{edge}/{role}/{given_set}, got {len(row)}"
    r = row.iloc[0]
    sc, st, vc, vt = r["satisfied_count"], r["satisfied_total"], r["violated_count"], r["violated_total"]
    coverage = st + vt
    impact = 100.0 * vc / vt - 100.0 * sc / st
    theta = math.log((sc / (st - sc)) / (vc / (vt - vc)))
    color = "#2ca02c" if theta < 0 else "#d62728"
    label = f"{lesson} · {display_problem_type(pt)} · {short_edge(edge)} ({ROLE_SHORT[role]})"
    records.append({"label": label, "coverage": coverage, "impact": impact, "color": color})

records.sort(key=lambda r: r["impact"])

labels = [r["label"] for r in records]
impacts = [r["impact"] for r in records]
coverages = [r["coverage"] for r in records]
colors = [r["color"] for r in records]

log_cov = np.log10(coverages)
sizes = 30 + 400 * (log_cov - log_cov.min()) / (log_cov.max() - log_cov.min())

fig, ax = plt.subplots(figsize=(10, 9), dpi=150)
y = np.arange(len(records))

for yi, impact, color in zip(y, impacts, colors):
    ax.hlines(yi, 0, impact, color=color, alpha=0.5, linewidth=2, zorder=2)
ax.scatter(impacts, y, s=sizes, color=colors, alpha=0.9, edgecolor="white", linewidth=0.8, zorder=3)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8)
ax.axvline(0, color="#888888", linewidth=1, zorder=1)
ax.set_xlabel("Error reduction at D (pp): Viol.% − Sat.%")
ax.set_title("Rule impact, all 21 rules — read directly from raw CSV\n(dot size = coverage, i.e. total student attempts at D)",
              fontsize=13, pad=14)
ax.grid(True, axis="x", linestyle="-", linewidth=0.4, color="#e5e5e5", zorder=0)

legend_covs = [max(2000, int(min(coverages))), 20000, int(max(coverages))]
legend_sizes = 30 + 400 * (np.log10(legend_covs) - log_cov.min()) / (log_cov.max() - log_cov.min())
handles = [plt.scatter([], [], s=s, color="#999999", edgecolor="white", linewidth=0.8,
                        label=f"{c:,}") for s, c in zip(legend_sizes, legend_covs)]
size_legend = ax.legend(handles=handles, title="Coverage\n(attempts at D)", loc="lower right",
                         fontsize=8, title_fontsize=8.5, labelspacing=1.3, borderpad=1.2)
ax.add_artist(size_legend)

from matplotlib.lines import Line2D
color_legend = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=9, label='Reduces error at D'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=9, label='Increases error at D'),
]
ax.legend(handles=color_legend, loc="upper left", fontsize=8.5, frameon=True)
ax.add_artist(size_legend)

fig.tight_layout()
fig.savefig("figure_coverage_impact_raw_csv.png", dpi=150, bbox_inches="tight")
print("saved figure_coverage_impact_raw_csv.png")
for r in sorted(records, key=lambda r: -r["impact"]):
    print(f"{r['label']:70s} coverage={r['coverage']:>8,.0f}  impact={r['impact']:+7.2f}pp")
