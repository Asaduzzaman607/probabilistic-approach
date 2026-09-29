"""
Coverage-% lollipop chart, v4 (final): instance-weighted coverage, pooled
by problem_type (across ALL given_sets), per methodology confirmed with
the paper's PI.

coverage_pct = 100 x (sum of n_students, over problems where this edge WAS
                       recommended, ANYWHERE in this problem_type --
                       across every given_set variant)
                    / (sum of n_students, over ALL problems in this
                       problem_type -- across every given_set variant)

Line length (impact) = Viol.% - Sat.%, computed per
(problem_type, edge, role, given_set) row, exactly as Tables 6-7 do.

Run from the `code/` directory, with `output/` as a sibling of `code/`:
  ../output/change3/edge_table_with_counts_by_type_llm_significance.csv
  ../output/change4/edge_table_with_counts_by_type_llm_significance.csv
  ../output/prerequisites_change3.json
  ../output/prerequisites_change4.json
  ../output/full_sample_change3.csv
  ../output/full_sample_change4.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import json

FILES_SIG = {
    "change3": "../output/change3/edge_table_with_counts_by_type_llm_significance.csv",
    "change4": "../output/change4/edge_table_with_counts_by_type_llm_significance.csv",
}
FILES_PREREQ = {
    "change3": "../output/prerequisites_change3.json",
    "change4": "../output/prerequisites_change4.json",
}
FILES_SAMPLE = {
    "change3": "../output/full_sample_change3.csv",
    "change4": "../output/full_sample_change4.csv",
}
ORDER_BASIS = "first_touch"
ERROR_STEP = "B"  # D = dependent/primary outcome step


def normalize(gs):
    tokens = gs.lower().split("+")
    tokens = [t.split("-")[0].strip() for t in tokens]
    return "+".join(sorted(tokens))


def load_prereqs(lesson):
    with open(FILES_PREREQ[lesson]) as f:
        data = json.load(f)
    rows = []
    for entry in data:
        for p, d in entry["prerequisites"]:
            rows.append({"lms_id": entry["lms-id"], "edge": f"{p}->{d}"})
    return pd.DataFrame(rows)


def load_full_sample(lesson):
    df = pd.read_csv(FILES_SAMPLE[lesson]).rename(columns={"lms-id": "lms_id"})
    df["given_set_norm"] = df["given_set"].apply(normalize)
    return df


# The 21 identifying keys for Tables 6 & 7.
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

SHORT = {"NumeratorQuantity1": "NumQ1", "NumeratorQuantity2": "NumQ2", "DenominatorQuantity1": "DenomQ1"}
def short_edge(edge):
    left, right = edge.split("->")
    return f"{SHORT[left]}→{SHORT[right]}"

def display_problem_type(pt):
    tokens = pt.split("_")
    if len(tokens) >= 2 and tokens[1] == "find":
        tokens = [tokens[1], tokens[0]] + tokens[2:]
    return "_".join(tokens)

ROLE_SHORT = {"into_unknown": "into unk.", "between_givens": "betw. givens", "out_of_unknown": "out of unk."}

sig_dfs, prereq_dfs, sample_dfs = {}, {}, {}
for lesson in FILES_SIG:
    sig = pd.read_csv(FILES_SIG[lesson])
    sig = sig[(sig["order_basis"] == ORDER_BASIS) & (sig["error_step"] == ERROR_STEP)].copy()
    sig["given_set_norm"] = sig["given_set"].apply(normalize)
    sig_dfs[lesson] = sig

    prereqs = load_prereqs(lesson)
    problems = load_full_sample(lesson)
    prereq_dfs[lesson] = prereqs.merge(problems[["lms_id", "problem_type", "given_set_norm"]], on="lms_id")
    sample_dfs[lesson] = problems

records = []
for lesson, pt, edge, role, given_set in KEYS:
    gsn = normalize(given_set)

    sig = sig_dfs[lesson]
    row = sig[(sig["problem_type"] == pt) & (sig["given_set_norm"] == gsn) &
              (sig["edge"] == edge) & (sig["role"] == role)]
    assert len(row) == 1, f"expected 1 sig row for {lesson}/{pt}/{edge}/{role}/{given_set}, got {len(row)}"
    r = row.iloc[0]
    sc, st, vc, vt = r["satisfied_count"], r["satisfied_total"], r["violated_count"], r["violated_total"]
    impact = 100.0 * vc / vt - 100.0 * sc / st
    theta = math.log((sc / (st - sc)) / (vc / (vt - vc)))
    color = "#2ca02c" if theta < 0 else "#d62728"

    # instance-weighted coverage, pooled across ALL given_sets within this
    # problem_type
    problems = sample_dfs[lesson]
    bucket = problems[problems["problem_type"] == pt]           # pooled: no given_set filter
    n_total_instances = bucket["n_students"].sum()

    prereqs = prereq_dfs[lesson]
    recommended_lms_ids = prereqs[(prereqs["problem_type"] == pt) &
                                   (prereqs["edge"] == edge)]["lms_id"].unique()  # pooled: no given_set filter
    n_recommended_instances = bucket[bucket["lms_id"].isin(recommended_lms_ids)]["n_students"].sum()

    coverage_pct = 100.0 * n_recommended_instances / n_total_instances

    label = f"{lesson} · {display_problem_type(pt)} · {short_edge(edge)} ({ROLE_SHORT.get(role, role)})"
    records.append({"label": label, "coverage_pct": coverage_pct, "impact": impact, "color": color,
                     "n_recommended_instances": n_recommended_instances, "n_total_instances": n_total_instances})

records.sort(key=lambda r: r["impact"])

labels = [r["label"] for r in records]
impacts = [r["impact"] for r in records]
coverage_pcts = [r["coverage_pct"] for r in records]
colors = [r["color"] for r in records]

sizes = 30 + 4 * np.array(coverage_pcts)

fig, ax = plt.subplots(figsize=(10, 9), dpi=150)
y = np.arange(len(records))

for yi, impact, color in zip(y, impacts, colors):
    ax.hlines(yi, 0, impact, color=color, alpha=0.5, linewidth=2, zorder=2)
ax.scatter(impacts, y, s=sizes, color=colors, alpha=0.9, edgecolor="white", linewidth=0.8, zorder=3)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8)
ax.axvline(0, color="#888888", linewidth=1, zorder=1)
ax.set_xlabel("Error reduction at D (pp): Viol.% − Sat.%")
ax.set_title("Rule impact vs. LLM recommendation coverage\n"
              "(dot size = % of instances, across the problem type, where the LLM recommended this rule)",
              fontsize=11, pad=14)
ax.grid(True, axis="x", linestyle="-", linewidth=0.4, color="#e5e5e5", zorder=0)

legend_pcts = [5, 25, 50, 75, 100]
legend_sizes = 30 + 4 * np.array(legend_pcts)
handles = [plt.scatter([], [], s=s, color="#999999", edgecolor="white", linewidth=0.8,
                        label=f"{p}%") for s, p in zip(legend_sizes, legend_pcts)]
size_legend = ax.legend(handles=handles, title="Coverage\n(% of instances)", loc="lower right",
                         fontsize=8, title_fontsize=8.5,
                         handleheight=4.5, labelspacing=2.6, borderpad=1.4,
                         handletextpad=1.5)
ax.add_artist(size_legend)

from matplotlib.lines import Line2D
color_legend = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=9, label='Reduces error at D'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=9, label='Increases error at D'),
]
ax.legend(handles=color_legend, loc="upper left", fontsize=8.5, frameon=True)
ax.add_artist(size_legend)

fig.tight_layout()
fig.savefig("figure_coverage_impact.png", dpi=150, bbox_inches="tight")
print("saved figure_coverage_impact.png\n")
for r in sorted(records, key=lambda r: -r["impact"]):
    print(f"{r['label']:70s} coverage={r['coverage_pct']:5.1f}%  "
          f"({r['n_recommended_instances']:,.0f}/{r['n_total_instances']:,.0f} instances, pooled by problem_type)  "
          f"impact={r['impact']:+7.2f}pp")
