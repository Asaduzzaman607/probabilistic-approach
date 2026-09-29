"""
Independent verification: checks that the hardcoded ROWS in
verify_table_against_raw_data.py exactly reproduce the numbers in Tables 6 & 7
of the paper.

Two checks per row:
  1. Internal arithmetic: does satD_count/satD_total actually equal the
     displayed Sat.% at D? Same for Viol.%. (Catches typos in the totals,
     which are NOT printed in the table -- only count and % are shown --
     so a wrong total would silently give a wrong % if not checked this way.)
  2. Cross-check against a SECOND, independently-typed transcription of the
     same 21 rows taken straight from the LaTeX source (not copy-pasted from
     the plotting script) -- counts, displayed %, and theta_D must match.

Run this whenever the LaTeX table changes, to make sure the scatter script's
ROWS were updated too.
"""

import importlib.util
import sys

# ---------------------------------------------------------------
# 1. Load ROWS from the actual verification script (not a copy) so this
#    check always tests whatever file will actually get run. Requires
#    the real CSVs at ../output/change{3,4}/... (verify_table_against_raw_data.py
#    reads them as it executes), so this must be run from the code/ directory.
# ---------------------------------------------------------------
SCRIPT_PATH = "verify_table_against_raw_data.py"
spec = importlib.util.spec_from_file_location("raw_data_mod", SCRIPT_PATH)
mod = importlib.util.module_from_spec(spec)
# the imported script runs top-to-bottom (printing its own MATCH/NO MATCH
# report as a side effect) and defines ROWS along the way
spec.loader.exec_module(mod)
ROWS = mod.ROWS

# ---------------------------------------------------------------
# 2. Independent transcription straight from the LaTeX table text
#    (7a5c94fa-attachment.txt), typed fresh -- not copied from ROWS.
#    Fields: lesson, problem_type, edge, role,
#            satD_pct_displayed, satD_count,
#            violD_pct_displayed, violD_count,
#            theta_D
# ---------------------------------------------------------------
LATEX_ROWS = [
    ("change3", "change_find_amount", "DenomQ1→NumQ1", "into unk.", 12.7, 6284, 25.7, 50945, -0.86),
    ("change3", "change_find_amount", "NumQ2→NumQ1",   "into unk.", 21.9, 52202, 53.6, 5026, -1.41),
    ("change3", "change_find_percent", "DenomQ1→NumQ2", "into unk.", 18.8, 1529, 34.5, 22171, -0.82),
    ("change3", "change_find_percent", "NumQ1→NumQ2",   "into unk.", 14.3, 1100, 34.9, 22601, -1.17),
    ("change3", "change_find_percent", "DenomQ1→NumQ1", "betw. givens", 18.3, 86, 30.2, 849, -0.66),

    ("change4", "change_find_amount", "NumQ2→NumQ1",   "into unk.", 12.9, 23415, 39.7, 2102, -1.47),
    ("change4", "change_find_amount", "DenomQ1→NumQ1", "into unk.", 4.9, 1765, 15.8, 23754, -1.27),
    ("change4", "change_find_percent", "NumQ1→DenomQ1", "betw. givens", 88.6, 13226, 82.8, 1760, 0.48),
    ("change4", "change_find_percent", "DenomQ1→NumQ2", "into unk.", 30.8, 178, 18.0, 805, 0.71),
    ("change4", "change_find_percent", "NumQ1→NumQ2",   "into unk.", 26.3, 148, 18.6, 835, 0.44),
    ("change4", "change_find_percent", "DenomQ1→NumQ2", "into unk.", 10.6, 531, 18.6, 6474, -0.65),
    ("change4", "change_find_percent", "NumQ1→NumQ2",   "into unk.", 5.7, 263, 19.2, 6742, -1.39),
    ("change4", "prop_find_part", "DenomQ1→NumQ1", "into unk.", 7.6, 886, 42.8, 14547, -2.21),
    ("change4", "prop_find_part", "NumQ2→NumQ1",   "into unk.", 17.8, 3471, 50.7, 8758, -1.56),
    ("change4", "prop_find_part", "NumQ2→DenomQ1", "betw. givens", 5.0, 55, 20.5, 159, -1.56),
    ("change4", "prop_find_percent", "DenomQ1→NumQ1", "betw. givens", 7.7, 510, 25.4, 8632, -1.43),
    ("change4", "prop_find_percent", "DenomQ1→NumQ2", "into unk.", 7.2, 1251, 12.3, 1231, -0.60),
    ("change4", "prop_find_percent", "NumQ1→NumQ2",   "into unk.", 7.0, 1246, 13.0, 1236, -0.69),
    ("change4", "prop_find_total", "NumQ1→DenomQ1", "into unk.", 15.4, 4841, 74.0, 4015, -2.81),
    ("change4", "prop_find_total", "NumQ2→DenomQ1", "into unk.", 16.3, 1705, 34.1, 2733, -0.97),
    ("change4", "prop_find_total", "NumQ2→NumQ1",   "betw. givens", 34.4, 8079, 48.8, 11573, -0.60),
]

assert len(ROWS) == 21, f"ROWS has {len(ROWS)} rows, expected 21"
assert len(LATEX_ROWS) == 21, f"LATEX_ROWS has {len(LATEX_ROWS)} rows, expected 21"

errors = []

for i, (r, lr) in enumerate(zip(ROWS, LATEX_ROWS)):
    lesson, ptype, edge, role, satD_c, satD_t, violD_c, violD_t, theta = r
    l_lesson, l_ptype, l_edge, l_role, l_satD_pct, l_satD_c, l_violD_pct, l_violD_c, l_theta = lr

    # --- identity fields must match ---
    for a, b, name in [(lesson, l_lesson, "lesson"), (ptype, l_ptype, "problem_type"),
                        (edge, l_edge, "edge"), (role, l_role, "role")]:
        if a != b:
            errors.append(f"Row {i}: {name} mismatch: ROWS={a!r} vs LATEX={b!r}")

    # --- theta must match ---
    if abs(theta - l_theta) > 1e-9:
        errors.append(f"Row {i} ({ptype}/{edge}): theta mismatch: ROWS={theta} vs LATEX={l_theta}")

    # --- counts must match ---
    if satD_c != l_satD_c:
        errors.append(f"Row {i} ({ptype}/{edge}): satD_count mismatch: ROWS={satD_c} vs LATEX={l_satD_c}")
    if violD_c != l_violD_c:
        errors.append(f"Row {i} ({ptype}/{edge}): violD_count mismatch: ROWS={violD_c} vs LATEX={l_violD_c}")

    # --- internal arithmetic: does count/total reproduce the displayed %? ---
    computed_sat_pct = 100.0 * satD_c / satD_t
    computed_viol_pct = 100.0 * violD_c / violD_t
    if abs(computed_sat_pct - l_satD_pct) > 0.05:
        errors.append(f"Row {i} ({ptype}/{edge}): Sat% mismatch: "
                       f"{satD_c}/{satD_t}={computed_sat_pct:.2f}% vs LaTeX displays {l_satD_pct}%")
    if abs(computed_viol_pct - l_violD_pct) > 0.05:
        errors.append(f"Row {i} ({ptype}/{edge}): Viol% mismatch: "
                       f"{violD_c}/{violD_t}={computed_viol_pct:.2f}% vs LaTeX displays {l_violD_pct}%")

print(f"Checked {len(ROWS)} rows.")
if errors:
    print(f"\n{len(errors)} MISMATCH(ES) FOUND:\n")
    for e in errors:
        print(" -", e)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED: ROWS in verify_table_against_raw_data.py exactly")
    print("reproduce the counts, percentages, and rule weights in the LaTeX tables.")
