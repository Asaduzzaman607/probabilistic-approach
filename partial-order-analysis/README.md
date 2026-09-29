# LLM-Inferred Prerequisite Rules — Impact & Coverage Analysis (Anonymous)

Code for the analysis and figure in this paper, submitted for
double-blind review.

## What's in here

```
run_demo.sh        -- one-command setup + demo run (see Quickstart, below)
requirements.txt   -- pip dependencies
code/
  pipeline/         -- the analysis pipeline (raw logs -> results)
  demo_synthetic_run.py       -- RUN THIS: a working demo (see below)
  verify_table_*.py           -- checks that the paper's tables are correct
  make_figure_*.py            -- builds the paper's figure
  check_coverage_metric_definition.py
sample_data/        -- the real 34-problem sample + real LLM output for it
```

**Data note:** the raw student logs and full problem metadata are not
included (data-use restrictions) and not otherwise shared. `sample_data/`
has real data — which 34 problems were sampled and what the LLM proposed
for each — but no student records. Because of this, only
`demo_synthetic_run.py` (below) can actually run start to finish here;
everything else in `code/` needs the unshared raw data to run, and is
included so the method can be read and checked.

## Quickstart (one command)

```bash
bash run_demo.sh
```

This sets up a local virtual environment, installs dependencies, and
runs the demo below. Run it from the repository root (macOS/Linux;
on Windows, see the manual steps instead).

## Run the demo manually

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd code
python3 demo_synthetic_run.py
```

Either way, this runs the real pipeline code (`analyze` →
`edge_table_with_counts.py` → `by_type_significance.py`) on the real
sample problems and real LLM output, with **made-up student attempts**
standing in for the unshared real logs. It's here to prove the code
works end-to-end — the numbers it prints are not the paper's real
results.

## What each script does

- `code/pipeline/` — the full pipeline: loads raw logs and metadata,
  classifies each problem, draws the stratified sample, calls the LLM,
  computes compliance and error rates, and runs the significance tests.
  Driven by `run_study_cli.py --stage {filter,select,annotate,analyze}`.
- `verify_table_self_consistency.py` / `verify_table_against_raw_data.py`
  — independent checks that the paper's published tables match the
  underlying data.
- `check_coverage_metric_definition.py` — diagnostic behind the coverage
  metric's definition.
- `make_figure_coverage_impact.py` — builds the paper's figure.
- `make_figure_coverage_impact_raw_csv.py` — same figure, alternate
  build straight from the raw CSV.

## Coverage metric

For a rule within a problem type:

```
coverage_pct = 100 x (instances where the LLM recommended this rule)
                    / (all instances in that problem type)
```

An "instance" is one student's attempt at one problem.
