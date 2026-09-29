#!/usr/bin/env python3
"""
run_study_cli.py
----------------
Runs the whole partial-order study from the terminal, rebuilding everything
the notebook session was holding in memory.

    export GEMINI_API_KEY=...
    python run_study_cli.py --stage all

Stages (each writes to --out and can be run on its own):
    filter   metadata -> metadata restricted to problems present in the logs
    select   stratified template sample per lesson
    annotate LLM -> prerequisites_{lesson}.json  (k samples, modal set kept)
    analyze  logs + json -> coverage, error tables, tests, baselines

`filter` and `select` are cheap; `annotate` costs API calls; `analyze` loads
the 325 MB logs. Re-running an earlier stage overwrites its outputs, so the
usual loop is: run `all` once, then `analyze` repeatedly while reading results.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

import order_effects_analysis as OE
import run_partial_order_study as S
from data_loading import load_log_csv, load_metadata_csv
from problem_type_stratify import classify_metadata

D = "./dataset/"
LESSONS = {
    "change3": {"log": D + "change3-data.csv", "meta": D + "metadata-3.csv", "per_cell": 8},
    "change4": {"log": D + "change4-data.csv", "meta": D + "metadata-4.csv", "per_cell": 3},
}


def p_inlog(out: Path, lesson: str) -> Path:
    return out / f"{lesson}_meta_inlog.csv"


def p_prereq(out: Path, lesson: str) -> Path:
    return out / f"prerequisites_{lesson}.json"


# ------------------------------------------------------------------ filter --
def stage_filter(out: Path, drop_unknown_given_set: bool = True):
    """Keep only problems students actually attempted. Whole cells of the
    metadata (everything with final-amount given, in this dataset) never
    appear in the logs, and sampling them wastes LLM calls on problems no
    student can be scored against."""
    for lesson, cfg in LESSONS.items():
        log = load_log_csv(cfg["log"], "change")
        log_ids = {OE._clean_id(x) for x in log["problem_id"].dropna().unique()}
        meta = load_metadata_csv(cfg["meta"])
        keep = meta[meta["lms-id"].map(lambda p: OE._clean_id(p) in log_ids)]

        if drop_unknown_given_set:
            # Rows whose quantities live only in the problem text (no numeric
            # metadata) classify fine but can't be assigned to a template
            # cell -- exclude them from the grid rather than sampling a cell
            # you can't describe in the paper.
            path = out / f"_tmp_{lesson}.csv"
            keep.to_csv(path, index=False)
            c = classify_metadata(load_metadata_csv(str(path)), lesson)
            named = c.loc[c["given_set"].astype(str).str.strip() != "", "lms-id"]
            keep = keep[keep["lms-id"].isin(named)]
            path.unlink()

        keep.to_csv(p_inlog(out, lesson), index=False)
        print(f"[filter] {lesson}: {len(keep)}/{len(meta)} problems kept "
              f"-> {p_inlog(out, lesson).name}")


# ------------------------------------------------------------------ select --
def stage_select(out: Path) -> dict[str, pd.DataFrame]:
    samples = {}
    for lesson, cfg in LESSONS.items():
        samples[lesson] = S.step1_select(str(p_inlog(out, lesson)), lesson,
                                          out_dir=str(out), per_cell=cfg["per_cell"])
    return samples


def load_samples(out: Path) -> dict[str, pd.DataFrame]:
    return {l: pd.read_csv(out / f"template_sample_{l}.csv", dtype=str)
            for l in LESSONS}


def build_full_samples(out: Path, min_students: int = 0) -> dict[str, pd.DataFrame]:
    """Every problem with log coverage, not a stratified subset. Used when
    the point is to run the ordering analysis on the whole attempted-problem
    population rather than a template sample.

    min_students: drop problems attempted by fewer than this many distinct
    students. A problem with 8 attempters can still produce a "significant"
    Fisher test off a lopsided split (see mix4-078/mix4-158 in the pilot) --
    that's noise being mistaken for signal, not a finding. Filtering here,
    before annotation, also saves LLM calls on problems that could never
    support a reliable per-problem test anyway."""
    samples = {}
    for lesson, cfg in LESSONS.items():
        meta = load_metadata_csv(str(p_inlog(out, lesson)))
        c = classify_metadata(meta, lesson)
        c = c[c["problem_type"].notna()]

        if min_students > 0:
            log = load_log_csv(cfg["log"], "change")
            n_students = log.groupby("problem_id")["student_id"].nunique()
            n_students.index = n_students.index.map(OE._clean_id)
            c["n_students"] = c["lms-id"].map(OE._clean_id).map(n_students).fillna(0).astype(int)
            before = len(c)
            c = c[c["n_students"] >= min_students]
            print(f"[full] {lesson}: dropped {before - len(c)} problems below "
                  f"{min_students} students")

        text_cols = [col for col in ("problem-text", "question-text") if col in meta.columns]
        c = c.merge(meta[["lms-id"] + text_cols], on="lms-id", how="left")
        c.to_csv(out / f"full_sample_{lesson}.csv", index=False)
        samples[lesson] = c
        print(f"[full] {lesson}: {len(c)} problems (all in-log, classified"
              f"{f', >={min_students} students' if min_students else ''})")
    return samples


# ---------------------------------------------------------------- annotate --
def make_llm_fn(model_name: str):
    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("set GEMINI_API_KEY (export GEMINI_API_KEY=...)")

    client = genai.Client(api_key=key)
    counter = {"n": 0}

    def llm_fn(system: str, user: str) -> str:
        counter["n"] += 1
        print(f"[LLM] call {counter['n']} -> {model_name}", flush=True)
        response = client.models.generate_content(
            model=model_name,
            contents=user,
            config=types.GenerateContentConfig(system_instruction=system, temperature=0.7),
        )
        return response.text
    return llm_fn


def stage_annotate(out: Path, samples: dict, model_name: str, k: int):
    llm_fn = make_llm_fn(model_name)
    # one call first, so a bad key/model fails loudly instead of as k silently
    # caught exceptions inside step2_annotate
    probe = S.build_prompt(samples["change3"].iloc[0])
    llm_fn(*probe)

    frames = []
    for lesson, sample in samples.items():
        st = S.step2_annotate(sample, llm_fn, str(p_prereq(out, lesson)), k=k)
        st.insert(0, "lesson", lesson)
        frames.append(st)
        print(f"[annotate] {lesson}: {len(sample)} problems -> {p_prereq(out, lesson).name}")
    stability = pd.concat(frames, ignore_index=True)
    stability.to_csv(out / "stability.csv", index=False)
    print(stability.to_string(index=False))
    if "agreement_rate" in stability:
        low = stability[stability["agreement_rate"] < 0.8]
        print(f"[annotate] {len(low)} problem(s) below 0.8 agreement across k={k} samples")


# ----------------------------------------------------------------- analyze --
def stage_analyze(out: Path, outcome: str):
    for lesson, cfg in LESSONS.items():
        print(f"\n===== {lesson} =====")
        S.step3_analyze(cfg["log"], str(p_inlog(out, lesson)), lesson,
                        str(p_prereq(out, lesson)), out_dir=str(out / lesson),
                        outcome=outcome)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "filter", "select", "annotate", "analyze"])
    ap.add_argument("--out", default="./study")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("-k", type=int, default=5, help="LLM samples per problem")
    ap.add_argument("--outcome", default="any_quantity_error",
                    help="any_quantity_error | FinalAnswer__correct | <step>__correct")
    ap.add_argument("--full", action="store_true",
                    help="annotate EVERY in-log problem instead of a stratified "
                         "sample -- 164+326=490 problems, ~2450 LLM calls at k=5. "
                         "Skips --stage select.")
    ap.add_argument("--min-students", type=int, default=0,
                    help="with --full, drop problems attempted by fewer than this "
                         "many distinct students (e.g. 100). 0 = no filter.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stage = args.stage

    if stage in ("all", "filter"):
        stage_filter(out)
    if args.full:
        samples = build_full_samples(out, min_students=args.min_students)
    elif stage in ("all", "select"):
        samples = stage_select(out)
    elif stage == "annotate":
        samples = load_samples(out)
    if stage in ("all", "annotate"):
        stage_annotate(out, samples, args.model, args.k)
    if stage in ("all", "analyze"):
        stage_analyze(out, args.outcome)


if __name__ == "__main__":
    main()
