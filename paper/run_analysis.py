#!/usr/bin/env python3
"""Stage the preserved scripts into an external workspace and run a paper step.

Only filesystem paths are adapted. Model fitting, seeds, matching and scientific
definitions remain those of the archived sources.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tokenize
import io

SOURCE_ROOT = Path(__file__).resolve().parent / "scripts"
STEPS = {
    "master": "build_final_paper_master.py",
    "tables": "build_revision_tables_v2.py",
    "physiology": "run_revision_physiology_models.py",
    "stage-ahi": "build_stage_ahi_anova_mixed_v1.py",
    "cognition-disease": "run_revision_cognition_disease_target_bsi_v1.py",
    "cognition-figure": "build_revision_cognition_disease_figure_v3.py",
    "mortality-selection": "run_revision_mortality_v2.py",
    "mortality-raw": "run_revision_mortality_cohort_models_v1.py",
    "mortality-score": "run_revision_mortality_bsi_score_cv_v1.py",
    "mortality-comparators": "run_revision_mortality_comparator_all_study_type_v1.py",
    "mortality-figure": "build_revision_mortality_summary_figure_v3.py",
    "rem-nrem": "run_revision_rem_nrem_outcomes_v1.py",
    "study-type": "run_revision_study_type_sensitivity_v1.py",
    "posture": "run_revision_posture_v1.py",
    "osa-csa": "run_revision_osa_csa_ahi_strata_v1.py",
    "raw-bsi-sensitivity": "run_revision_raw_bsi_sensitivity_v1.py",
}


def adapt_paths(text, workspace):
    edits = {}
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.STRING:
            try:
                value = ast.literal_eval(token.string)
            except (ValueError, SyntaxError):
                value = None
            if isinstance(value, str):
                for prefix, target in (("/home/wolfgang/repos/", workspace / "inputs"),
                                       ("/media/cdac_sleep/", workspace / "inputs" / "cdac_sleep")):
                    if value.startswith(prefix):
                        replacement = str(target / value.removeprefix(prefix))
                        edits[value] = replacement
                        token = token._replace(string=repr(replacement))
                        break
        tokens.append(token)
    return tokenize.untokenize(tokens), edits


def prepare_workspace(workspace):
    workspace = workspace.resolve()
    repo = SOURCE_ROOT.parents[1]
    if workspace == repo or repo in workspace.parents:
        raise ValueError("Use an external workspace so analysis data and outputs stay outside the release repository")
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "inputs").mkdir(exist_ok=True)
    records = []
    for source in sorted(SOURCE_ROOT.rglob("*.py")):
        original = source.read_text()
        adapted, edits = adapt_paths(original, workspace)
        target = workspace / source.relative_to(SOURCE_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_text() != adapted:
            raise ValueError(f"Refusing to overwrite different workspace code: {target}")
        target.write_text(adapted)
        records.append({"script": str(source.relative_to(SOURCE_ROOT)),
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "workspace_sha256": hashlib.sha256(adapted.encode()).hexdigest(),
                        "path_adaptations": edits})
    (workspace / "bsi_code_provenance.json").write_text(json.dumps(records, indent=2) + "\n")
    return workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--step", choices=sorted(STEPS))
    parser.add_argument("--include-mgh", action="store_true",
                        help="Include S0001/MGH in mortality scripts whose original defaults list BIDMC and MrOS")
    parser.add_argument("--set-path", action="append", default=[], metavar="MODULE.CONSTANT=PATH",
                        help="Override a script's global input/output Path; e.g. run_revision_cognition_disease_target_bsi_v1.MASTER=/data/master.csv")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    workspace = prepare_workspace(args.workspace)
    if args.step is None:
        print(f"Prepared code-only workspace: {workspace}")
        return 0
    os.chdir(workspace)
    sys.path[:0] = [str(workspace / "REVISION"), str(workspace)]
    name = Path(STEPS[args.step]).stem
    spec = importlib.util.spec_from_file_location(name, workspace / "REVISION" / STEPS[args.step])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if args.include_mgh:
        if args.step not in ("mortality-score", "mortality-raw", "mortality-comparators"):
            raise ValueError("--include-mgh applies to mortality-score, mortality-raw, and mortality-comparators")
        module.COHORTS["S0001"] = "MGH"
    for item in args.set_path:
        lhs, value = item.split("=", 1)
        module_name, constant = lhs.rsplit(".", 1)
        target = sys.modules.get(module_name)
        if target is None or not isinstance(getattr(target, constant, None), Path):
            raise ValueError(f"{lhs} is not a loaded script Path constant")
        setattr(target, constant, Path(value).resolve())
    script_args = args.script_args
    if script_args[:1] == ["--"]:
        script_args = script_args[1:]
    sys.argv = [STEPS[args.step], *script_args]
    # Inputs and run metadata stay in the external workspace, alongside results.
    with (workspace / "bsi_analysis_runs.jsonl").open("a") as handle:
        handle.write(json.dumps({"step": args.step, "script_args": script_args,
                                 "include_mgh": args.include_mgh,
                                 "path_overrides": args.set_path}) + "\n")
    module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
