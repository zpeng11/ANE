#!/usr/bin/env python3
"""Run local ANE regressions serially and preserve their evidence.

This runner is the explicit hardware entry point (make validate /
make validate-bench).  Use --provenance-only for the offline metadata check
performed by the test suite: it never builds a probe and never calls ANE.

Recorded provenance is portable: paths are stored relative to the checkout,
not as absolute /Users/... paths.  Timings are bounded synthetic observations;
they are not claimed to be byte-identical across runs or devices.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
CASES = ROOT / "opcases"


def sanitize(text):
    """Remove reviewer-specific roots from captured text."""
    text = str(text)
    for prefix, replacement in ((str(ROOT), "."), (str(Path.home()), "~")):
        text = text.replace(prefix, replacement)
    return text


def portable_path(path):
    """Return a path relative to the checkout (or its portable form)."""
    try:
        return sanitize(os.path.relpath(Path(path), ROOT))
    except ValueError:
        return sanitize(path)


def capture(command):
    command = [str(part) for part in command]
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        return {"command": command, "exit_code": result.returncode,
                "stdout": sanitize(result.stdout), "stderr": sanitize(result.stderr)}
    except OSError as error:
        return {"command": command, "exit_code": None, "stdout": "",
                "stderr": sanitize(error)}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_files(paths):
    return {portable_path(path): sha256_file(path)
            for path in paths if Path(path).is_file()}


def source_paths():
    suffixes = {".m", ".h", ".c", ".py", ".sh"}
    return [p for p in sorted(ROOT.iterdir())
            if p.is_file() and (p.suffix in suffixes or p.name == "Makefile")]


def case_paths(cases_dir=CASES):
    return sorted(p for p in Path(cases_dir).iterdir() if p.is_file())


def hash_build_outputs():
    if not BUILD.is_dir():
        return {}
    return hash_files(sorted(p for p in BUILD.iterdir() if p.is_file()))


def index_manifest(cases_dir=CASES):
    """Summarize opcases/_index.txt against the files on disk."""
    cases_dir = Path(cases_dir)
    index = cases_dir / "_index.txt"
    names = []
    if index.is_file():
        names = [line.strip() for line in index.read_text().splitlines() if line.strip()]
    on_disk = {p.stem for p in cases_dir.glob("*.mil")}
    declared = set(names)
    duplicates = sorted({name for name in declared if names.count(name) > 1})
    missing_meta = []
    meta_errors = []
    for name in sorted(declared):
        meta = cases_dir / (name + ".meta")
        if not meta.is_file():
            missing_meta.append(name)
            continue
        fields = meta.read_text().rstrip("\n").split(",")
        if len(fields) != 4:
            meta_errors.append(f"{name}: expected 4 meta fields, found {len(fields)}")
            continue
        label, in_bytes, out_bytes, weights = fields
        if label != name:
            meta_errors.append(f"{name}: meta label is {label!r}")
        try:
            if int(in_bytes) <= 0 or int(out_bytes) <= 0:
                raise ValueError
        except ValueError:
            meta_errors.append(f"{name}: invalid byte counts {in_bytes!r},{out_bytes!r}")
        if weights not in ("", "-"):
            path = Path(weights)
            base = cases_dir.parent
            if path.is_absolute() or ".." in path.parts:
                meta_errors.append(f"{name}: non-portable weights path {weights!r}")
            elif not (base / path).is_file():
                meta_errors.append(f"{name}: missing weights file {weights!r}")
    return {
        "index_present": index.is_file(),
        "declared": len(names),
        "unique": len(declared),
        "names": sorted(declared),
        "duplicate_names": duplicates,
        "mil_files": len(on_disk),
        "missing_mil": sorted(declared - on_disk),
        "extra_mil": sorted(on_disk - declared),
        "missing_meta": sorted(missing_meta),
        "meta_errors": meta_errors,
        "sorted": names == sorted(names),
    }


def manifest_errors(manifest):
    """Return human-readable reasons the case manifest cannot be trusted."""
    errors = []
    if not manifest["index_present"]:
        errors.append("missing opcases/_index.txt")
    if not manifest["declared"]:
        errors.append("empty case manifest")
    if manifest["duplicate_names"]:
        errors.append("duplicate cases: " + ", ".join(manifest["duplicate_names"]))
    if manifest["missing_mil"]:
        errors.append("missing MIL: " + ", ".join(manifest["missing_mil"]))
    if manifest["extra_mil"]:
        errors.append("unindexed MIL: " + ", ".join(manifest["extra_mil"]))
    if manifest["missing_meta"]:
        errors.append("missing meta: " + ", ".join(manifest["missing_meta"]))
    errors.extend(manifest["meta_errors"])
    if not manifest["sorted"]:
        errors.append("_index.txt is not sorted")
    return errors


def survey_row_check(path, declared_names):
    """Require exactly one survey row per declared case; no hardcoded counts."""
    rows = []
    if Path(path).is_file():
        with open(path, newline="") as table:
            rows = list(csv.DictReader(table))
    ops = [(row.get("op") or "").strip() for row in rows]
    duplicates = sorted({name for name in set(ops) if ops.count(name) > 1})
    declared = set(declared_names)
    missing = sorted(declared - set(ops))
    extra = sorted(set(ops) - declared)
    return {
        "ok": (len(rows) == len(declared) and not duplicates
               and not missing and not extra),
        "rows": len(rows),
        "declared": len(declared),
        "duplicate_rows": duplicates,
        "missing_rows": missing,
        "extra_rows": extra,
    }


def classify_survey_csv(path):
    """Split survey outcomes into numeric verification vs acceptance/refusal."""
    rows = []
    if Path(path).is_file():
        with open(path, newline="") as table:
            rows = list(csv.DictReader(table))
    status_counts = {}
    numeric_counts = {}
    for row in rows:
        status = (row.get("status") or "UNKNOWN").strip()
        numeric = (row.get("numeric") or "NA").strip()
        status_counts[status] = status_counts.get(status, 0) + 1
        numeric_counts[numeric] = numeric_counts.get(numeric, 0) + 1
    verified = numeric_counts.get("PASS", 0)
    acceptance_only = numeric_counts.get("ACCEPTANCE_ONLY", 0)
    compile_rejected = status_counts.get("COMPILE_FAIL", 0)
    numeric_failures = sum(count for kind, count in numeric_counts.items()
                           if kind not in {"PASS", "ACCEPTANCE_ONLY", "NA", ""})
    return {
        "total": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "numeric_counts": dict(sorted(numeric_counts.items())),
        "numeric_verified": verified,
        "acceptance_only": acceptance_only,
        "compile_rejected": compile_rejected,
        "numeric_failures": numeric_failures,
    }


def git_provenance():
    commit = capture(["git", "rev-parse", "HEAD"])
    status = capture(["git", "status", "--porcelain"])
    entries = [line for line in status["stdout"].splitlines() if line.strip()]
    return {
        "commit": commit["stdout"].strip() or "unknown",
        "dirty": bool(entries),
        "status_entries": len(entries),
        "status": entries,
    }


def clang_provenance():
    version = capture(["xcrun", "clang", "--version"])
    if version["exit_code"] != 0:
        version = capture(["clang", "--version"])
    return version


def metadata_skeleton(args, output, cases_dir=CASES):
    return {
        "started_at": datetime.datetime.now().astimezone().isoformat(),
        "mode": "provenance-only" if args.provenance_only else "hardware",
        "output": portable_path(output),
        "platform": platform.platform(),
        "python": sys.version,
        "benchmarks_requested": args.bench,
        "scope": ("Serial private-ANE probes; operator, numerical, layout and "
                  "bounded synthetic performance validation only"),
        "performance_reproducibility": (
            "Timings are bounded synthetic observations, not byte-identical "
            "reproductions and not a cross-device guarantee"),
        "git": git_provenance(),
        "clang": clang_provenance(),
        "system": [capture(cmd) for cmd in
                   (["sw_vers"], ["uname", "-m"], ["uptime"], ["pmset", "-g", "therm"])],
        "manifest": index_manifest(cases_dir),
        "source_sha256": hash_files(source_paths()),
        "case_sha256": hash_files(case_paths(cases_dir)),
        "binary_sha256": {},
        "binary_hashes_recorded_at": ("not built (provenance-only)"
                                      if args.provenance_only else "pending build"),
    }


def build_commands(args, output):
    out_csv = portable_path(output / "survey_results.csv")
    out_survey = portable_path(output / "survey")
    commands = [
        ("build", ["make", "all"], 180),
        ("device", ["build/devinfo"], 30),
        ("quantization", ["build/int8_quant_probe", "--json"]
         + (["--bench"] if args.bench else []), 180),
        ("weight-compression", ["build/constexpr_scale_probe", "--json"]
         + (["--bench"] if args.bench else []), 180),
        ("multi-input", ["build/multi_input"], 60),
        ("multi-io", ["build/multi_io"], 60),
        ("static-shapes", ["build/static_shape_compile"], 120),
        ("surface-sizes", ["build/surface_capacity"], 60),
        ("operator-survey", ["./run_survey.sh", "--csv", out_csv,
                             "--results", out_survey, "--layouts", "align64"], 180),
    ]
    if args.bench:
        commands += [
            ("dispatch", ["build/dispatch_latency"], 120),
            ("dispatch-mapping", ["build/mapping_overhead"], 120),
        ]
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="New directory for logs and metadata (must not exist)")
    parser.add_argument("--bench", action="store_true",
                        help="Include bounded synthetic timing tests, not model inference")
    parser.add_argument("--provenance-only", action="store_true",
                        help="Record portable provenance and exit; no build, no probes")
    parser.add_argument("--cases", type=Path, default=CASES,
                        help="Case directory for manifest checks (default: opcases)")
    parser.add_argument("--check-manifest", action="store_true",
                        help="Validate the case manifest and exit; no hardware")
    args = parser.parse_args()
    cases_dir = args.cases.resolve()

    if args.check_manifest:
        manifest = index_manifest(cases_dir)
        errors = manifest_errors(manifest)
        if errors:
            print("manifest: FAIL", file=sys.stderr)
            for error in errors:
                print("  " + error, file=sys.stderr)
            return 1
        print(f"manifest: PASS ({manifest['unique']} unique cases)")
        return 0

    stamp = datetime.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S%z")
    output = args.output or ROOT / "results" / stamp
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    metadata = metadata_skeleton(args, output, cases_dir)
    manifest_errors_found = manifest_errors(metadata["manifest"])
    metadata["manifest"]["ok"] = not manifest_errors_found
    metadata["manifest"]["errors"] = manifest_errors_found
    metadata["survey_classification"] = {
        "status": "not_run",
        "reason": "provenance-only" if args.provenance_only else "not completed",
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps([], indent=2) + "\n")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    if args.provenance_only:
        metadata["finished_at"] = datetime.datetime.now().astimezone().isoformat()
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Provenance: {portable_path(output)}")
        return 0

    if manifest_errors_found:
        metadata["status"] = "manifest-failed"
        metadata["finished_at"] = datetime.datetime.now().astimezone().isoformat()
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("run_validation: case manifest failed; refusing to run hardware", file=sys.stderr)
        for error in manifest_errors_found:
            print("  " + error, file=sys.stderr)
        return 1

    results = []
    for name, command, timeout in build_commands(args, output):
        start = datetime.datetime.now().astimezone()
        log_path = output / (name + ".log")
        with log_path.open("w") as log:
            try:
                result = subprocess.run(command, cwd=ROOT, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=timeout)
                status, code = ("PASS" if result.returncode == 0 else "FAIL"), result.returncode
            except subprocess.TimeoutExpired:
                status, code = "TIMEOUT", None
            except OSError as error:
                log.write(sanitize(error) + "\n")
                status, code = "ERROR", None
        row = {"name": name, "command": command, "status": status, "exit_code": code,
               "log": log_path.name, "started_at": start.isoformat(),
               "duration_seconds": (datetime.datetime.now().astimezone() - start).total_seconds()}
        results.append(row)
        summary_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"{name}: {status} ({portable_path(log_path)})", flush=True)
        if name == "build":
            metadata["binary_sha256"] = hash_build_outputs()
            metadata["binary_hashes_recorded_at"] = "after build"
            (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            if status != "PASS":
                break

    survey_csv = output / "survey_results.csv"
    metadata["survey_classification"] = (
        classify_survey_csv(survey_csv) if survey_csv.is_file()
        else {"status": "not_run", "reason": "survey did not produce a CSV"})
    row_check = survey_row_check(survey_csv, metadata["manifest"]["names"])
    metadata["survey_row_check"] = row_check
    metadata["finished_at"] = datetime.datetime.now().astimezone().isoformat()
    metadata["system_after"] = [capture(["uptime"]), capture(["pmset", "-g", "therm"])]
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    failed = any(row["status"] != "PASS" for row in results) or not row_check["ok"]
    if not row_check["ok"]:
        print("run_validation: survey rows do not match the declared case set", file=sys.stderr)
        for key in ("missing_rows", "extra_rows", "duplicate_rows"):
            if row_check[key]:
                print(f"  {key}: " + ", ".join(row_check[key]), file=sys.stderr)
    print(f"Evidence: {portable_path(output)}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
