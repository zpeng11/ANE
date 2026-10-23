#!/usr/bin/env python3
"""Publish a curated, path-redacted copy of one historical ANE validation run."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

REQUIRED = ("metadata.json", "summary.json", "survey_results.csv")
SCOPE = "Serial private-ANE probes; operator, numerical, layout and synthetic performance evidence only"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def selected_files(source):
    paths = [p for p in source.iterdir()
             if p.is_file() and (p.suffix == ".log" or p.name in {
                 "metadata.json", "summary.json", "smoke-summary.json", "survey_results.csv"})]
    paths += [p for p in (source / "survey" / "stdout").glob("*.log") if p.is_file()]
    paths += [p for p in (source / "survey").glob("*")
              if p.is_file() and p.name in {"survey_errors.log", "run_meta.txt"}]
    return sorted(paths)


def publish(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    for name in REQUIRED:
        if not (source / name).is_file():
            raise ValueError(f"Missing required evidence: {name}")
    metadata = json.loads((source / "metadata.json").read_text())
    json.loads((source / "summary.json").read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".reference-", dir=output.parent))
    try:
        records = {}
        for path in selected_files(source):
            relative = path.relative_to(source)
            original = path.read_bytes()
            text = original.decode("utf-8")
            changes = []
            redacted = re.sub(r"(?:/[^\s\"'<>]*)?/experiments(?=/|[\s\"'<>]|$)",
                              "${EXPERIMENTS}", text)
            redacted = re.sub(r"/Users/[^/\s\"'<>]+", "${HOME}", redacted)
            redacted = re.sub(r"(?:/private)?/var/folders/[^/\s\"'<>]+/[^/\s\"'<>]+/[TC](?=/|[\s\"'<>]|$)",
                              "${TMPDIR}", redacted)
            if redacted != text:
                changes.append("replace absolute experiment/home/temporary paths with placeholders")
            if relative.as_posix() == "metadata.json":
                content = json.loads(redacted)
                if content.get("scope") != SCOPE:
                    content["scope"] = SCOPE
                    changes.append("replace historical application-specific scope with generic scope")
                redacted = json.dumps(content, indent=2) + "\n"
            normalized = re.sub(r"[ \t]+(?=\r?$)", "", redacted, flags=re.MULTILINE)
            if normalized != redacted:
                changes.append("remove trailing horizontal whitespace from publication text")
            published = normalized.encode("utf-8")
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(published)
            records[relative.as_posix()] = {
                "original_sha256": sha256(original),
                "published_sha256": sha256(published),
                "transformations": changes,
            }
        manifest = {
            "schema_version": 1,
            "evidence_started_at": metadata.get("started_at"),
            "historical_evidence": True,
            "source_hashes_describe": "the historical source snapshot, not the current cleaned source tree",
            "selection": "root logs and run summaries; per-case stdout; survey errors and run metadata",
            "excluded": "binary dumps, duplicate aggregate survey stdout/summary, historical delivery hashes",
            "files": records,
        }
        (temporary / "publication.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (temporary / "README.md").write_text(
            "# Historical ANE reference evidence\n\n"
            "This is a curated publication copy of a historical run, not a run of the "
            "current cleaned source tree. Original command names and source hashes are "
            "retained to avoid inventing provenance. See the main findings report for "
            "the tested environment, result interpretation, and limitations.\n\n"
            "`summary.json` describes the main validation run; `smoke-summary.json`, "
            "when present, records additional bounded measurements. `metadata.json` "
            "contains the historical environment and source hashes. `survey_results.csv` "
            "is the sole published operator status table for this run. Per-case output "
            "is in `survey/stdout/`; compiler diagnostics are in `survey/survey_errors.log`.\n\n"
            "Absolute experiment/home/temporary paths are replaced with `${EXPERIMENTS}`, "
            "`${HOME}`, and `${TMPDIR}` placeholders. The historical application-specific metadata scope "
            "is generalized. `publication.json` records original and published SHA-256 "
            "hashes and every text transformation. Original local archives remain unchanged. "
            "Duplicate aggregate logs, binary dumps, and obsolete delivery manifests "
            "are intentionally omitted.\n\n"
            "Historical acceptance-only output can report `LAYOUT=compact` even when "
            "the runner selected `align64`. Do not use that field as layout evidence; "
            "it is a reporting defect corrected in the current probe. Numeric-golden "
            "results and the explicit run configuration are separate evidence.\n\n"
            "Performance measurements are observations, not reproducible byte-for-byte "
            "outputs or cross-device promises. To produce a new run, follow "
            "[the usage guide](../../README.md); do not overwrite this snapshot.\n"
        )
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original validation run directory")
    parser.add_argument("output", type=Path, help="New publication directory (must not exist)")
    args = parser.parse_args()
    try:
        count = publish(args.source, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Published {count} evidence files to {args.output}")


if __name__ == "__main__":
    main()
