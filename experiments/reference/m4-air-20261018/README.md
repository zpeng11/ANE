# Historical ANE reference evidence

This is a curated publication copy of a historical run, not a run of the current cleaned source tree. Original command names and source hashes are retained to avoid inventing provenance. See the main findings report for the tested environment, result interpretation, and limitations.

`summary.json` describes the main validation run; `smoke-summary.json`, when present, records additional bounded measurements. `metadata.json` contains the historical environment and source hashes. `survey_results.csv` is the sole published operator status table for this run. Per-case output is in `survey/stdout/`; compiler diagnostics are in `survey/survey_errors.log`.

Absolute experiment/home/temporary paths are replaced with `${EXPERIMENTS}`, `${HOME}`, and `${TMPDIR}` placeholders. The historical application-specific metadata scope is generalized. `publication.json` records original and published SHA-256 hashes and every text transformation. Original local archives remain unchanged. Duplicate aggregate logs, binary dumps, and obsolete delivery manifests are intentionally omitted.

Historical acceptance-only output can report `LAYOUT=compact` even when the runner selected `align64`. Do not use that field as layout evidence; it is a reporting defect corrected in the current probe. Numeric-golden results and the explicit run configuration are separate evidence.

Performance measurements are observations, not reproducible byte-for-byte outputs or cross-device promises. To produce a new run, follow [the usage guide](../../README.md); do not overwrite this snapshot.
