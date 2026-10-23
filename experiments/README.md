# ANE inference-path verification experiments

Optional, inference-only probes for the private `_ANEInMemoryModel` path used by this
project. They check which MIL graphs the runtime accepts, whether selected operators and
quantization forms produce the expected values, how IOSurfaces must be laid out, and what
a successful call costs. They do not cover training, model deployment, or end-to-end
application acceptance.

- [Findings and evidence](M4AIR_FINDINGS.md) — the sole results and evidence report.
- [Historical evidence snapshot](reference/m4-air-20261018/) — curated copy of the
  2026-10-18 run; see its [provenance notes](reference/m4-air-20261018/README.md).

## Scope

In scope:

- Compile/load/eval acceptance for a fixed set of MIL operators.
- Full-output CPU comparisons for selected operators and quantization forms.
- Runtime INT8 quantize/dequantize (Q/DQ) semantics and compile-time weight
  dequantization representations.
- IOSurface physical layout, allocation capacity, and multi-input/multi-output requests.
- Bounded per-call dispatch/mapping timing and small synthetic throughput graphs.

Out of scope:

- Training and training benchmarks (see the repository root).
- Model deployment, model compatibility, or end-to-end model/application acceptance.
- Long thermal/pressure runs, concurrent GPU workloads, and resident-model stress.
- Public Core ML behavior or a guarantee for other ANE variants or macOS versions.

The baseline host is a MacBook Air M4 with 16 GB RAM, ANE `h16g` (16 cores), running
macOS 26.3.1 (25D2128). Results are host- and shape-specific. Shape coverage is exactly
the shapes named in the findings report; no general shape support is claimed.

## Dependencies

- macOS on Apple Silicon. The tested baseline is the M4 Air host above.
- Xcode command-line tools (`xcrun clang`) and the system frameworks Foundation,
  IOSurface, and Accelerate. No third-party C libraries are required.
- `make` is the supported build and run interface. Do not assume where build artifacts
  are written; use the targets below.
- Python 3.11 or newer for the generator, survey, validation, and publishing scripts.
  They use only the Python standard library; no third-party packages are required.
- All probe sources and scripts currently live flat in this directory.

## Glossary

- **Q / DQ**: MIL `quantize` and `dequantize` operations. Q maps floating-point values to
  quantized integers using a scale and zero point; DQ maps them back.
- **RNE**: round-half-to-even, the rounding used by the CPU Core ML-style reference
  formula here. The observed runtime INT8 path behaves differently; see the findings.
- **Reference / golden**: a deterministic CPU implementation used as the expected output.
- **Stride**: the physical byte distance between rows or elements in an IOSurface. It can
  be larger than the logically packed tensor, so logical byte counts are not layout
  evidence by themselves.
- **Premap**: mapping the request's IOSurfaces once with
  `mapIOSurfacesWithRequest:cacheInference:` before timed evaluations, rather than
  mapping on every evaluation.

## Commands

Run from this directory:

```bash
make all
make test
make verify-cases
make validate
make validate-bench
```

| Target | Behavior |
|---|---|
| `make all` | Build the C/Objective-C probes. Does not execute ANE work. |
| `make test` | Run offline checks and probe self-tests. **Never calls ANE.** |
| `make verify-cases` | Regenerate and verify the deterministic case set. Does not call ANE; it is not an execution result. |
| `make validate` | Build and run the bounded numerical/acceptance suite on ANE. Writes a new evidence directory. |
| `make validate-bench` | Run `validate` plus the bounded synthetic dispatch/mapping timing probes. Writes a new evidence directory. |

Notes:

- Run one ANE validation at a time. Concurrent probes interfere with timing.
- Each validation run creates a new evidence directory and does not overwrite an earlier
  run or the published reference snapshot. Raw diagnostics may contain machine-specific
  paths; they are ignored local artifacts, not publication-safe copies. Use the publisher
  and inspect its output before sharing a run.
- The quantization probes accept `--selftest`, `--emit-mil`, `--json`, `--bench`, and
  `--quiet`. `--selftest` and `--emit-mil` are offline; `--json` produces machine-readable
  results; `--bench` adds bounded synthetic timing.
- `./run_survey.sh --list --csv /tmp/ane-cases.csv` writes the case table with `NOT_RUN`
  without running probes or creating logs. Never treat that list as an execution result.
- For a cross-device rerun, first save and inspect the physical surface/stride data with
  the survey dump option, then rerun with one explicit layout. The current baseline uses
  explicit `align64`.
- `mil_gen.py` is the only generator entry point. Weighted/vision cases are delegated to
  `weighted_cases.py`; do not run that module directly.

## Probes

| File | Purpose and boundary |
|---|---|
| `int8_quant_probe.m` | Q/DQ numerical semantics, constant per-axis scales, and synthetic chain acceptance with optional timing. |
| `constexpr_scale_probe.m` | Affine, blockwise, and LUT weight representations with full CPU goldens. Correctness only, not a memory/bandwidth claim. |
| `op_probe.m` | Compile/load/eval a single MIL case; selected CPU goldens, explicit layouts, and raw output. |
| `mil_gen.py` / `weighted_cases.py` | Deterministic generation of the survey cases. `mil_gen.py` is the sole entry point. |
| `multi_input.m` | One- and two-input requests with and without weight blobs; each input's contribution is checked numerically. |
| `multi_io.m` | Two differently shaped inputs, two outputs, and full CPU comparison. |
| `static_shape_compile.m` | Compile+load cost for concrete static shapes. Does not claim dynamic-shape support. |
| `surface_capacity.m` | Under-, exact-, and over-sized input/output IOSurfaces, with numerical checks on accepted cases. |
| `dispatch_latency.m` | Timing-only wall-clock cost of successful synchronous evaluations. Does not assert numerics. |
| `mapping_overhead.m` | Premap versus per-eval mapping, output-count effects, and mapping-cache comparison. |
| `devinfo.m` / `api_surface.m` | Device and private-selector enumeration. Not execution, async, or zero-copy proof. |
| `int8_decompose.m` | Weight/activation precision combinations on a synthetic convolution. Performance-only. |
| `io_dtype_bench.m` | Direct I/O dtype comparison on a synthetic compute-heavy graph. |
| `inmem_peak_spatial.m` / `sustained.m` | Shape throughput sweep and bounded sustained timing. Not model end-to-end validation. |
| `compile_limit.m` | Bounded compile/load resource smoke test. No fixed per-process limit is asserted. |
| `cpu_gemm_bench.m` | Accelerate synthetic GEMM; GFLOPS is `2mnk/(ms*1e6)`. CPU-only reference, not ANE. |
| `load_sensitivity.sh` / `cpuhog.c` | Optional controlled-load experiments; not part of normal regression. |

## Core verification versus optional diagnostics

Core verification is the bounded `make validate` suite: build, device information,
quantization, weight compression, multi-input/output, static-shape compile/load, surface
capacity, and the operator survey. `make validate-bench` adds dispatch and mapping timing.

`inmem_peak_spatial`, `sustained`, `compile_limit`, `load_sensitivity.sh`/`cpuhog.c`, and
`api_surface` are optional diagnostics. `devinfo` is included to record the validation
host, not as a correctness check. Long-duration pressure, thermal steady state, concurrent
GPU load, CPU-hog sensitivity, and private-API enumeration do not establish end-to-end
behavior and are not part of the current acceptance criteria.

## Relation to the repository benchmarks

The root repository documents training throughput and a large dense INT8 W8A8 benchmark.
The inference-only `int8_decompose` probe reproduces the same W8A8 speedup pattern under a
separate harness. That is a reproduction, not a new discovery, and it does not imply that
other W8A8 graphs, short graphs, or per-channel/dynamic quantization schemes speed up.
The root training and benchmark results are not modified or superseded by these probes.

## Historical evidence

`reference/m4-air-20261018/` is a sanitized publication copy of a historical run, not
output from the current cleaned tree. Its logs retain the command and source names used at
the time; the source hashes in its metadata describe that historical snapshot and do not
match the current files. The old acceptance-only `LAYOUT=compact` field is a reporting
defect and is not layout evidence. The reference [README](reference/m4-air-20261018/README.md)
and [publication manifest](reference/m4-air-20261018/publication.json) document the
transformations, exclusions, and original hashes. [publish_reference.py](publish_reference.py)
is the optional tool used to publish such snapshots; it does not run probes.

See [M4AIR_FINDINGS.md](M4AIR_FINDINGS.md) for the measured results, evidence levels, and
tested-host/shape limits.
