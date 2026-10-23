# M4 Air: ANE inference-path findings

**Host:** MacBook Air M4, 16 GB RAM, ANE `h16g` (16 cores), macOS 26.3.1 (25D2128).
**Run date:** 2026-10-18.
**Primary evidence:** [run summary](reference/m4-air-20261018/summary.json),
[environment and historical source hashes](reference/m4-air-20261018/metadata.json), and
[bounded supplementary measurements](reference/m4-air-20261018/smoke-summary.json).

This report covers only the private `_ANEInMemoryModel` inference path. It does not
evaluate model deployment or compatibility, training, or end-to-end application
acceptance. Results are observations from the tested host and shapes; they are
not promises for other ANE variants, macOS versions, or workloads.

## 1. Tested host and limits

- One Apple M4 Air with 16 GB RAM and ANE `h16g`, macOS 26.3.1 (25D2128). The device probe
  reports one ANE with 16 cores ([device.log](reference/m4-air-20261018/device.log)).
- All tests use the private `_ANEInMemoryModel` compile/load/eval path with IOSurface I/O.
  Public Core ML, GPU, and Metal integration are not tested.
- Inputs/outputs are primarily fp16 IOSurfaces. The tests do not establish bf16, direct
  int8 I/O, or general dtype behavior.
- Shape coverage is limited to the shapes named in each section. There is no claim of
  continuous shape support, dynamic-shape support, or cross-device layout compatibility.
- Runs are serial. Long thermal/pressure runs, concurrent GPU work, CPU-hog sensitivity,
  resident-model stress, and private-API enumeration are outside the core acceptance.
- Performance numbers are wall-clock observations from this host and run. They are not
  byte-for-byte reproducible and should not be used as exact replication targets.

## 2. Evidence levels and status vocabulary

| Level | Meaning |
|---|---|
| Numerical verification | Deterministic inputs, full logical output readback, and comparison with an explicit CPU formula. |
| Execution acceptance | Compile, load, and eval succeeded; numeric semantics were not checked. |
| Synthetic performance | Timing of successful evals; not end-to-end application latency. |
| Interface enumeration | A selector or class exists; the corresponding workflow was not demonstrated. |

The survey CSV has one row per case with `status` and `numeric` columns. `status=OK` means
compile/load/eval succeeded. `numeric=PASS` (written as `NUMERIC=PASS` in probe stdout)
means a CPU golden comparison passed; `numeric=ACCEPTANCE_ONLY` means only execution was
checked. `COMPILE_FAIL` is an exploratory result: the submitted MIL form was rejected,
which does not prove the hardware lacks the operation.

A suite-level `PASS` or exit code 0 means the suite completed under its protocol. It does
**not** mean every case was a numerical pass. Runtime, load, numerical, and infrastructure
failures cause a nonzero exit; compile rejections are recorded separately.

The current run had three runtime-quantization forms rejected, one compile-time weight form
rejected, and nine operator-survey compile rejections. These are supported-form limits, not
global hardware conclusions.

## 3. Useful new claims versus reproduced claims

| Claim | Kind | Evidence |
|---|---|---|
| Dense W8A8 gives about 1.88x over FP16 on the tested 128-layer, 512-channel 64x64 convolution | **Reproduced** from the repository's root benchmark, not a new discovery | [int8-decompose.log](reference/m4-air-20261018/int8-decompose.log) |
| Runtime INT8 Q/DQ uses round-half-away-from-zero and adds zero point before rounding, unlike the CPU RNE reference | New observation | [quantization.log](reference/m4-air-20261018/quantization.log) |
| Per-token `axis=3` rank-1 `[S]` Q/DQ was rejected for the tested shape | New tested-form limit | [quantization.log](reference/m4-air-20261018/quantization.log) |
| Affine, blockwise, and 16-entry uint4 LUT weight dequantization forms produce correct outputs | New coverage | [weight-compression.log](reference/m4-air-20261018/weight-compression.log) |
| Of 76 operator cases, 67 compiled/loaded/evaluated, 7 had CPU goldens, and 9 were compile-rejected | New coverage | [survey_results.csv](reference/m4-air-20261018/survey_results.csv) |
| The tested fp16 NCHW cases needed 64-byte physical row padding | New tested-layout evidence | [survey stdout](reference/m4-air-20261018/survey/stdout) |
| For the tested `[1,512,1,64]` case, under-sized input/output surfaces were rejected; exact or 2x surfaces were accepted | New tested-shape evidence | [surface-sizes.log](reference/m4-air-20261018/surface-sizes.log) |
| Multi-input and multi-output graphs produce correct values with explicit `weights=@{}` | New coverage | [multi-input.log](reference/m4-air-20261018/multi-input.log), [multi-io.log](reference/m4-air-20261018/multi-io.log) |
| A successful synchronous eval costs roughly 0.1–0.2 ms; premap helps some cases but not all | New bounded observation | [dispatch.log](reference/m4-air-20261018/dispatch.log), [dispatch-mapping.log](reference/m4-air-20261018/dispatch-mapping.log) |
| Static-shape compile+load costs about 44–51 ms for the tested 8-layer graphs | New bounded observation | [static-shapes.log](reference/m4-air-20261018/static-shapes.log) |

"Reproduced" means the result confirms an existing root benchmark observation under a
separate inference-only harness. It is not a new discovery. "New" means the claim was not
previously documented in this repository.

## 4. Runtime INT8 Q / DQ

Sources: [int8_quant_probe.m](int8_quant_probe.m) and the shared
[ane_quant_common.h](ane_quant_common.h). Result:
[quantization.log](reference/m4-air-20261018/quantization.log).

Result: **31 PASS, 0 FAIL, 3 UNSUPPORTED**. Of the 31 passes, 17 are numerical checks and
14 are acceptance-only synthetic benchmark cases. Do not read the suite total as 31
numerical passes.

### 4.1 Supported forms

The following were verified numerically on `[1,8,1,256]` fp16 cases:

- Scalar constant scale for Q, DQ, and QDQ.
- Non-uniform rank-1 `[C]` constant scale with `axis=1` for Q, DQ, and QDQ.
- Scalar and vector zero points, including nonzero zero points.
- Quantization of an input tensor and of a signed-permutation convolution output.
- Saturation, positive/negative half-integer boundaries, and rounding boundaries.

MIL `axis` is a named constant input and belongs in the `quantize(...)` /
`dequantize(...)` argument list. Scale must be a scalar or rank-1 constant. Passing an
arbitrary rank-4 broadcast tensor or a runtime-computed tensor as scale was not verified.

Per-token `axis=3` rank-1 `[S]` Q, DQ, and QDQ were compile-rejected for the tested concrete
shape. This bounds the tested form only; it does not prove that every per-token expression
is impossible. Rewrites such as transposes or fixed-scale Q/DQ after normalization were not
tested.

### 4.2 Observed rounding and zero-point order

The measured inputs match:

```text
q_observed = clamp_int8(round_half_away_from_zero(x / scale + zero_point))
dq         = (q - zero_point) * scale
```

The CPU comparison also keeps the Core ML-style reference:

```text
q_reference = clamp_int8(round_half_even(x / scale) + zero_point)
```

On the deliberately half-integer-heavy test set, the two differ. For example, with
`x=0.0625`, `scale=0.125`, and `zero_point=5`, the observed result is 6 while the RNE
reference gives 5. The difference is systematic, not random noise.

With a nonzero zero point, the observed path matches “add zero point, then round,” not
“round, then add zero point.” Across the three discriminating cases, the add-after
interpretation produces 28 + 1 + 2 = 31 mismatches. The JSON aggregate `rne_mismatches`
(6858) is a sum of per-case element mismatches, not an error rate or a unique-element
count.

This finding applies to this INT8 quantize path. It does not describe the standalone
`round` operator, NaN/Inf behavior, or untested ranges, and it should not be used to
publish an e4m3 round-half-even operation.

## 5. Weight dequantization and compression

Source: [constexpr_scale_probe.m](constexpr_scale_probe.m). Result:
[weight-compression.log](reference/m4-air-20261018/weight-compression.log).

Result: **7 PASS, 0 FAIL, 1 UNSUPPORTED**.

| Tested form | Result |
|---|---|
| Affine scalar scale, zero_point 0 or 3 | Numerical PASS |
| Affine `[C]` scale and `[C]` zero_point, `axis=0` | Numerical PASS, per output channel |
| All-equal vector versus scalar affine | Bit-identical output |
| Affine vector `axis=1` | Compile-rejected for this concrete form |
| `constexpr_blockwise_shift_scale`, row-wise scale | Numerical PASS |
| iOS18 `constexpr_lut_to_dense`, 16-entry LUT, uint4 indices | Numerical PASS |

Passing cases use the standard v2 BLOB container. The file header records the actual chunk
count; each chunk has a dtype tag, a uint64 data size, and a uint64 absolute payload offset.
The `+10` byte is inside the size field, not a separate bit-width field. Offline self-tests
cover multiple chunks and offsets.

These results show that the tested weight representations can be consumed by this private
path and produce correct outputs. They do **not** show that the runtime retains the
compressed footprint or uses INT4 multiply-accumulate. The compiler may expand weights at
compile or load time. Claims about bandwidth, memory, or compute benefit require measuring
the compiled artifact, resident memory, and the target subgraph.

## 6. Operator survey

The published table is
[survey_results.csv](reference/m4-air-20261018/survey_results.csv); per-case stdout is in
[survey/stdout](reference/m4-air-20261018/survey/stdout) and raw compiler diagnostics are in
[survey_errors.log](reference/m4-air-20261018/survey/survey_errors.log).

There are **76 unique cases**:

- **67 execution-successful**: 7 with CPU numerical goldens and 60 acceptance-only.
- **9 compile-rejected**.
- **0** numerical failures, unconfirmed layouts, load/eval failures, or infrastructure
  failures.

### 6.1 Numerical checks

- pad constant/reflect/replicate/causal: deterministic input, full output, CPU comparison.
- depth_to_space and pixel_shuffle: CPU references use the corresponding DCR/CRD ordering.
- softmax: CPU comparison; maximum absolute error about `2.186e-4`, measured with an
  explicit tolerance.

### 6.2 Acceptance-only coverage

Coverage includes regular/depthwise/transposed convolution, pooling, several normalization
and activation ops, reshape/transpose/slice/concat/tile/crop, nearest resize, bilinear
resize with explicit parameters, and reduce_mean/max/min/sum. Acceptance does not promise
arbitrary shapes or arbitrary combinations.

### 6.3 Compile-rejected cases

`gather_axis0`, `gather_axis1`, `gather_along_axis`, `logical_and_select`, `logical_xor`,
`mod`, `reduce_prod`, `rsqrt`, and `upsample_bilinear`.

The generic `InvalidMILProgram` error alone does not establish that the hardware lacks an
arithmetic capability. It means this submitted MIL form did not compile.

## 7. IOSurface layout, capacity, and multi-I/O

### 7.1 Logical bytes are not physical layout

For the tested fp16 NCHW cases, packing and unpacking with **each row rounded up to a
64-byte boundary** produced correct numeric results:

- pad output W=34: logical row 68 B, measured physical row 128 B.
- pixel shuffle input W=8 and output W=16: logical rows 16 B and 32 B; both measured
  physical rows 64 B.
After discovering the layout, the pad and shuffle cases were rerun with a **single explicit
`align64` layout** and passed numerically. This is not a candidate that happened to fit.
The 64-byte row alignment is the verified configuration for these numeric-golden cases;
it is not a universal law for all ranks, dtypes, or SoCs, and it does not replace the model
I/O descriptor.

Separately, the timing-only dispatch probe successfully evaluated `[1,768,1,1]` using
`768 * 64 = 49152 B` of physical space for 1536 logical bytes. It has no CPU golden and
is acceptance-only evidence: this does not prove numerical correctness or that this
allocation is the minimum required size.

`surface_capacity` tests `[1,512,1,64]` fp16 with input/output surfaces below, at, and above
the declared size ([surface-sizes.log](reference/m4-air-20261018/surface-sizes.log)):

- Input at 0.5x or output at 0.5x: eval rejected.
- Exact or 2x input/output: eval succeeded and the full numerical check passed with
  maximum difference 0.

The `0x1d` status is a failure symptom, not a unique size diagnostic.

The historical acceptance-only rows may report `LAYOUT=compact` even when the run selected
`align64`. That field is a reporting defect in the old probe and is not layout evidence.
Only numeric-golden rows with an explicit layout support the layout claims above.

### 7.2 Multi-input and multi-output

- [multi-input.log](reference/m4-air-20261018/multi-input.log): one input, two-input
  addition, and two-input weighted convolution all passed CPU goldens. A descriptor call
  without BLOBFILE weights still requires `weights=@{}`; passing `nil` returned a nil
  descriptor.
- [multi-io.log](reference/m4-air-20261018/multi-io.log): `a[1,1,64,64] * b[1,1,64,96]`
  produced `y` and `z` of shape `[1,1,64,96]`; both outputs matched the CPU golden with
  maximum difference 0, and both inputs contribute.

### 7.3 Not yet demonstrated

API enumeration is not execution proof. There is no numerical/synchronization/no-CPU-copy
integration test for Metal producer → ANE → Metal consumer. Cross-device shared events,
`startOffset` sub-surfaces, and asynchronous chaining were not accepted end-to-end. The
existing memcpy-based CPU I/O does not establish GPU↔ANE zero-copy.

## 8. Per-call latency and synthetic throughput

All timing loops check the evaluate return value; failed calls are excluded. The numbers
are synchronous, serial, wall-clock measurements including driver, mapping, and wait
overhead. They are not pure ANE hardware execution time and are not an isolated immutable
submission constant.

### 8.1 Synchronous eval

[dispatch.log](reference/m4-air-20261018/dispatch.log) measured successful trivial-add
evals across six shapes from `[1,64,1,1]` to `[1,4096,1,256]`: **104.19–181.39 µs/eval**.
The small shapes `[1,64,1,1]` through `[1,768,1,1]` and `[1,768,1,256]` were
**129.40–158.14 µs**. All timed evals succeeded (2000, 2000, 2000, 1000, 500, and 200 per
shape).

[mapping_overhead.log](reference/m4-air-20261018/dispatch-mapping.log) measured:

- Small graph, plain evaluation: 207.00 µs; premap: 159.09 µs.
- Small graph with 2 outputs: 170.56 µs; with 3 outputs: 175.83 µs.
- Large graph `[1,768,1,256]`: 128.58 µs plain; 133.61 µs premapped, i.e. not faster.

Premap can help, but the benefit is not a fixed percentage. A rough budget of
0.1–0.2 ms per call implies that 100 serial small-graph calls cost about 10–20 ms, enough
to matter for a 60 fps deadline. This budget must not be added again to a per-graph time
that already includes submission overhead. Interleaved A/B runs, repeated trials, and the
target graph are required before using these numbers for a real design.

### 8.2 Synthetic convolution chains at different widths

From the acceptance-only portion of the runtime quantization probe. Each mode uses the same
base weight seed and input; the quantized modes change the function, so no CPU numeric
alignment was attempted for these large chains. Each entry is a small number of iterations,
not enough for a stable speedup ratio or P95.

| Synthetic graph | FP16 ms | Per-tensor W8A8 ms |
|---|---:|---:|
| 4096→1024→1024→1024, M=96 | 0.3160 | 0.2157 |
| 1024→1024→1024→1024, M=96 | 0.1838 | 0.1490 |
| 4096→1024→1024→1024, M=256 | 0.6348 | 0.3378 |
| 512→512→512→512, M=4096 | 0.5033 | 0.3956 |
| 768→2048→768, M=256 | 0.2113 | 0.1678 |
| 768→768→2048, M=256 | 0.1656 | 0.1266 |

These are local linear graphs, not a complete FFN or attention block. The fixed M and
channel widths bound the tested shapes; the table does not generalize to other workloads.

### 8.3 W8A8 on the large dense benchmark is reproduced

[int8-decompose.log](reference/m4-air-20261018/int8-decompose.log), using 128 layers of
512-channel 64x64 synthetic convolution:

| Mode | ms/eval | Equivalent TOPS |
|---|---:|---:|
| FP16 weights + FP16 activations | 14.90 | 18.45 |
| INT8 weights + FP16 activations | 14.85 | 18.51 |
| FP16 weights + Q/DQ activations | 14.90 | 18.45 |
| INT8 weights + Q/DQ activations | 7.93 | 34.66 |

This configuration is **1.88x**. It reproduces the root benchmark's W8A8 observation under
a separate harness; it is **not a new discovery**. It supports “this graph pattern is
faster,” but does not prove that the sole mechanism is L2 SRAM throttling or an INT8 MAC
switch, and it does not guarantee the same gain for per-channel/dynamic quantization,
short graphs, or other workloads. The root benchmark reported 18.6 FP16 and 35.1 INT8
TOPS in a different run; small differences are expected, and exact replication is not
promised.

## 9. Static compile/load, capacity, and bounded measurements

- [static-shapes.log](reference/m4-air-20261018/static-shapes.log): seven concrete static
  shapes at 512 channels and 8 layers compiled and loaded at **44.4–51.0 ms** each. The
  probe uses only fully specified static shapes and makes no dynamic-shape claim.
- [capacity-smoke.log](reference/m4-air-20261018/capacity-smoke.log): a bounded smoke run
  recorded **5 successful compile/load/eval cycles before failure or limit**. This does not
  establish a fixed per-process limit or the absence of resource leaks.
- [io-dtype.log](reference/m4-air-20261018/io-dtype.log): on the synthetic compute-heavy
  graph, fp32 direct I/O was 1.1570 ms/eval and fp16 was 1.1456 ms/eval, about **1%**
  difference. This does not imply the same difference for real large activation transfers.
- [sustained-smoke.log](reference/m4-air-20261018/sustained-smoke.log): 16x conv 512ch
  sp64, 100 evals, block 50: 0.240 ms/eval (2.24 TFLOPS) then 0.208 ms/eval (2.59 TFLOPS).
  This is a short bounded run, not a thermal steady state.
- The synthetic CPU GEMM benchmark reports `2mnk/(ms*1e6)` GFLOPS and is not an ANE result.

No training throughput, all-SoC SRAM capacity, or thermal/power hard conclusion is retained
from this run. Those require separate, longer measurements.

## 10. Optional diagnostics and open interface work

The following are optional and not part of current acceptance:
`inmem_peak_spatial`, `sustained`, `compile_limit`, `load_sensitivity.sh`/`cpuhog.c`,
`api_surface`, and `devinfo`. Long pressure, CPU-hog sensitivity, GPU-concurrent load, and
private-API enumeration do not form core verification.

Open items:

- Metal ↔ ANE synchronization with no CPU copy, asynchronous execution, and cross-device
  resource lifetime.
- End-to-end shared-event, sub-surface, and async-chaining acceptance.
- General shape/dtype support beyond the exact tested cases.
- Compiled-artifact size, resident memory, and target-subgraph timing needed to support
  compression or bandwidth claims.

## 11. Evidence provenance and naming

The [reference/m4-air-20261018](reference/m4-air-20261018/) directory is a curated,
sanitized publication copy of the historical run, not output from the current cleaned
source tree. Absolute experiment and home paths were replaced with `${EXPERIMENTS}` and
`${HOME}` placeholders, and the historical application-specific metadata scope was
generalized. The [publication manifest](reference/m4-air-20261018/publication.json) records
original and published SHA-256 hashes and every text transformation; the original local
archive is unchanged. Binary dumps, duplicate aggregate survey logs, and obsolete delivery
hashes were intentionally omitted.

Logs retain the command and source names used at the time. In particular, historical
`dispatch_floor`, `dispatch_floor2`, `shape_test`, `shape_test2`, `multi_input_test2`, and
`multi_io_gen` correspond to the current `dispatch_latency`, `mapping_overhead`,
`static_shape_compile`, `surface_capacity`, `multi_input`, and `multi_io` probes.
Historical `mil_gen2.py` corresponds to `weighted_cases.py`; `mil_gen.py` remains the sole
generator entry point. Source hashes in [metadata.json](reference/m4-air-20261018/metadata.json)
describe the historical snapshot and will not match the cleaned tree.

The old acceptance-only `LAYOUT=compact` field is a reporting defect corrected in the
current probe. Do not use it as layout evidence; use the numeric-golden rows and the
explicit `align64` run configuration.

## 12. Reproducing and where to go next

Use the [experiments usage guide](README.md) for build, offline-test, and validation
commands and for the dependency list. A new run writes a new evidence directory and must
not overwrite this reference snapshot. The historical log commands are records of the
original run, not the current usage contract.
