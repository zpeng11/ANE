// constexpr_scale_probe.m — deterministic tests for compile-time weight dequantization.
//
// Scope
// -----
// Required (numerically checked against a CPU golden):
//   * constexpr_affine_dequantize with a scalar scale and scalar zero_point,
//     standard int8 BLOB weights, then a 1x1 conv;
//   * the same with a nonzero scalar zero_point;
//   * a rank-1 per-output-channel scale (axis=0, scale [C]) and a rank-1
//     per-output-channel zero_point, with the same standard BLOB weights;
//   * a vector whose entries are all equal to the scalar, which must produce a
//     bit-identical output to the scalar control.
//
// Concrete cases (numerically verified on the 2026-10-18 hardware run):
//   * constexpr_blockwise_shift_scale with one scale per output channel;
//   * constexpr_lut_to_dense with a 16-entry LUT and uint4 indices, iOS18 grammar.
// Both matched the scalar affine output exactly (0 mismatches in the validation
// log).  They remain per-concrete-case tests: a refusal is reported UNSUPPORTED,
// not as a universal hardware claim.
//
// Exploratory case:
//   * axis=1 vector scale/zero_point.  Spec-legal, refused on the validated
//     environment.  A compile rejection is UNSUPPORTED; if accepted, it is
//     numerically checked and a mismatch is FAIL.  Axis semantics are not
//     generalised beyond that observation.
//
// Removed on purpose: rank>1 affine scale/zero_point (invalid: the MIL spec allows
// rank 0 or 1), and the iOS16 LUT spelling under an ios18 function.
//
// Golden arithmetic
// -----------------
//   W[o,i] = scale[a] * (q[o,i] - zero_point[a]),  a = o for axis=0, a = i for axis=1
//   y[o,s] = sum_i W[o,i] * x[i,s]
// The test values are chosen so every product and sum is exactly representable in
// fp16 (weights are multiples of 1/8, inputs multiples of 1/4, at most 8 terms).
// The comparison is therefore exact, not tolerance-based.
//
// Build / run
// -----------
//   make constexpr_scale_probe
//   ./constexpr_scale_probe              # quick correctness, no timing
//   ./constexpr_scale_probe --bench      # adds bounded timing loops
//   ./constexpr_scale_probe --json       # one machine-readable JSON object
//   ./constexpr_scale_probe --selftest   # offline blob/MIL/CPU checks, no ANE
//
// Environment: CSP_DUMP=1 prints generated MIL, CSP_ITERS=n sets bench iterations.
#import "ane_quant_common.h"

// ---------------------------------------------------------------- literal helpers
static NSString *scaleVecLiteralEq(int n) {
    NSMutableString *s = [NSMutableString stringWithFormat:@"tensor<fp16, [%d]>([", n];
    for (int i = 0; i < n; i++)
        [s appendFormat:@"0x1p-3%@", i + 1 == n ? @"" : @", "];
    [s appendString:@"])"];
    return s;
}

static NSString *scaleVecLiteralVar(int n) {
    NSMutableString *s = [NSMutableString stringWithFormat:@"tensor<fp16, [%d]>([", n];
    for (int i = 0; i < n; i++)
        [s appendFormat:@"0x1p-%d%@", 3 + (i % 3), i + 1 == n ? @"" : @", "];
    [s appendString:@"])"];
    return s;
}

static NSString *zpVecLiteralZero(int n) {
    NSMutableString *s = [NSMutableString stringWithFormat:@"tensor<int8, [%d]>([", n];
    for (int i = 0; i < n; i++)
        [s appendString:i + 1 == n ? @"0" : @"0, "];
    [s appendString:@"])"];
    return s;
}

static NSString *zpVecLiteralVar(int n) {
    NSMutableString *s = [NSMutableString stringWithFormat:@"tensor<int8, [%d]>([", n];
    for (int i = 0; i < n; i++)
        [s appendFormat:@"%d%@", (i % 3) - 1, i + 1 == n ? @"" : @", "];
    [s appendString:@"])"];
    return s;
}

// ---------------------------------------------------------------- MIL generation
static NSString *genAffineMIL(int C, int S, int axis,
                              NSString *scaleExpr, NSString *zpExpr,
                              NSUInteger wOff)
{
    NSMutableString *m = [NSMutableString stringWithString:aneMilHeader()];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", C, S];
    [m appendString:aneConvConsts()];
    [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> W = constexpr_affine_dequantize()"
        "[axis = int32(%d), name = string(\"W\"), quantized_data = tensor<int8, [%d, %d, 1, 1]>"
        "(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu))), "
        "scale = %@, zero_point = %@];\n",
        C, C, axis, C, C, (unsigned long)wOff, scaleExpr, zpExpr];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> y = conv(dilations = c_dilations, "
        "groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, "
        "weight = W, x = x)[name = string(\"y\")];\n", C, S];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

// One row scale per output channel: data [C,C,1,1], scale [C,1,1,1].
static NSString *genBlockwiseMIL(int C, int S, NSUInteger dataOff, NSUInteger scaleOff) {
    NSMutableString *m = [NSMutableString stringWithString:aneMilHeader()];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", C, S];
    [m appendString:aneConvConsts()];
    [m appendFormat:@"        tensor<int8, [%d, %d, 1, 1]> D = const()[name = string(\"D\"), "
        "val = tensor<int8, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), "
        "offset = uint64(%lu)))];\n", C, C, C, C, (unsigned long)dataOff];
    [m appendFormat:@"        tensor<fp16, [%d, 1, 1, 1]> SC = const()[name = string(\"SC\"), "
        "val = tensor<fp16, [%d, 1, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), "
        "offset = uint64(%lu)))];\n", C, C, (unsigned long)scaleOff];
    [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> W = constexpr_blockwise_shift_scale("
        "data = D, scale = SC)[name = string(\"W\")];\n", C, C];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> y = conv(dilations = c_dilations, "
        "groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, "
        "weight = W, x = x)[name = string(\"y\")];\n", C, S];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

// iOS18 grammar: indices are the output shape, LUT is rank K+2 with
// [..., NUM_PALETTES, VECTOR_SIZE]; no `shape` argument.
static NSString *genLUTMIL(int C, int S, NSUInteger idxOff, NSUInteger lutOff) {
    NSMutableString *m = [NSMutableString stringWithString:aneMilHeader()];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", C, S];
    [m appendString:aneConvConsts()];
    [m appendFormat:@"        tensor<uint4, [%d, %d, 1, 1]> IDX = const()[name = string(\"IDX\"), "
        "val = tensor<uint4, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), "
        "offset = uint64(%lu)))];\n", C, C, C, C, (unsigned long)idxOff];
    [m appendFormat:@"        tensor<fp16, [1, 1, 1, 1, 16, 1]> LUT = const()[name = string(\"LUT\"), "
        "val = tensor<fp16, [1, 1, 1, 1, 16, 1]>(BLOBFILE(path = "
        "string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n",
        (unsigned long)lutOff];
    [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> W = constexpr_lut_to_dense("
        "indices = IDX, lut = LUT)[name = string(\"W\")];\n", C, C];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> y = conv(dilations = c_dilations, "
        "groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, "
        "weight = W, x = x)[name = string(\"y\")];\n", C, S];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

// ---------------------------------------------------------------- CPU golden
static void buildW(double *W, const int8_t *q, int C, int axis,
                   const double *scale, const int *zp) {
    for (int o = 0; o < C; o++)
        for (int i = 0; i < C; i++) {
            int a = (axis == 0) ? o : i;
            W[(size_t)o * C + i] = scale[a] * ((double)q[(size_t)o * C + i] - (double)zp[a]);
        }
}

static void cpuConv(const double *W, const _Float16 *x, int C, int S, _Float16 *y) {
    for (int o = 0; o < C; o++)
        for (int s = 0; s < S; s++) {
            double sum = 0.0;
            for (int i = 0; i < C; i++)
                sum += W[(size_t)o * C + i] * (double)x[(size_t)i * S + s];
            y[(size_t)o * S + s] = (_Float16)sum;
        }
}

static long cmpFp16(const _Float16 *got, const _Float16 *want, size_t n,
                    double *maxAbs, char *detail, size_t detailLen) {
    long mism = 0;
    *maxAbs = 0.0;
    for (size_t k = 0; k < n; k++) {
        double g = (double)got[k], w = (double)want[k];
        double d = fabs(g - w);
        if (d > *maxAbs) *maxAbs = d;
        if (g != w) {
            if (mism == 0)
                snprintf(detail, detailLen, "first[k=%zu] got=%.6g want=%.6g", k, g, w);
            mism++;
        }
    }
    return mism;
}

// ---------------------------------------------------------------- one affine case
static int g_iters = 10;

static int runAffineCase(const char *name, int C, int S, int axis,
                         NSString *scaleExpr, NSString *zpExpr,
                         const double *scale, const int *zp,
                         const int8_t *q, NSUInteger wOff,
                         ANEBlob *blob, const _Float16 *x,
                         _Float16 *refOut, _Float16 *capOut, BOOL required,
                         ANECaseList *list)
{
    double *W = malloc((size_t)C * C * sizeof(double));
    _Float16 *want = malloc((size_t)C * S * sizeof(_Float16));
    _Float16 *got = malloc((size_t)C * S * sizeof(_Float16));
    buildW(W, q, C, axis, scale, zp);
    cpuConv(W, x, C, S, want);

    NSString *mil = genAffineMIL(C, S, axis, scaleExpr, zpExpr, wOff);
    size_t inB = (size_t)C * S * 2, outB = inB;
    ANEResult r = aneRunCase(mil, blob.data, x, inB, outB, g_bench ? g_iters : 0, got, outB);

    int status;
    long mism = 0;
    double maxAbs = 0.0;
    char detail[640] = {0};
    if (!r.compileOK) {
        // Only a real compiler rejection can be UNSUPPORTED.  Setup failures
        // (private API missing, descriptor nil, temp-dir/write errors) are FAIL.
        if (r.compileAttempted && r.compilerRejected && !required)
            status = ST_UNSUPPORTED;
        else
            status = ST_FAIL;
        snprintf(detail, sizeof detail, "%s", r.err);
    } else if (!r.loadOK || !r.evalOK) {
        status = ST_FAIL;
        snprintf(detail, sizeof detail, "%s", r.err);
    } else {
        mism = cmpFp16(got, want, (size_t)C * S, &maxAbs, detail, sizeof detail);
        status = mism ? ST_FAIL : ST_PASS;
        if (mism && detail[0] == 0) snprintf(detail, sizeof detail, "max|d|=%.6g", maxAbs);
    }
    if (status != ST_FAIL && !r.unloadOK) {
        status = ST_FAIL;
        snprintf(detail, sizeof detail, "%s", r.err[0] ? r.err : "UNLOAD_FAIL");
    }
    int evaluated = r.compileOK && r.loadOK && r.evalOK;
    aneRecord(list, name, status, r.ms, evaluated ? (long)C * S : 0, mism, 0, detail);
    if (status == ST_PASS) {
        if (refOut) memcpy(refOut, got, outB);
        if (capOut) memcpy(capOut, got, outB);
    }

    free(W); free(want); free(got);
    return status == ST_FAIL;
}

// Concrete/exploratory case:
//   compile rejection        -> UNSUPPORTED (the form is refused at compile time)
//   compile OK, load/eval fail -> FAIL
//   compile OK, numbers differ -> FAIL
// Only a compile-time rejection is treated as UNSUPPORTED; once the compiler
// accepts the program, its runtime behaviour is a test result, not an excuse.
static int runExploreCase(const char *name, NSString *mil, ANEBlob *blob,
                          const _Float16 *x, size_t nBytes,
                          const _Float16 *want, ANECaseList *list)
{
    _Float16 *got = malloc(nBytes);
    ANEResult r = aneRunCase(mil, blob.data, x, nBytes, nBytes,
                             g_bench ? g_iters : 0, got, nBytes);
    int status;
    long mism = 0;
    double maxAbs = 0.0;
    char detail[640] = {0};
    if (!r.compileOK) {
        // UNSUPPORTED only when the compiler actually rejected the MIL.  A
        // missing private API / descriptor / file write is infrastructure FAIL.
        if (r.compileAttempted && r.compilerRejected)
            status = ST_UNSUPPORTED;
        else
            status = ST_FAIL;
        snprintf(detail, sizeof detail, "%s", r.err);
    } else if (!r.loadOK || !r.evalOK) {
        status = ST_FAIL;
        snprintf(detail, sizeof detail, "%s", r.err);
    } else {
        mism = cmpFp16(got, want, nBytes / 2, &maxAbs, detail, sizeof detail);
        status = mism ? ST_FAIL : ST_PASS;
    }
    if (status != ST_FAIL && !r.unloadOK) {
        status = ST_FAIL;
        snprintf(detail, sizeof detail, "%s", r.err[0] ? r.err : "UNLOAD_FAIL");
    }
    int evaluated = r.compileOK && r.loadOK && r.evalOK;
    aneRecord(list, name, status, r.ms, evaluated ? (long)(nBytes / 2) : 0, mism, 0, detail);
    free(got);
    return status == ST_FAIL;
}

// ---------------------------------------------------------------- offline self-test
static int stCheck(int cond, const char *what) {
    printf("  %-60s %s\n", what, cond ? "PASS" : "FAIL");
    return cond ? 0 : 1;
}

static int runSelftest(void) {
    int fail = 0;
    printf("constexpr_scale_probe self-test (no ANE calls)\n");

    ANEBlob *b = [ANEBlob new];
    _Float16 s16[8];
    for (int i = 0; i < 8; i++) s16[i] = (_Float16)(0.125 * (i + 1));
    NSUInteger off = [b put:s16 bytes:sizeof s16 dtype:ANE_BLOB_FP16];
    NSData *d = b.data;
    const uint8_t *p = d.bytes;
    fail += stCheck(off == 64 && d.length == 64 + 64 + 16, "fp16 chunk at 64, payload at 128");
    uint32_t dt; uint64_t sz, at;
    memcpy(&dt, p + 68, 4); memcpy(&sz, p + 72, 8); memcpy(&at, p + 80, 8);
    fail += stCheck(dt == ANE_BLOB_FP16 && sz == 16 && at == 128,
                    "dtype=1, uint64 size=16, uint64 offset=128");
    uint32_t count;
    memcpy(&count, p + 0, 4);
    fail += stCheck(count == 1, "BLOB file header chunk count = 1 after first put");
    uint8_t extra[8] = {0};
    NSUInteger off2 = [b put:extra bytes:sizeof extra dtype:ANE_BLOB_FP16];
    NSData *d2 = b.data;
    const uint8_t *p2 = d2.bytes;
    memcpy(&count, p2 + 0, 4);
    fail += stCheck(count == 2 && off2 == 192,
                    "BLOB chunk count increments to 2 (second chunk at 192)");

    int8_t q[64];
    for (int k = 0; k < 64; k++) q[k] = (int8_t)((k % 16) - 8);
    double scale[8], W[64];
    int zp[8];
    for (int i = 0; i < 8; i++) { scale[i] = 0.125; zp[i] = 0; }
    buildW(W, q, 8, 0, scale, zp);
    fail += stCheck(W[0] == 0.125 * (double)q[0] && W[9] == 0.125 * (double)q[9],
                    "axis=0 CPU golden W[o,i]=scale*(q-zp)");

    NSString *aff = genAffineMIL(8, 16, 0, @"fp16(0x1p-3)", @"int8(0)", 64);
    fail += stCheck([aff containsString:@"constexpr_affine_dequantize()[axis = int32(0)"],
                    "affine parameters including axis are in [ ... ]");
    fail += stCheck([aff containsString:@"quantized_data = tensor<int8, [8, 8, 1, 1]>(BLOBFILE"],
                    "affine quantized_data is a standard BLOB");
    fail += stCheck([aff containsString:@"scale = fp16(0x1p-3)"] &&
                    ![aff containsString:@"main<ios16>"], "iOS18 scalar affine text");

    NSString *bw = genBlockwiseMIL(8, 16, 64, 192);
    fail += stCheck([bw containsString:@"constexpr_blockwise_shift_scale(data = D, scale = SC)"],
                    "blockwise grammar: inputs in ( ... )");
    NSString *lut = genLUTMIL(8, 16, 64, 192);
    fail += stCheck([lut containsString:@"tensor<uint4, [8, 8, 1, 1]>"] &&
                    [lut containsString:@"tensor<fp16, [1, 1, 1, 1, 16, 1]>"] &&
                    [lut containsString:@"constexpr_lut_to_dense(indices = IDX, lut = LUT)"] &&
                    [lut rangeOfString:@"shape ="].location == NSNotFound,
                    "iOS18 LUT grammar: uint4 indices, rank-6 LUT, no shape");

    printf("self-test: %s\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}

// ---------------------------------------------------------------- offline MIL emission
static int emitMILs(void) {
    const int C = 8, S = 16;
    int8_t q[64];
    for (int k = 0; k < 64; k++) q[k] = (int8_t)((k % 16) - 8);
    ANEBlob *b = [ANEBlob new];
    NSUInteger wOff = [b put:q bytes:sizeof q dtype:ANE_BLOB_INT8];
    _Float16 rowScale[8], lut[16];
    for (int i = 0; i < 8; i++) rowScale[i] = (_Float16)0.125;
    for (int k = 0; k < 16; k++) lut[k] = (_Float16)(0.125 * (k - 8));
    NSUInteger rsOff = [b put:rowScale bytes:sizeof rowScale dtype:ANE_BLOB_FP16];
    uint8_t packed[32];
    for (int k = 0; k < 64; k += 2)
        packed[k / 2] = (uint8_t)(((q[k] + 8) & 0xF) | (((q[k + 1] + 8) & 0xF) << 4));
    NSUInteger idxOff = [b put:packed bytes:sizeof packed dtype:ANE_BLOB_UINT4];
    NSUInteger lutOff = [b put:lut bytes:sizeof lut dtype:ANE_BLOB_FP16];

    printf("#### affine scalar\n%s\n",
           [genAffineMIL(C, S, 0, @"fp16(0x1p-3)", @"int8(0)", wOff) UTF8String]);
    printf("#### affine vector axis=0\n%s\n",
           [genAffineMIL(C, S, 0, scaleVecLiteralVar(C), zpVecLiteralVar(C), wOff) UTF8String]);
    printf("#### affine vector axis=1 (exploratory)\n%s\n",
           [genAffineMIL(C, S, 1, scaleVecLiteralVar(C), zpVecLiteralVar(C), wOff) UTF8String]);
    printf("#### blockwise row scale\n%s\n", [genBlockwiseMIL(C, S, wOff, rsOff) UTF8String]);
    printf("#### LUT 16-entry uint4 iOS18\n%s\n", [genLUTMIL(C, S, idxOff, lutOff) UTF8String]);
    return 0;
}

// ---------------------------------------------------------------- main
int main(int argc, char **argv) {
    mach_timebase_info(&g_tb);
    g_json  = aneHasFlag(argc, argv, "--json");
    g_quiet = aneHasFlag(argc, argv, "--quiet");
    g_bench = aneHasFlag(argc, argv, "--bench");
    if (aneHasFlag(argc, argv, "--selftest")) return runSelftest();
    if (aneHasFlag(argc, argv, "--emit-mil")) return emitMILs();

    if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW) &&
        !dlopen("/System/Library/Frameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)) {
        fprintf(stderr, "cannot load AppleNeuralEngine\n");
        return 2;
    }
    g_dumpMIL = getenv("CSP_DUMP") != NULL;
    g_iters = getenv("CSP_ITERS") ? atoi(getenv("CSP_ITERS")) : 10;
    if (g_iters < 1) g_iters = 1;

    const int C = 8, S = 256;
    const size_t n = (size_t)C * S;

    if (!g_json && !g_quiet) {
        printf("=== ANE constexpr weight-dequantization probe ===\n");
        printf("shape %d->%d conv1x1 over [1,%d,1,%d]; exact fp16 goldens; %s\n\n",
               C, C, C, S, g_bench ? "with bench" : "correctness only");
    }

    // Deterministic int8 weights q in [-8,7]; W = scale*(q-zp) stays exactly
    // representable in fp16 for the scales used below.
    int8_t q[64];
    for (int k = 0; k < 64; k++) q[k] = (int8_t)((k % 16) - 8);

    ANEBlob *blob = [ANEBlob new];
    NSUInteger wOff = [blob put:q bytes:sizeof q dtype:ANE_BLOB_INT8];

    // Deterministic fp16 input, multiples of 0.25 in [-2,2].
    _Float16 *x = malloc(n * sizeof(_Float16));
    Rng r;
    rngSeed(&r, 0x5EED);
    for (size_t k = 0; k < n; k++) x[k] = (_Float16)((double)rngInt(&r, -8, 8) * 0.25);

    double scaleEq[8], scaleVar[8];
    int zpZero[8], zpThree[8], zpVar[8];
    for (int i = 0; i < 8; i++) {
        scaleEq[i] = 0.125;
        scaleVar[i] = ldexp(1.0, -(3 + (i % 3)));
        zpZero[i] = 0;
        zpThree[i] = 3;
        zpVar[i] = (i % 3) - 1;
    }

    ANECaseList list = {0};
    int failures = 0;
    _Float16 *refOut = malloc(n * sizeof(_Float16));
    _Float16 *eqOut = malloc(n * sizeof(_Float16));
    BOOL haveRef = NO, haveEq = NO;

    if (!g_json && !g_quiet) printf("[1] required affine dequantization (axis=0)\n");
    failures += runAffineCase("affine scalar scale, zp=0", C, S, 0,
                              @"fp16(0x1p-3)", @"int8(0)",
                              scaleEq, zpZero, q, wOff, blob, x, refOut, NULL, YES, &list);
    haveRef = (list.v[list.n - 1].status == ST_PASS);
    failures += runAffineCase("affine scalar scale, zp=3", C, S, 0,
                              @"fp16(0x1p-3)", @"int8(3)",
                              scaleEq, zpThree, q, wOff, blob, x, NULL, NULL, YES, &list);
    failures += runAffineCase("affine vector scale+zp [C], axis=0", C, S, 0,
                              scaleVecLiteralVar(C), zpVecLiteralVar(C),
                              scaleVar, zpVar, q, wOff, blob, x, NULL, NULL, YES, &list);
    failures += runAffineCase("affine all-equal vector [C], axis=0", C, S, 0,
                              scaleVecLiteralEq(C), zpVecLiteralZero(C),
                              scaleEq, zpZero, q, wOff, blob, x, NULL, eqOut, YES, &list);
    haveEq = (list.v[list.n - 1].status == ST_PASS);

    // Strong layout check: the all-equal vector case must be bit-identical to the
    // scalar control (both were also checked against the CPU golden).
    {
        int ok = haveRef && haveEq && memcmp(refOut, eqOut, n * 2) == 0;
        aneRecord(&list, "scalar vs all-equal vector bit-identical", ok ? ST_PASS : ST_FAIL,
                  0, (long)n, ok ? 0 : 1, 0,
                  ok ? "" : (haveRef && haveEq ? "outputs differ" : "reference case did not pass"));
        if (!ok) failures++;
    }

    if (!g_json && !g_quiet) printf("\n[2] concrete cases and one exploratory axis case\n");

    // axis=1: spec-legal but refused on the validated environment.  A compile
    // rejection is UNSUPPORTED; if the compiler accepts it, the output must match
    // the CPU golden, otherwise this is a FAIL (not an "exploratory" free pass).
    {
        double *W = malloc((size_t)C * C * sizeof(double));
        _Float16 *want = malloc(n * sizeof(_Float16));
        buildW(W, q, C, 1, scaleVar, zpVar);
        cpuConv(W, x, C, S, want);
        NSString *mil = genAffineMIL(C, S, 1, scaleVecLiteralVar(C), zpVecLiteralVar(C), wOff);
        failures += runExploreCase("EXPLORE affine vector axis=1", mil, blob, x, n * 2,
                                   want, &list);
        free(W); free(want);
    }

    // blockwise with one scale per output channel -> same W as the scalar case.
    if (!haveRef) {
        aneRecord(&list, "blockwise row scale (concrete)", ST_SKIP, 0, 0, 0, 0,
                  "scalar reference case did not pass");
    } else {
        _Float16 *rowScale = malloc((size_t)C * sizeof(_Float16));
        for (int i = 0; i < C; i++) rowScale[i] = (_Float16)0.125;
        NSUInteger rsOff = [blob put:rowScale bytes:(NSUInteger)C * 2 dtype:ANE_BLOB_FP16];
        free(rowScale);
        _Float16 *want = malloc(n * sizeof(_Float16));
        memcpy(want, refOut, n * sizeof(_Float16));
        NSString *mil = genBlockwiseMIL(C, S, wOff, rsOff);
        failures += runExploreCase("blockwise row scale (concrete)", mil, blob, x, n * 2,
                                   want, &list);
        free(want);
    }

    // 16-entry LUT: lut[k] = 0.125*(k-8), idx = q+8, so W == 0.125*q.
    if (!haveRef) {
        aneRecord(&list, "LUT 16-entry uint4 iOS18 (concrete)", ST_SKIP, 0, 0, 0, 0,
                  "scalar reference case did not pass");
    } else {
        uint8_t packed[32];
        for (int k = 0; k < 64; k += 2) {
            uint8_t a = (uint8_t)(q[k] + 8);
            uint8_t bb = (uint8_t)(q[k + 1] + 8);
            packed[k / 2] = (uint8_t)((a & 0xF) | ((bb & 0xF) << 4));
        }
        _Float16 lut[16];
        for (int k = 0; k < 16; k++) lut[k] = (_Float16)(0.125 * (k - 8));
        NSUInteger idxOff = [blob put:packed bytes:sizeof packed dtype:ANE_BLOB_UINT4];
        NSUInteger lutOff = [blob put:lut bytes:sizeof lut dtype:ANE_BLOB_FP16];
        _Float16 *want = malloc(n * sizeof(_Float16));
        memcpy(want, refOut, n * sizeof(_Float16));
        NSString *mil = genLUTMIL(C, S, idxOff, lutOff);
        failures += runExploreCase("LUT 16-entry uint4 iOS18 (concrete)", mil, blob, x, n * 2,
                                   want, &list);
        free(want);
    }

    int pass = 0, fail = 0, unsup = 0, skip = 0;
    for (int i = 0; i < list.n; i++) {
        switch (list.v[i].status) {
            case ST_PASS: pass++; break;
            case ST_FAIL: fail++; break;
            case ST_SKIP: skip++; break;
            default: unsup++; break;
        }
    }
    if (g_json) {
        anePrintJSON("constexpr_scale_probe", g_bench ? "bench" : "default", &list, 0, 0);
    } else if (!g_quiet) {
        printf("\nsummary: PASS=%d FAIL=%d SKIP=%d UNSUPPORTED=%d\n", pass, fail, skip, unsup);
        printf("note: axis=1 refusal is a concrete UNSUPPORTED, not a universal limit;\n"
               "      blockwise row scale and LUT16 uint4 are concrete numeric cases\n"
               "      (both PASSED on the 2026-10-18 run when accepted).\n");
        printf("result: %s\n", fail ? "FAIL" : "PASS");
    }
    free(refOut);
    free(eqOut);
    free(x);
    return (fail || failures) ? 1 : 0;
}
