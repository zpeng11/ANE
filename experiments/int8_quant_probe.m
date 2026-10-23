// int8_quant_probe.m — deterministic tests for the MIL runtime quantize / dequantize
// path on the ANE.
//
// Scope (what this file deliberately does and does not claim)
// -----------------------------------------------------------
// Legal MIL forms tested:
//   * quantize:   input fp16, output int8, scale const (scalar or rank-1 vector),
//                 axis as a *named const input* in the op's (...) argument list,
//                 zero_point const (scalar and rank-1 vector).
//   * dequantize: input int8, output fp16, same scale/axis/zero_point constraints.
//   * QDQ:        the two ops back to back on the same scale.
//   * Q / DQ / QDQ are run on the graph input and on a conv output.
//   * per-channel (axis=1, scale [C]) and per-token (axis=3, scale [S]) rank-1
//     const scales; scalar baseline; saturation and zero-point controls.
//   * axis=3 is spec-legal but the validated environment refused it.  Those three
//     cases are therefore concrete exploratory cases: a refusal is reported as
//     UNSUPPORTED, not as a universal hardware limit and not as a required FAIL.
//
// What is intentionally absent:
//   * computed / graph-produced scales (reduce_max, abs, ...).  The MIL interface
//     requires `scale` to be const; testing a non-const scale is an invalid-syntax
//     test, not a capability test, so it is not kept here.
//   * rank>1 scale tensors.  The MIL spec allows a scalar or a rank-1 vector only.
//   * end-to-end application measurements.  The synthetic convolution chains
//     check compilation, loading and execution, not full-output numerical fidelity.
//
// Golden arithmetic
// -----------------
// The audited machine rounds exact ties away from zero in int8 quantize.  That is
// the golden rule used here (roundHalfAway).  Every Q/QDQ case also reports how
// many observed values disagree with round-to-nearest-even (RNE); the test does
// not silently assume RNE and does not treat the RNE count as a failure.
// The RNE comparator is the CoreML spec form
//     q = clamp(round_half_even(x / scale) + zero_point),
// not round_half_even(x/scale + zero_point).  For odd zero_point these differ at
// ties: x=0.0625, scale=0.125, zp=5 gives spec RNE 5 but observed 6.
//
// With a nonzero zero_point the validated hardware matches
//     q = clamp(round_half_away(x / scale + zero_point))
// (zero_point added before the tie is resolved), not
//     clamp(round_half_away(x / scale) + zero_point).
// The two differ at ties whose sign changes when the integer zero_point is added.
// Each zero_point case reports both the golden mismatch count and the add-after
// mismatch count; the latter is expected to be nonzero and is not a failure.
//
// Determinism and exit status
// ---------------------------
// All inputs and weights come from a seeded PRNG.  A required case that fails to
// compile/load/evaluate, or that produces a different number, is a FAIL.  A
// concrete exploratory case (axis=3 or rank-1 zero_point) whose MIL the compiler
// rejects is UNSUPPORTED; if the compiler accepts it, it must then load/evaluate
// and match numerically, otherwise it is FAIL.  The process exits nonzero if and
// only if at least one FAIL is recorded.
//
// Build / run
// -----------
//   make int8_quant_probe
//   ./int8_quant_probe              # quick: correctness + acceptance, no timing
//   ./int8_quant_probe --bench      # adds bounded timing loops
//   ./int8_quant_probe --json       # one machine-readable JSON object
//   ./int8_quant_probe --selftest   # offline arithmetic/MIL/blob checks, no ANE
//
// Environment: IQP_DUMP=1 prints each generated MIL, IQP_ITERS=n sets bench
// iterations (default 3).
#import "ane_quant_common.h"

// ---------------------------------------------------------------- case model
enum { OP_Q = 0, OP_DQ = 1, OP_QDQ = 2 };
enum { GRAN_SCALAR = 0, GRAN_CHANNEL = 1, GRAN_TOKEN = 2 };
enum { ZP_NONE = 0, ZP_SCALAR = 1, ZP_CHANNEL = 2 };

typedef struct {
    const char *name;
    int op, gran, zp, sat, onConv, dqScaleShift, required;
} QuantCase;

// Nonuniform but exactly representable powers of two.  Per-channel index uses
// the channel, per-token index uses the spatial position.
static int axisScaleExp(int idx) { return 3 + (idx % 3); }
static double axisScaleValue(int idx) { return ldexp(1.0, -axisScaleExp(idx)); }
static int axisZeroPoint(int idx) { return (idx % 3) - 1; }

static double scaleAt(int gran, int c, int s) {
    if (gran == GRAN_CHANNEL) return axisScaleValue(c);
    if (gran == GRAN_TOKEN) return axisScaleValue(s);
    return ldexp(1.0, -3);            // 0.125
}

static int zpAt(int mode, int c) {
    if (mode == ZP_SCALAR) return 5;
    if (mode == ZP_CHANNEL) return axisZeroPoint(c);
    return 0;
}

static NSString *scaleVecLiteral(int n) {
    NSMutableString *s = [NSMutableString stringWithFormat:@"tensor<fp16, [%d]>([", n];
    for (int i = 0; i < n; i++)
        [s appendFormat:@"0x1p-%d%@", axisScaleExp(i), i + 1 == n ? @"" : @", "];
    [s appendString:@"])"];
    return s;
}

static NSString *zpVecLiteral(int n) {
    NSMutableString *s = [NSMutableString stringWithFormat:@"tensor<int8, [%d]>([", n];
    for (int i = 0; i < n; i++)
        [s appendFormat:@"%d%@", axisZeroPoint(i), i + 1 == n ? @"" : @", "];
    [s appendString:@"])"];
    return s;
}

static BOOL caseInputIsInt8(const QuantCase *tc) {
    return tc->op == OP_DQ && !tc->onConv;
}

// ---------------------------------------------------------------- MIL generation
// The conv in the on-conv cases is an exact signed cyclic shift:
//     W[o][i] = +1 or -1   where i == (o + 1) % C
// so conv output o is exactly +/- input channel (o+1)%C.  That keeps the CPU
// golden independent of fp16 accumulation ordering.
static NSString *genRuntimeMIL(const QuantCase *tc, int C, int S, NSUInteger wOff) {
    NSMutableString *m = [NSMutableString stringWithString:aneMilHeader()];
    BOOL in8 = caseInputIsInt8(tc);
    [m appendFormat:@"    func main<ios18>(tensor<%@, [1, %d, 1, %d]> x) {\n",
        in8 ? @"int8" : @"fp16", C, S];

    NSString *prev = @"x";
    if (tc->onConv) {
        [m appendString:aneConvConsts()];
        [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), "
            "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), "
            "offset = uint64(%lu)))];\n",
            C, C, C, C, (unsigned long)wOff];
        [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> cv = conv(dilations = c_dilations, "
            "groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, "
            "weight = W, x = x)[name = string(\"cv\")];\n", C, S];
        prev = @"cv";
    }

    if (tc->op != OP_DQ) {
        [m appendString:@"        tensor<string, []> qd = const()[name = string(\"qd\"), "
                          "val = tensor<string, []>(\"int8\")];\n"];
    }

    if (tc->gran == GRAN_SCALAR) {
        [m appendString:@"        fp16 sc = const()[name = string(\"sc\"), val = fp16(0x1p-3)];\n"];
    } else {
        int n = (tc->gran == GRAN_CHANNEL) ? C : S;
        [m appendFormat:@"        tensor<fp16, [%d]> sc = const()[name = string(\"sc\"), val = %@];\n",
            n, scaleVecLiteral(n)];
        int axis = (tc->gran == GRAN_CHANNEL) ? 1 : 3;
        [m appendFormat:@"        tensor<int32, []> ax = const()[name = string(\"ax\"), "
            "val = tensor<int32, []>(%d)];\n", axis];
    }

    if (tc->zp == ZP_SCALAR) {
        [m appendString:@"        int8 zp = const()[name = string(\"zp\"), val = int8(5)];\n"];
    } else if (tc->zp == ZP_CHANNEL) {
        [m appendFormat:@"        tensor<int8, [%d]> zp = const()[name = string(\"zp\"), val = %@];\n",
            C, zpVecLiteral(C)];
    }

    NSString *dqScale = @"sc";
    if (tc->dqScaleShift) {
        [m appendString:@"        fp16 dqs = const()[name = string(\"dqs\"), val = fp16(0x1p-2)];\n"];
        dqScale = @"dqs";
    }

    NSString *qName = nil;
    if (tc->op == OP_Q || tc->op == OP_QDQ) {
        NSMutableString *args = [NSMutableString stringWithFormat:@"input = %@, scale = sc", prev];
        if (tc->gran != GRAN_SCALAR) [args appendString:@", axis = ax"];
        if (tc->zp != ZP_NONE)      [args appendString:@", zero_point = zp"];
        [args appendString:@", output_dtype = qd"];
        [m appendFormat:@"        tensor<int8, [1, %d, 1, %d]> qv = quantize(%@)[name = string(\"qv\")];\n",
            C, S, args];
        qName = @"qv";
    }

    if (tc->op == OP_QDQ || tc->op == OP_DQ) {
        NSString *from = (tc->op == OP_DQ) ? prev : qName;
        NSMutableString *args = [NSMutableString stringWithFormat:@"input = %@, scale = %@", from, dqScale];
        if (tc->gran != GRAN_SCALAR) [args appendString:@", axis = ax"];
        if (tc->zp != ZP_NONE)      [args appendString:@", zero_point = zp"];
        [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> dq = dequantize(%@)[name = string(\"dq\")];\n",
            C, S, args];
    }

    NSString *out = (tc->op == OP_Q) ? @"qv" : @"dq";
    [m appendFormat:@"    } -> (%@);\n}\n", out];
    return m;
}

// ---------------------------------------------------------------- CPU golden
static double convOutAt(const _Float16 *x, int C, int S, int c, int s) {
    int src = (c + 1) % C;
    double sign = (c & 1) ? -1.0 : 1.0;
    return sign * (double)x[(size_t)src * (size_t)S + (size_t)s];
}

static _Float16 *genActivation(int C, int S, int gran, int sat, uint64_t seed) {
    _Float16 *v = malloc((size_t)C * (size_t)S * sizeof(_Float16));
    Rng r;
    rngSeed(&r, seed);
    // Values that exercise both sides of the int8 saturation boundary.  All are
    // exact in fp16 and in the 0.125 grid.
    static const double edge[] = {
        127.0, 127.5, 128.0, 128.5, 200.0, -127.0, -127.5, -128.0, -128.5,
        -200.0, 126.5, -126.5, 0.5, -0.5, 2.5, -2.5
    };
    const int nedge = (int)(sizeof edge / sizeof edge[0]);
    for (int c = 0; c < C; c++) {
        for (int s = 0; s < S; s++) {
            double x;
            if (sat) {
                x = edge[(c * S + s) % nedge] * 0.125;
            } else {
                double m = (double)rngInt(&r, -250, 250) / 2.0;   // half-integer
                x = m * scaleAt(gran, c, s);
            }
            v[(size_t)c * (size_t)S + (size_t)s] = (_Float16)x;
        }
    }
    return v;
}

static int8_t *genQInput(int C, int S, uint64_t seed) {
    int8_t *v = malloc((size_t)C * (size_t)S);
    Rng r;
    rngSeed(&r, seed);
    static const int edge[] = { -128, 127, -127, 126, 0, 1, -1, 64 };
    for (int c = 0; c < C; c++)
        for (int s = 0; s < S; s++)
            v[(size_t)c * (size_t)S + (size_t)s] =
                (c * S + s < 8) ? (int8_t)edge[c * S + s]
                                : (int8_t)rngInt(&r, -128, 127);
    return v;
}

// ---------------------------------------------------------------- one quantization case
// Returns 1 when a FAIL was recorded (i.e. the process should exit nonzero).
static int runQuantCase(const QuantCase *tc, int C, int S,
                        ANEBlob *convBlob, NSUInteger wOff,
                        ANECaseList *list, long *rneTotal, long *zpAltTotal,
                        uint64_t seed, int timedIters)
{
    BOOL in8 = caseInputIsInt8(tc);
    _Float16 *act = NULL;
    int8_t *qin = NULL;
    size_t inBytes;
    if (in8) {
        qin = genQInput(C, S, seed);
        inBytes = (size_t)C * (size_t)S;
    } else {
        act = genActivation(C, S, tc->gran, tc->sat, seed);
        inBytes = (size_t)C * (size_t)S * 2;
    }
    size_t outBytes = (size_t)C * (size_t)S * (tc->op == OP_Q ? 1 : 2);
    void *out = calloc(1, outBytes);

    NSString *mil = genRuntimeMIL(tc, C, S, wOff);
    ANEResult r = aneRunCase(mil, tc->onConv ? convBlob.data : nil,
                             in8 ? (const void *)qin : (const void *)act,
                             inBytes, outBytes, timedIters, out, outBytes);

    int status;
    long mism = 0, rne = 0, altMism = 0;
    double maxAbs = 0.0;
    char detail[640] = {0};
    if (!r.compileOK) {
        // UNSUPPORTED is reserved for an actual compiler rejection.  Private API
        // lookup, descriptor creation, temp-dir/file writes and other setup
        // failures are infrastructure failures (FAIL).
        if (r.compileAttempted && r.compilerRejected && !tc->required)
            status = ST_UNSUPPORTED;
        else
            status = ST_FAIL;
        snprintf(detail, sizeof detail, "%s", r.err);
    } else if (!r.loadOK || !r.evalOK) {
        // The compiler accepted the program; a later failure is a real failure.
        status = ST_FAIL;
        snprintf(detail, sizeof detail, "%s", r.err);
    } else {
        if (tc->op == OP_Q) {
            const int8_t *got = (const int8_t *)out;
            for (int c = 0; c < C; c++) {
                for (int s = 0; s < S; s++) {
                    double x = tc->onConv ? convOutAt(act, C, S, c, s)
                                          : (double)act[(size_t)c * S + s];
                    double sc = scaleAt(tc->gran, c, s);
                    int zp = zpAt(tc->zp, c);
                    int exp = qReference(x, sc, zp);
                    int expAlt = qReferenceAddAfter(x, sc, zp);
                    int expR = qReferenceRNE(x, sc, zp);
                    int g = got[(size_t)c * S + s];
                    if (g != exp) {
                        if (mism == 0)
                            snprintf(detail, sizeof detail, "first[c=%d,s=%d] got=%d want=%d",
                                     c, s, g, exp);
                        mism++;
                    }
                    if (g != expAlt) altMism++;
                    if (g != expR) rne++;
                    double d = fabs((double)g - (double)exp);
                    if (d > maxAbs) maxAbs = d;
                }
            }
        } else {
            const _Float16 *got = (const _Float16 *)out;
            for (int c = 0; c < C; c++) {
                for (int s = 0; s < S; s++) {
                    double sc = scaleAt(tc->gran, c, s);
                    double dsc = tc->dqScaleShift ? sc * 2.0 : sc;
                    int zp = zpAt(tc->zp, c);
                    double g = (double)got[(size_t)c * S + s];
                    double exp, expR = 0.0;
                    if (tc->op == OP_DQ) {
                        int q = qin[(size_t)c * S + s];
                        exp = dqReference(q, dsc, zp);
                    } else {
                        double x = tc->onConv ? convOutAt(act, C, S, c, s)
                                              : (double)act[(size_t)c * S + s];
                        int qe = qReference(x, sc, zp);
                        exp = dqReference(qe, dsc, zp);
                        int qeAlt = qReferenceAddAfter(x, sc, zp);
                        double expAlt = dqReference(qeAlt, dsc, zp);
                        int qeR = qReferenceRNE(x, sc, zp);
                        expR = dqReference(qeR, dsc, zp);
                        if (g != expAlt) altMism++;
                    }
                    if (g != exp) {
                        if (mism == 0)
                            snprintf(detail, sizeof detail, "first[c=%d,s=%d] got=%.6g want=%.6g",
                                     c, s, g, exp);
                        mism++;
                    }
                    if (tc->op == OP_QDQ && g != expR) rne++;
                    double d = fabs(g - exp);
                    if (d > maxAbs) maxAbs = d;
                }
            }
        }
        status = mism ? ST_FAIL : ST_PASS;
        if (tc->zp != ZP_NONE && tc->op != OP_DQ) {
            // Expose the discriminating formula count rather than hiding it.
            if (mism == 0)
                snprintf(detail, sizeof detail,
                         "zp add-before OK; add-after mism=%ld", altMism);
            else {
                size_t used = strlen(detail);
                snprintf(detail + used, sizeof detail - used,
                         "%szp add-after mism=%ld", used ? "; " : "", altMism);
            }
        } else if (mism && detail[0] == 0) {
            snprintf(detail, sizeof detail, "max|d|=%.6g", maxAbs);
        }
        *rneTotal += rne;
        *zpAltTotal += altMism;
    }
    // A model that loaded but failed to unload is an infrastructure failure; it
    // must not remain PASS.  If the case already failed, preserve its status/detail.
    if (status != ST_FAIL && !r.unloadOK) {
        status = ST_FAIL;
        snprintf(detail, sizeof detail, "%s", r.err[0] ? r.err : "UNLOAD_FAIL");
    }

    int evaluated = r.compileOK && r.loadOK && r.evalOK;
    aneRecord(list, tc->name, status, r.ms, evaluated ? (long)C * S : 0,
              mism, rne, detail);
    free(out);
    free(act);
    free(qin);
    return status == ST_FAIL;
}

// ---------------------------------------------------------------- conv-chain acceptance
// This section is acceptance, not fidelity: it checks that a concrete graph form
// compiles, loads and evaluates at fixed synthetic shapes.  No CPU numeric comparison
// is claimed here.  Channel bookkeeping is explicit: layer i's input channels are
// exactly layer i-1's output channels, and the output surface is sized from the
// final channel count the generator actually produced.
typedef enum { GK_CHAIN = 0, GK_BOWTIE = 1, GK_WIDE = 2 } GraphKind;

static NSString *genGraphMIL(GraphKind kind, int Cin, int Cout, int M, int depth,
                             int mode, ANEBlob *blob, int *outC, double *flops)
{
    Rng rw;
    rngSeed(&rw, 0xC0FFEEull ^ ((uint64_t)Cin << 32) ^ ((uint64_t)Cout << 8) ^
                 (uint64_t)M ^ ((uint64_t)depth << 48));
    NSMutableString *m = [NSMutableString stringWithString:aneMilHeader()];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", Cin, M];
    [m appendString:aneConvConsts()];

    int prevC = Cin;
    NSString *prev = @"x";
    double f = 0.0;
    for (int i = 0; i < depth; i++) {
        int il = prevC, ol;
        if (kind == GK_BOWTIE)      ol = (i % 2 == 0) ? Cout : Cin;
        else if (kind == GK_WIDE)   ol = (i == 0) ? Cin : Cout;
        else                        ol = Cout;
        f += 2.0 * (double)M * (double)il * (double)ol;

        NSUInteger n = (NSUInteger)il * (NSUInteger)ol;
        NSUInteger off;
        if (mode == 0) {
            _Float16 *w = malloc(n * 2);
            for (NSUInteger k = 0; k < n; k++)
                w[k] = (_Float16)((double)rngInt(&rw, -127, 127) * 0.0078125);
            off = [blob put:w bytes:n * 2 dtype:ANE_BLOB_FP16];
            free(w);
            [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> W%d = const()[name = string(\"W%d\"), "
                "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), "
                "offset = uint64(%lu)))];\n",
                ol, il, i, i, ol, il, (unsigned long)off];
        } else {
            int8_t *q = malloc(n);
            for (NSUInteger k = 0; k < n; k++)
                q[k] = (int8_t)rngInt(&rw, -127, 127);
            off = [blob put:q bytes:n dtype:ANE_BLOB_INT8];
            free(q);
            [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> W%d = "
                "constexpr_affine_dequantize()[axis = int32(0), name = string(\"W%d\"), "
                "quantized_data = tensor<int8, [%d, %d, 1, 1]>(BLOBFILE(path = "
                "string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu))), "
                "scale = fp16(0x1p-7), zero_point = int8(0)];\n",
                ol, il, i, i, ol, il, (unsigned long)off];
        }

        [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> c%d = conv(dilations = c_dilations, "
            "groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, "
            "weight = W%d, x = %@)[name = string(\"c%d\")];\n",
            ol, M, i, i, prev, i];
        prevC = ol;
        prev = [NSString stringWithFormat:@"c%d", i];
        // mode 0 (fp16) and mode 1 (int8 weights, fp16 activations) have no
        // activation quantize/dequantize; only modes 2 and 3 insert Q/DQ.
        if (mode <= 1 || i == depth - 1) continue;

        // Activation Q/DQ between layers; the next conv consumes d%d.
        [m appendFormat:@"        tensor<string, []> qd%d = const()[name = string(\"qd%d\"), "
            "val = tensor<string, []>(\"int8\")];\n", i, i];
        if (mode == 2) {
            [m appendFormat:@"        fp16 qs%d = const()[name = string(\"qs%d\"), val = fp16(0x1p-3)];\n",
                i, i];
            [m appendFormat:@"        tensor<int8, [1, %d, 1, %d]> q%d = quantize(input = c%d, "
                "scale = qs%d, output_dtype = qd%d)[name = string(\"q%d\")];\n",
                ol, M, i, i, i, i, i];
            [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> d%d = dequantize(input = q%d, "
                "scale = qs%d)[name = string(\"d%d\")];\n",
                ol, M, i, i, i, i];
        } else {
            [m appendFormat:@"        tensor<fp16, [%d]> qs%d = const()[name = string(\"qs%d\"), "
                "val = %@];\n", ol, i, i, scaleVecLiteral(ol)];
            [m appendFormat:@"        tensor<int32, []> ax%d = const()[name = string(\"ax%d\"), "
                "val = tensor<int32, []>(1)];\n", i, i];
            [m appendFormat:@"        tensor<int8, [1, %d, 1, %d]> q%d = quantize(input = c%d, "
                "scale = qs%d, axis = ax%d, output_dtype = qd%d)[name = string(\"q%d\")];\n",
                ol, M, i, i, i, i, i, i];
            [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> d%d = dequantize(input = q%d, "
                "scale = qs%d, axis = ax%d)[name = string(\"d%d\")];\n",
                ol, M, i, i, i, i, i];
        }
        prev = [NSString stringWithFormat:@"d%d", i];
    }
    [m appendFormat:@"    } -> (%@);\n}\n", prev];
    *outC = prevC;
    *flops = f;
    return m;
}

static _Float16 *genRandomActivation(size_t n, uint64_t seed) {
    _Float16 *v = malloc(n * sizeof(_Float16));
    Rng r;
    rngSeed(&r, seed);
    for (size_t k = 0; k < n; k++)
        v[k] = (_Float16)((double)rngInt(&r, -1000, 1000) / 1000.0 * 0.5);
    return v;
}

static const char *graphModeName(int mode) {
    switch (mode) {
        case 0: return "fp16";
        case 1: return "int8W-fp16A";
        case 2: return "W8A8-per-tensor";
        default: return "W8A8-per-channel";
    }
}

static int runGraphCase(const char *name, GraphKind kind, int Cin, int Cout, int M,
                        int depth, int mode, ANECaseList *list, int iters)
{
    ANEBlob *blob = [ANEBlob new];
    int outC = 0;
    double flops = 0.0;
    NSString *mil = genGraphMIL(kind, Cin, Cout, M, depth, mode, blob, &outC, &flops);
    size_t inBytes = (size_t)Cin * (size_t)M * 2;
    size_t outBytes = (size_t)outC * (size_t)M * 2;
    _Float16 *act = genRandomActivation((size_t)Cin * (size_t)M, 0x9000ull + (uint64_t)M);
    ANEResult r = aneRunCase(mil, blob.data, act, inBytes, outBytes,
                             g_bench ? iters : 0, NULL, 0);
    int ok = r.compileOK && r.loadOK && r.evalOK && r.unloadOK;
    char detail[640] = {0};
    if (!ok) snprintf(detail, sizeof detail, "%s", r.err);
    else snprintf(detail, sizeof detail, "ACCEPTANCE_ONLY; no CPU golden");
    if (ok && g_bench && r.ms > 0.0)
        snprintf(detail, sizeof detail, "ACCEPTANCE_ONLY; %.1f GFLOP/s", flops / r.ms / 1e6);
    aneRecord(list, name, ok ? ST_PASS : ST_FAIL, r.ms, (long)outC * M, 0, 0, detail);
    free(act);
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------- offline self-test
static int selftestCheck(int cond, const char *what) {
    printf("  %-58s %s\n", what, cond ? "PASS" : "FAIL");
    return cond ? 0 : 1;
}

static int runSelftest(void) {
    int fail = 0;
    printf("int8_quant_probe self-test (no ANE calls)\n");

    // Standard v2 blob layout: fp16=1, int8=4, uint64 size/offset, no +10 field.
    ANEBlob *b = [ANEBlob new];
    int8_t p[16];
    for (int i = 0; i < 16; i++) p[i] = (int8_t)i;
    NSUInteger off = [b put:p bytes:16 dtype:ANE_BLOB_INT8];
    NSData *d = b.data;
    const uint8_t *q = d.bytes;
    fail += selftestCheck(off == 64 && d.length == 144, "int8 chunk at 64, payload at 128, size 144");
    fail += selftestCheck(q[64] == 0xEF && q[65] == 0xBE && q[66] == 0xAD && q[67] == 0xDE,
                          "chunk magic 0xDEADBEEF");
    uint32_t dt; uint64_t sz, at;
    memcpy(&dt, q + 68, 4); memcpy(&sz, q + 72, 8); memcpy(&at, q + 80, 8);
    fail += selftestCheck(dt == ANE_BLOB_INT8 && sz == 16 && at == 128,
                          "dtype=4, uint64 size=16, uint64 offset=128");
    fail += selftestCheck(q[128] == 0 && q[143] == 15, "int8 payload intact");
    uint32_t count;
    memcpy(&count, q + 0, 4);
    fail += selftestCheck(count == 1, "BLOB file header chunk count = 1 after first put");
    uint8_t extra[8] = {0};
    NSUInteger off2 = [b put:extra bytes:sizeof extra dtype:ANE_BLOB_FP16];
    NSData *d2 = b.data;
    const uint8_t *q2 = d2.bytes;
    memcpy(&count, q2 + 0, 4);
    fail += selftestCheck(count == 2 && off2 == 192,
                          "BLOB chunk count increments to 2 (second chunk at 192)");

    // Observed tie behaviour used as the golden rule, plus the RNE counter.
    fail += selftestCheck(qReference(0.0625, 0.125, 0) == 1, "Q tie +0.5 rounds away");
    fail += selftestCheck(qReference(-0.0625, 0.125, 0) == -1, "Q tie -0.5 rounds away");
    fail += selftestCheck(qReference(0.3125, 0.125, 0) == 3, "Q tie 2.5 rounds away");
    fail += selftestCheck(qReferenceRNE(0.0625, 0.125, 0) == 0 &&
                          qReferenceRNE(0.3125, 0.125, 0) == 2, "RNE counter sees even ties");
    fail += selftestCheck(qReference(0.0625, 0.125, 5) == 6 &&
                          qReferenceRNE(0.0625, 0.125, 5) == 5,
                          "odd zp tie +0.5: observed=6, specRNE=5");
    fail += selftestCheck(qReference(128.5, 0.125, 0) == 127 &&
                          qReference(-128.5, 0.125, 0) == -128, "int8 saturation clamps");
    fail += selftestCheck(dqReference(127, 0.125, 5) == 15.25, "dequantize (q-zp)*scale");
    // Nonzero zero_point is added before the tie is resolved; the add-after form
    // differs at sign-crossing ties and is retained only for the count.
    fail += selftestCheck(qReference(-0.4375, 0.125, 5) == 2 &&
                          qReferenceAddAfter(-0.4375, 0.125, 5) == 1,
                          "zp=5 tie -3.5: add-before=2, add-after=1");
    fail += selftestCheck(qReference(0.0625, 0.125, -1) == -1 &&
                          qReferenceAddAfter(0.0625, 0.125, -1) == 0,
                          "zp=-1 tie +0.5: add-before=-1, add-after=0");

    // Generated MIL keeps axis as a named const input, not an attribute.
    QuantCase tc = { "selftest", OP_QDQ, GRAN_CHANNEL, ZP_NONE, 0, 0, 0, 1 };
    NSString *mil = genRuntimeMIL(&tc, 8, 16, 64);
    fail += selftestCheck([mil containsString:@"axis = ax"], "axis passed as named input");
    fail += selftestCheck([mil rangeOfString:@"axis = int32"].location == NSNotFound,
                          "axis not emitted as an attribute");
    fail += selftestCheck([mil containsString:@"tensor<fp16, [8]> sc = const()"],
                          "rank-1 scale is a const tensor");
    fail += selftestCheck([mil containsString:@"tensor<int8, [1, 8, 1, 16]> qv = quantize"],
                          "quantize output dtype/shape");

    // DQ consumes an int8 graph input; no quantize is emitted.
    QuantCase dq = { "selftest", OP_DQ, GRAN_TOKEN, ZP_NONE, 0, 0, 0, 1 };
    NSString *dmil = genRuntimeMIL(&dq, 8, 16, 0);
    fail += selftestCheck([dmil containsString:@"tensor<int8, [1, 8, 1, 16]> x"] &&
                          [dmil containsString:@"axis = ax"] &&
                          [dmil rangeOfString:@"= quantize("].location == NSNotFound,
                          "DQ on int8 input with rank-1 token scale");

    // Graph bookkeeping: a bowtie must end on Cin, a wide form on Cout, and the
    // next conv must consume the QDQ output (d0), not the raw conv output (c0).
    {
        ANEBlob *gb = [ANEBlob new];
        int oc = 0;
        double fl = 0;
        NSString *bow = genGraphMIL(GK_BOWTIE, 8, 16, 4, 2, 2, gb, &oc, &fl);
        fail += selftestCheck(oc == 8 && [bow containsString:@"tensor<fp16, [1, 8, 1, 4]> c1"] &&
                              [bow containsString:@"x = d0"] && [bow containsString:@"-> (c1)"],
                              "bowtie chain: Cin follows previous output, ends on Cin");
        NSString *wide = genGraphMIL(GK_WIDE, 8, 16, 4, 2, 2, gb, &oc, &fl);
        fail += selftestCheck(oc == 16 && [wide containsString:@"tensor<fp16, [1, 16, 1, 4]> c1"],
                              "square-then-wide chain: output allocation uses final Cout");

        // mode 1 is int8 weights with fp16 activations: it must not emit Q/DQ.
        ANEBlob *gb1 = [ANEBlob new];
        NSString *m1 = genGraphMIL(GK_CHAIN, 8, 16, 4, 2, 1, gb1, &oc, &fl);
        fail += selftestCheck([m1 containsString:@"constexpr_affine_dequantize"] &&
                              [m1 rangeOfString:@"= quantize("].location == NSNotFound &&
                              [m1 rangeOfString:@"= dequantize("].location == NSNotFound,
                              "mode 1 (int8W-fp16A) has no activation Q/DQ");
    }

    printf("self-test: %s\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}

// ---------------------------------------------------------------- offline MIL emission
static int emitMILs(void) {
    const int C = 8, S = 16;
    ANEBlob *b = [ANEBlob new];
    _Float16 *w = malloc((size_t)C * C * 2);
    for (int o = 0; o < C; o++)
        for (int i = 0; i < C; i++)
            w[(size_t)o * C + i] = (i == (o + 1) % C)
                ? ((o & 1) ? (_Float16)-1.0 : (_Float16)1.0) : (_Float16)0.0;
    NSUInteger off = [b put:w bytes:(NSUInteger)C * C * 2 dtype:ANE_BLOB_FP16];
    free(w);

    QuantCase cases[] = {
        { "Q scalar input",            OP_Q,   GRAN_SCALAR,  ZP_NONE,   0, 0, 0, 1 },
        { "Q channel input",           OP_Q,   GRAN_CHANNEL, ZP_NONE,   0, 0, 0, 1 },
        { "DQ token input",            OP_DQ,  GRAN_TOKEN,   ZP_NONE,   0, 0, 0, 1 },
        { "QDQ scalar zp=5 input",     OP_QDQ, GRAN_SCALAR,  ZP_SCALAR, 0, 0, 0, 1 },
        { "Q conv output",             OP_Q,   GRAN_SCALAR,  ZP_NONE,   0, 1, 0, 1 },
        { "QDQ channel conv output",   OP_QDQ, GRAN_CHANNEL, ZP_NONE,   0, 1, 0, 1 },
    };
    for (unsigned i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        printf("#### %s\n%s\n", cases[i].name, [genRuntimeMIL(&cases[i], C, S, off) UTF8String]);
    }
    for (int mode = 0; mode < 4; mode++) {
        ANEBlob *gb = [ANEBlob new];
        int oc = 0; double fl = 0;
        NSString *mil = genGraphMIL(GK_BOWTIE, 8, 16, 4, 2, mode, gb, &oc, &fl);
        printf("#### graph bowtie mode=%d outC=%d\n%s\n", mode, oc, [mil UTF8String]);
    }
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

    if (!dlopen("/System/Library/Frameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW) &&
        !dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)) {
        fprintf(stderr, "cannot load AppleNeuralEngine\n");
        return 2;
    }
    g_dumpMIL = getenv("IQP_DUMP") != NULL;
    int iters = getenv("IQP_ITERS") ? atoi(getenv("IQP_ITERS")) : 3;
    if (iters < 1) iters = 1;

    if (!g_json && !g_quiet) {
        printf("=== ANE int8 runtime quantize/dequantize probe ===\n");
        printf("golden: round-half-away (observed); RNE differences counted, not failed\n");
        printf("zero_point golden: clamp(round_away(x/scale + zp)); add-after count exposed\n");
        printf("mode: %s; input data: seeded SplitMix64 (deterministic)\n\n",
               g_bench ? "correctness + bench" : "correctness only");
    }

    const int C = 8, S = 256;

    // Exact signed-cyclic-shift conv used by the on-conv cases.
    ANEBlob *convBlob = [ANEBlob new];
    NSUInteger convOff = 0;
    {
        _Float16 *w = malloc((size_t)C * C * 2);
        for (int o = 0; o < C; o++)
            for (int i = 0; i < C; i++)
                w[(size_t)o * C + i] = (i == (o + 1) % C)
                    ? ((o & 1) ? (_Float16)-1.0 : (_Float16)1.0) : (_Float16)0.0;
        convOff = [convBlob put:w bytes:(NSUInteger)C * C * 2 dtype:ANE_BLOB_FP16];
        free(w);
    }

    QuantCase cases[] = {
        // On the graph input.
        { "Q  scalar   input",                 OP_Q,   GRAN_SCALAR,  ZP_NONE,   0, 0, 0, 1 },
        { "DQ scalar   input",                 OP_DQ,  GRAN_SCALAR,  ZP_NONE,   0, 0, 0, 1 },
        { "QDQ scalar  input",                 OP_QDQ, GRAN_SCALAR,  ZP_NONE,   0, 0, 0, 1 },
        { "Q  per-channel axis=1 [C] input",   OP_Q,   GRAN_CHANNEL, ZP_NONE,   0, 0, 0, 1 },
        { "DQ per-channel axis=1 [C] input",   OP_DQ,  GRAN_CHANNEL, ZP_NONE,   0, 0, 0, 1 },
        { "QDQ per-channel axis=1 [C] input",  OP_QDQ, GRAN_CHANNEL, ZP_NONE,   0, 0, 0, 1 },
        // axis=3 is spec-legal.  The validated environment refused it, so these
        // three are concrete exploratory cases: refusal -> UNSUPPORTED, not a
        // required failure and not a universal hardware limit.
        { "Q  per-token axis=3 [S] input",     OP_Q,   GRAN_TOKEN,   ZP_NONE,   0, 0, 0, 0 },
        { "DQ per-token axis=3 [S] input",     OP_DQ,  GRAN_TOKEN,   ZP_NONE,   0, 0, 0, 0 },
        { "QDQ per-token axis=3 [S] input",    OP_QDQ, GRAN_TOKEN,   ZP_NONE,   0, 0, 0, 0 },
        // zero_point is added before the tie is resolved; see qReference and the
        // "add-after mism" count in each detail string.
        { "Q  scalar + zero_point=5 input",   OP_Q,   GRAN_SCALAR,  ZP_SCALAR, 0, 0, 0, 1 },
        { "DQ scalar + zero_point=5 input",   OP_DQ,  GRAN_SCALAR,  ZP_SCALAR, 0, 0, 0, 1 },
        { "Q  per-channel + vector zp input", OP_Q,   GRAN_CHANNEL, ZP_CHANNEL,0, 0, 0, 0 },
        { "DQ per-channel + vector zp input", OP_DQ,  GRAN_CHANNEL, ZP_CHANNEL,0, 0, 0, 0 },
        { "QDQ per-channel + vector zp input",OP_QDQ, GRAN_CHANNEL, ZP_CHANNEL,0, 0, 0, 0 },
        { "Q  saturation scalar input",       OP_Q,   GRAN_SCALAR,  ZP_NONE,   1, 0, 0, 1 },
        { "QDQ saturation scalar input",      OP_QDQ, GRAN_SCALAR,  ZP_NONE,   1, 0, 0, 1 },
        // On a conv output (conv -> Q / DQ / QDQ).
        { "Q  scalar   conv output",           OP_Q,   GRAN_SCALAR,  ZP_NONE,   0, 1, 0, 1 },
        { "QDQ scalar  conv output",           OP_QDQ, GRAN_SCALAR,  ZP_NONE,   0, 1, 0, 1 },
        { "DQ conv output: Q 1/8 -> DQ 1/4",   OP_QDQ, GRAN_SCALAR,  ZP_NONE,   0, 1, 1, 1 },
        { "QDQ per-channel axis=1 conv output",OP_QDQ, GRAN_CHANNEL, ZP_NONE,   0, 1, 0, 1 },
    };
    const int NC = (int)(sizeof cases / sizeof cases[0]);

    ANECaseList list = {0};
    long rneTotal = 0, zpAltTotal = 0;
    int failures = 0;

    if (!g_json && !g_quiet) printf("[1] runtime Q/DQ/QDQ, shape [1,%d,1,%d]\n", C, S);
    for (int i = 0; i < NC; i++)
        failures += runQuantCase(&cases[i], C, S, convBlob, convOff, &list, &rneTotal,
                                 &zpAltTotal,
                                 0x10000ull + (uint64_t)i * 0x9E3779B9ull,
                                 g_bench ? iters : 0);

    // ------------------------------------------------------------ acceptance
    // These are conv1x1 chains, not complete FFN/ViT modules.  The first four are
    // Cin->Cout followed by (depth-1) Cout->Cout layers: the 4096->1024 case is a
    // contraction (output channels 1024), not an FFN expansion.
    if (!g_json && !g_quiet)
        printf("\n[2] synthetic-shape and graph-form acceptance (not numerical fidelity)\n");
    struct { const char *name; GraphKind kind; int Cin, Cout, M, depth; } shapes[] = {
        { "conv1x1 4096->1024->1024->1024 M=96",  GK_CHAIN,  4096, 1024,   96, 3 },
        { "conv1x1 1024->1024->1024->1024 M=96",  GK_CHAIN,  1024, 1024,   96, 3 },
        { "conv1x1 4096->1024->1024->1024 M=256", GK_CHAIN,  4096, 1024,  256, 3 },
        { "conv1x1 512->512->512->512 M=4096",    GK_CHAIN,   512,  512, 4096, 3 },
        { "conv1x1 bowtie 768->2048->768 M=256",  GK_BOWTIE,  768, 2048,  256, 2 },
        { "conv1x1 768->768->2048 M=256",         GK_WIDE,    768, 2048,  256, 2 },
    };
    const int NS = (int)(sizeof shapes / sizeof shapes[0]);
    for (int si = 0; si < NS; si++) {
        // Keep the default bounded: fp16 and per-tensor W8A8 on every shape;
        // int8 weights and per-channel W8A8 on the first shape only.
        int modes[4] = {0, 2, 1, 3};
        int nm = (si == 0) ? 4 : 2;
        for (int mi = 0; mi < nm; mi++) {
            int mode = modes[mi];
            char nmbuf[128];
            snprintf(nmbuf, sizeof nmbuf, "%s [%s]", shapes[si].name, graphModeName(mode));
            failures += runGraphCase(nmbuf, shapes[si].kind, shapes[si].Cin,
                                     shapes[si].Cout, shapes[si].M, shapes[si].depth,
                                     mode, &list, iters);
        }
    }

    // ------------------------------------------------------------ summary
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
        anePrintJSON("int8_quant_probe", g_bench ? "bench" : "default",
                     &list, rneTotal, zpAltTotal);
    } else if (!g_quiet) {
        printf("\nsummary: PASS=%d FAIL=%d SKIP=%d UNSUPPORTED=%d  RNE-diff-total=%ld"
               "  zp-add-after-diff-total=%ld\n",
               pass, fail, skip, unsup, rneTotal, zpAltTotal);
        printf("note: RNE differences are expected on this rounding rule and are not failures.\n");
        printf("note: zero_point golden is clamp(round_away(x/scale + zp)); the add-after\n"
               "      count is the number of elements that discriminate the two formulas.\n");
        printf("result: %s\n", fail ? "FAIL" : "PASS");
    }
    return (fail || failures) ? 1 : 0;
}
