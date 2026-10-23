// op_probe.m — compile+load+eval a MIL text file on the ANE in-memory path.
//
// Usage: ./op_probe <case.mil> <weights.bin|-> <inBytes> <outBytes> [<label>]
//
// stdout (exactly one line, machine readable):
//   CASE <label> STATUS <status> COMPILE=<OK|FAIL|NA> LOAD=<...> EVAL=<...> \
//        NUMERIC=<PASS|CANDIDATE|FAIL|UNVERIFIED|ACCEPTANCE_ONLY|NA> \
//        LAYOUT=<name|candidate|unknown|NA> BEST=<name|none|NA> \
//        IN_ROW_BYTES=<n|NA> OUT_ROW_BYTES=<n|NA> \
//        IN_STRIDE=<bytes,...|NA> OUT_STRIDE=<bytes,...|NA> \
//        EXIT=<n> MAXDIFF=<value|NA>
//
// STATUS is OK, COMPILE_FAIL, LOAD_FAIL, EVAL_FAIL, NUMERIC_FAIL,
// NUMERIC_LAYOUT_CANDIDATE, NUMERIC_UNVERIFIED, DESC_NIL, NOMIL, NOWEIGHTS,
// IOSURFACE_FAIL or RUN_ERROR.
//
// stderr carries the raw NSError text (multi-line, unmodified) for the
// separate survey_errors.log.  Nothing else is printed to stdout.
//
// Deterministic inputs and a NaN-sentinel output are used.  For
//   pad (constant/reflect/replicate), depth_to_space, pixel_shuffle, softmax
// the readback is compared against a CPU golden.  Everything else reports
// NUMERIC=ACCEPTANCE_ONLY: compile/load/eval only, never a claimed PASS.
// For acceptance-only cases LAYOUT/BEST/MAXDIFF carry no numeric meaning;
// LAYOUT is the layout that was actually evaluated (the first candidate that
// the private path accepted), not a verified or hardcoded choice.
//
// IOSURFACE LAYOUT
// The private ANE path binds a flat IOSurface; the hardware may use padded
// rows (observed: W=32 fp16 rows pass compact, W=34/W=16 fail).  This probe
// therefore packs logical tensor values into candidate physical layouts:
//   compact  : row stride = W*2 bytes
//   align64  : each W row rounded up to 64 bytes
//   align128 : rounded up to 128 bytes
//   align256 : rounded up to 256 bytes
// Override with OP_PROBE_LAYOUTS=compact,align64,... (or OP_PROBE_LAYOUT=name).
// Default is compact when both last dimensions are already 64-byte aligned;
// otherwise compact,align64,align128,align256 are tried.  An automatic
// multi-layout match is reported as NUMERIC=CANDIDATE, not PASS: candidate fit
// is not a semantics proof.  Rerun the case with one explicit layout
// (OP_PROBE_LAYOUT=align64 or OP_PROBE_IN/OUT_ROW_BYTES) to get PASS and the
// raw IN_ROW_BYTES/OUT_ROW_BYTES and IN_STRIDE/OUT_STRIDE.  If no candidate
// matches, the result is NUMERIC_UNVERIFIED because the physical layout is
// still unknown.  A mismatch is NUMERIC_FAIL only when the layout is
// confirmed (naturally 64-byte-aligned rows or an explicit single layout).
//
// OP_PROBE_DUMP_DIR=DIR writes <label>.<layout>.input.bin/.output.bin and a
// <label>.layouts.txt candidate summary for numeric diagnosis.
// OP_PROBE_DUMP_ATTRS=1 prints modelAttributes/program descriptions for golden
// cases to stderr.  OP_PROBE_IN_ROW_BYTES / OP_PROBE_OUT_ROW_BYTES test an
// explicit physical row stride (bytes) as a "custom" layout.  A single
// explicit layout (OP_PROBE_LAYOUT(S) or custom strides) is treated as
// confirmed: no match is NUMERIC_FAIL.  OP_PROBE_REQUIRE_LAYOUT=1 also makes
// a no-match NUMERIC_FAIL for the automatic multi-layout search.
//
// Exit status: 0 = compile/load/eval OK and numeric PASS, CANDIDATE,
// UNVERIFIED or acceptance-only; 2 usage, 3 missing MIL/weights, 4 descriptor nil,
// 10 compile, 11 load, 12 eval, 13 confirmed numeric mismatch,
// 14 IOSurface failure, 15 RUN_ERROR (internal read/allocation failure).
//
// Build: xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//          -o op_probe op_probe.m
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char g_label[256] = "case";

static void sanitize_copy(const char *in, char *out, size_t cap) {
    size_t j = 0;
    for (size_t i = 0; in && in[i] && j + 1 < cap; i++) {
        unsigned char c = (unsigned char)in[i];
        out[j++] = (c < 32 || c == ' ' || c == '/') ? '_' : (char)c;
    }
    out[j] = 0;
}

static IOSurfaceRef mk(size_t b) {
    if (b < 4096) b = 4096;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(b), (id)kIOSurfaceHeight:@1, (id)kIOSurfaceBytesPerElement:@1,
        (id)kIOSurfaceBytesPerRow:@(b), (id)kIOSurfaceAllocSize:@(b), (id)kIOSurfacePixelFormat:@0});
}

// Deterministic fp16 input grid, exactly representable: (k-50)/16, k in [0,100].
static _Float16 pattern_value(size_t i) {
    unsigned k = (unsigned)((i * 37u + 11u) % 101u);
    return (_Float16)(((int)k - 50) / 16.0f);
}

static const char *tok(BOOL ok) { return ok ? "OK" : "FAIL"; }

static int report(const char *status, const char *c, const char *l, const char *e,
                  const char *numeric, const char *layout, const char *best,
                  const char *in_row, const char *out_row,
                  const char *in_str, const char *out_str,
                  double maxdiff, int code) {
    if (maxdiff >= 0.0)
        printf("CASE %s STATUS %s COMPILE=%s LOAD=%s EVAL=%s NUMERIC=%s LAYOUT=%s BEST=%s "
               "IN_ROW_BYTES=%s OUT_ROW_BYTES=%s IN_STRIDE=%s OUT_STRIDE=%s EXIT=%d MAXDIFF=%.6g\n",
               g_label, status, c, l, e, numeric, layout, best,
               in_row, out_row, in_str, out_str, code, maxdiff);
    else
        printf("CASE %s STATUS %s COMPILE=%s LOAD=%s EVAL=%s NUMERIC=%s LAYOUT=%s BEST=%s "
               "IN_ROW_BYTES=%s OUT_ROW_BYTES=%s IN_STRIDE=%s OUT_STRIDE=%s EXIT=%d MAXDIFF=NA\n",
               g_label, status, c, l, e, numeric, layout, best,
               in_row, out_row, in_str, out_str, code);
    fflush(stdout);
    return code;
}

static void log_error(const char *phase, NSError *e) {
    if (!e) return;
    fprintf(stderr, "%s %s: %s\n", g_label, phase, [[e description] UTF8String]);
    fprintf(stderr, "%s %s userInfo: %s\n", g_label, phase,
            [[[e userInfo] description] UTF8String]);
}

// ---------------------------------------------------------------- MIL parsing
static char *read_ctext(const char *path) {
    NSData *d = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];
    if (!d) return NULL;
    char *s = (char *)malloc(d.length + 1);
    if (!s) return NULL;
    memcpy(s, d.bytes, d.length);
    s[d.length] = 0;
    return s;
}

static int parse_dims(const char *p, long *d, int maxd) {
    int n = 0;
    while (p && *p && *p != ']' && n < maxd) {
        while (*p == ' ' || *p == ',' || *p == '\t') p++;
        if (*p == ']') break;
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) return 0;
        d[n++] = v;
        p = end;
    }
    return n;
}

static int parse_case(const char *txt, long *ish, int *irk, long *osh, int *ork,
                      char *op, size_t opcap) {
    const char *p = strstr(txt, "func main<ios18>(tensor<fp16, [");
    if (!p) return 0;
    p += strlen("func main<ios18>(tensor<fp16, [");
    *irk = parse_dims(p, ish, 8);
    if (*irk <= 0) return 0;

    const char *q = strstr(p, "]> out = ");
    if (!q) return 0;
    const char *decl = NULL, *r = txt;
    while ((r = strstr(r, "tensor<fp16, [")) != NULL && r < q) {
        decl = r;
        r++;
    }
    if (!decl) return 0;
    *ork = parse_dims(decl + strlen("tensor<fp16, ["), osh, 8);
    if (*ork <= 0) return 0;

    const char *e = strstr(q, " out = ");
    if (!e) return 0;
    e += strlen(" out = ");
    size_t k = 0;
    while (e[k] && (isalnum((unsigned char)e[k]) || e[k] == '_') && k + 1 < opcap) {
        op[k] = e[k];
        k++;
    }
    op[k] = 0;
    return op[0] != 0;
}

static int find_i32_const(const char *txt, const char *name, long *v) {
    char pat[128];
    snprintf(pat, sizeof pat, "int32 %s = const()", name);
    const char *p = strstr(txt, pat);
    if (!p) return 0;
    p = strstr(p, "val = int32(");
    if (!p) return 0;
    p += strlen("val = int32(");
    char *end = NULL;
    *v = strtol(p, &end, 10);
    return end != p;
}

static int find_string_const(const char *txt, const char *name, char *out, size_t cap) {
    char pat[128];
    snprintf(pat, sizeof pat, "string %s = const()", name);
    const char *p = strstr(txt, pat);
    if (!p) return 0;
    p = strstr(p, "val = string(\"");
    if (!p) return 0;
    p += strlen("val = string(\"");
    size_t k = 0;
    while (p[k] && p[k] != '"' && k + 1 < cap) {
        out[k] = p[k];
        k++;
    }
    out[k] = 0;
    return k > 0;
}

static int find_fp16_const(const char *txt, const char *name, float *v) {
    char pat[128];
    snprintf(pat, sizeof pat, "fp16 %s = const()", name);
    const char *p = strstr(txt, pat);
    if (!p) return 0;
    p = strstr(p, "val = fp16(");
    if (!p) return 0;
    p += strlen("val = fp16(");
    char *end = NULL;
    float x = strtof(p, &end);
    if (end == p) return 0;
    *v = x;
    return 1;
}

static int find_i32_tensor8(const char *txt, const char *name, int *v) {
    char pat[160];
    snprintf(pat, sizeof pat, "tensor<int32, [8]> %s = const()", name);
    const char *p = strstr(txt, pat);
    if (!p) return 0;
    p = strstr(p, "val = tensor<int32, [8]>([");
    if (!p) return 0;
    p += strlen("val = tensor<int32, [8]>([");
    for (int i = 0; i < 8; i++) {
        while (*p == ' ' || *p == ',') p++;
        char *end = NULL;
        long x = strtol(p, &end, 10);
        if (end == p) return 0;
        v[i] = (int)x;
        p = end;
    }
    return 1;
}

// ---------------------------------------------------------------- case info
typedef enum { G_NONE = 0, G_PAD, G_D2S, G_PIXEL, G_SOFTMAX } GoldenKind;

typedef struct {
    long ish[8], osh[8];
    int irk, ork;
    char op[64];
    GoldenKind kind;
    int pd[8];
    int pad_mode;   // 0 constant, 1 reflect, 2 replicate
    float cv;
    long factor;
    long axis;
} CaseInfo;

static void parse_case_info(const char *txt, CaseInfo *ci) {
    memset(ci, 0, sizeof *ci);
    if (!txt || !parse_case(txt, ci->ish, &ci->irk, ci->osh, &ci->ork,
                            ci->op, sizeof ci->op))
        return;
    ci->kind = G_NONE;
    if (strcmp(ci->op, "pad") == 0 && ci->irk == 4 && ci->ork == 4) {
        char mode[32];
        float cv = 0.f;
        if (find_i32_tensor8(txt, "pd", ci->pd) &&
            find_string_const(txt, "md", mode, sizeof mode) &&
            find_fp16_const(txt, "cv", &cv)) {
            int m = (strcmp(mode, "constant") == 0) ? 0
                  : (strcmp(mode, "reflect") == 0) ? 1
                  : (strcmp(mode, "replicate") == 0) ? 2 : -1;
            if (m >= 0) {
                ci->kind = G_PAD;
                ci->pad_mode = m;
                ci->cv = cv;
            }
        }
    } else if ((strcmp(ci->op, "depth_to_space") == 0 ||
                strcmp(ci->op, "pixel_shuffle") == 0) &&
               ci->irk == 4 && ci->ork == 4) {
        long f = 0;
        if ((find_i32_const(txt, "bs", &f) || find_i32_const(txt, "uf", &f)) && f > 0) {
            ci->kind = (strcmp(ci->op, "depth_to_space") == 0) ? G_D2S : G_PIXEL;
            ci->factor = f;
        }
    } else if (strcmp(ci->op, "softmax") == 0) {
        long ax = -1;
        if (find_i32_const(txt, "ax", &ax)) {
            ci->kind = G_SOFTMAX;
            ci->axis = ax;
        }
    }
}

// ---------------------------------------------------------------- goldens
static double diff16(_Float16 got, double expect) {
    double d = fabs((double)got - expect);
    return (got != got) ? INFINITY : d;  // NaN sentinel / output
}

static long numel(const long *s, int rank) {
    long n = 1;
    for (int i = 0; i < rank; i++) n *= s[i];
    return n;
}

// MIL pad: pad[] is (before, after) per dimension in N,C,H,W,... order.
static double golden_pad(const _Float16 *in, const _Float16 *out, const long *ish, int rank,
                         const long *osh, const int *pad, int mode, float cv) {
    long total = numel(osh, rank);
    double maxd = 0.0;
    for (long oi = 0; oi < total; oi++) {
        long coord[8], rem = oi;
        for (int d = rank - 1; d >= 0; d--) {
            coord[d] = rem % osh[d];
            rem /= osh[d];
        }
        long in_off = 0, stride = 1;
        int outside = 0;
        for (int d = rank - 1; d >= 0; d--) {
            long n = ish[d];
            long j = coord[d] - pad[2 * d];
            if (mode == 0) {              // constant
                if (j < 0 || j >= n) { outside = 1; break; }
            } else if (mode == 1) {       // reflect (excluding the edge)
                if (j < 0) j = -j;
                if (j >= n) j = 2 * n - 2 - j;
                if (j < 0 || j >= n) { outside = 1; break; }
            } else {                      // replicate
                if (j < 0) j = 0;
                if (j >= n) j = n - 1;
            }
            in_off += j * stride;
            stride *= n;
        }
        double expect = outside ? (double)(_Float16)cv : (double)in[in_off];
        double d = diff16(out[oi], expect);
        if (d > maxd) maxd = d;
    }
    return maxd;
}

// depth_to_space (DCR, TF order) and pixel_shuffle (CRD, PyTorch order) for
// rank-4 NCHW, block f:
//   depth_to_space: in_c = ((h%f)*f + (w%f))*C_out + c
//   pixel_shuffle:  in_c = c*f*f + (h%f)*f + (w%f)
static double golden_block(const _Float16 *in, const _Float16 *out, const long *ish,
                           const long *osh, long f, int dcr) {
    long N = osh[0], C = osh[1], H = osh[2], W = osh[3];
    long iH = ish[2], iW = ish[3];
    double maxd = 0.0;
    for (long n = 0; n < N; n++)
        for (long c = 0; c < C; c++)
            for (long h = 0; h < H; h++)
                for (long w = 0; w < W; w++) {
                    long ic = dcr ? (((h % f) * f + (w % f)) * C + c)
                                  : (c * f * f + (h % f) * f + (w % f));
                    long off = ((n * ish[1] + ic) * iH + h / f) * iW + w / f;
                    long oo = ((n * C + c) * H + h) * W + w;
                    double d = diff16(out[oo], (double)in[off]);
                    if (d > maxd) maxd = d;
                }
    return maxd;
}

// Softmax along `axis` (normalized), comparing against a double-precision CPU
// reference with the same fp16 input values.
static double golden_softmax(const _Float16 *in, const _Float16 *out, const long *sh,
                             int rank, long axis) {
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) return INFINITY;
    long outer = 1, axlen = sh[axis], inner = 1;
    for (int i = 0; i < axis; i++) outer *= sh[i];
    for (int i = (int)axis + 1; i < rank; i++) inner *= sh[i];
    double *e = (double *)malloc(sizeof(double) * (size_t)axlen);
    if (!e) return NAN;  // allocation failure, not an arithmetic result
    double maxd = 0.0;
    for (long o = 0; o < outer; o++) {
        for (long k = 0; k < inner; k++) {
            double mx = -INFINITY;
            for (long a = 0; a < axlen; a++) {
                double v = (double)in[(o * axlen + a) * inner + k];
                if (v > mx) mx = v;
            }
            double sum = 0.0;
            for (long a = 0; a < axlen; a++) {
                double v = (double)in[(o * axlen + a) * inner + k];
                e[a] = exp(v - mx);
                sum += e[a];
            }
            for (long a = 0; a < axlen; a++) {
                long idx = (o * axlen + a) * inner + k;
                double d = diff16(out[idx], e[a] / sum);
                if (d > maxd) maxd = d;
            }
        }
    }
    free(e);
    return maxd;
}

static double golden_diff(const CaseInfo *ci, const _Float16 *in, const _Float16 *out) {
    switch (ci->kind) {
        case G_PAD:     return golden_pad(in, out, ci->ish, ci->irk, ci->osh,
                                          ci->pd, ci->pad_mode, ci->cv);
        case G_D2S:     return golden_block(in, out, ci->ish, ci->osh, ci->factor, 1);
        case G_PIXEL:   return golden_block(in, out, ci->ish, ci->osh, ci->factor, 0);
        case G_SOFTMAX: return golden_softmax(in, out, ci->ish, ci->irk, ci->axis);
        default:        return -1.0;
    }
}

static double golden_tol(const CaseInfo *ci) {
    return ci->kind == G_SOFTMAX ? 5e-3 : 1e-6;
}

// ---------------------------------------------------------------- layout
typedef struct { const char *name; long align; } LayoutDef;
static const LayoutDef LAYOUTS[] = {
    {"compact", 0}, {"align64", 64}, {"align128", 128}, {"align256", 256},
};
#define NLAYOUTS ((int)(sizeof LAYOUTS / sizeof LAYOUTS[0]))

typedef struct {
    const char *name;
    long align;
    long in_row, out_row;       // row stride in bytes
    long in_stride[8], out_stride[8];
    long in_total, out_total;   // packed bytes
} Layout;

typedef struct {
    const char *name;
    long in_row, out_row, in_total, out_total;
    int evaluated;
    double maxdiff;
} CandResult;

static long align_up(long x, long a) {
    return (a <= 1) ? x : ((x + a - 1) / a) * a;
}

// Strides in bytes; last dim is contiguous fp16, second-to-last row is padded.
static void make_strides(const long *shape, int rank, long row_bytes, long *st) {
    if (rank <= 0) return;
    st[rank - 1] = 2;
    if (rank >= 2) st[rank - 2] = row_bytes;
    for (int d = rank - 3; d >= 0; d--)
        st[d] = st[d + 1] * shape[d + 1];
}

static void fmt_strides(const long *st, int rank, char *buf, size_t cap) {
    size_t off = 0;
    if (cap == 0) return;
    buf[0] = 0;
    for (int i = 0; i < rank; i++) {
        int n = snprintf(buf + off, cap - off, "%s%ld", i ? "," : "", st[i]);
        if (n < 0) break;
        off += (size_t)n;
        if (off >= cap) { buf[cap - 1] = 0; break; }
    }
}

static int make_layout(const CaseInfo *ci, int have_shape,
                       const LayoutDef *def, Layout *L) {
    memset(L, 0, sizeof *L);
    L->name = def->name;
    L->align = def->align;
    if (!have_shape) return def->align == 0;
    if (ci->irk < 1 || ci->ork < 1) return 0;
    if ((ci->irk < 2 || ci->ork < 2) && def->align != 0) return 0;

    L->in_row = def->align
        ? align_up(ci->ish[ci->irk - 1] * 2, def->align)
        : ci->ish[ci->irk - 1] * 2;
    L->out_row = def->align
        ? align_up(ci->osh[ci->ork - 1] * 2, def->align)
        : ci->osh[ci->ork - 1] * 2;
    long in_rows = numel(ci->ish, ci->irk) / ci->ish[ci->irk - 1];
    long out_rows = numel(ci->osh, ci->ork) / ci->osh[ci->ork - 1];
    L->in_total = in_rows * L->in_row;
    L->out_total = out_rows * L->out_row;
    make_strides(ci->ish, ci->irk, L->in_row, L->in_stride);
    make_strides(ci->osh, ci->ork, L->out_row, L->out_stride);
    return 1;
}

static int aligned64(const CaseInfo *ci, int have_shape) {
    if (!have_shape || ci->irk < 1 || ci->ork < 1) return 0;
    return (ci->ish[ci->irk - 1] * 2) % 64 == 0 &&
           (ci->osh[ci->ork - 1] * 2) % 64 == 0;
}

static LayoutDef *find_layout_def(const char *name) {
    for (int i = 0; i < NLAYOUTS; i++)
        if (strcmp(LAYOUTS[i].name, name) == 0) return (LayoutDef *)&LAYOUTS[i];
    return NULL;
}

static void add_layout_obj(const Layout *L, Layout *out, int *n, int max) {
    if (*n >= max) return;
    for (int i = 0; i < *n; i++)
        if (out[i].in_row == L->in_row && out[i].out_row == L->out_row) return;
    out[(*n)++] = *L;
}

static void add_layout(const CaseInfo *ci, int have_shape, const LayoutDef *def,
                       Layout *out, int *n, int max) {
    if (*n >= max) return;
    Layout L;
    if (!make_layout(ci, have_shape, def, &L)) return;
    add_layout_obj(&L, out, n, max);
}

// Explicit row strides from OP_PROBE_IN_ROW_BYTES / OP_PROBE_OUT_ROW_BYTES.
static int make_layout_custom(const CaseInfo *ci, int have_shape, Layout *L) {
    if (!have_shape || ci->irk < 1 || ci->ork < 1) return 0;
    const char *irs = getenv("OP_PROBE_IN_ROW_BYTES");
    const char *ors = getenv("OP_PROBE_OUT_ROW_BYTES");
    long il = ci->ish[ci->irk - 1] * 2, ol = ci->osh[ci->ork - 1] * 2;
    long ir = (irs && *irs) ? strtol(irs, NULL, 10) : il;
    long ore = (ors && *ors) ? strtol(ors, NULL, 10) : ol;
    if (ir < il || ore < ol || (ir & 1) || (ore & 1)) return 0;
    memset(L, 0, sizeof *L);
    L->name = "custom";
    L->align = -1;
    L->in_row = ir;
    L->out_row = ore;
    long in_rows = numel(ci->ish, ci->irk) / ci->ish[ci->irk - 1];
    long out_rows = numel(ci->osh, ci->ork) / ci->osh[ci->ork - 1];
    L->in_total = in_rows * L->in_row;
    L->out_total = out_rows * L->out_row;
    make_strides(ci->ish, ci->irk, L->in_row, L->in_stride);
    make_strides(ci->osh, ci->ork, L->out_row, L->out_stride);
    return 1;
}

static int collect_layouts(const CaseInfo *ci, int have_shape, Layout *out, int max) {
    int n = 0;
    const char *irs = getenv("OP_PROBE_IN_ROW_BYTES");
    const char *ors = getenv("OP_PROBE_OUT_ROW_BYTES");
    int use_custom = (irs && *irs) || (ors && *ors);
    const char *env = getenv("OP_PROBE_LAYOUTS");
    if (!env || !*env) env = getenv("OP_PROBE_LAYOUT");

    if (use_custom) {
        Layout L;
        if (make_layout_custom(ci, have_shape, &L)) add_layout_obj(&L, out, &n, max);
    }
    if (env && *env) {
        char buf[256];
        snprintf(buf, sizeof buf, "%s", env);
        char *save = NULL;
        for (char *t = strtok_r(buf, ", ", &save); t; t = strtok_r(NULL, ", ", &save)) {
            LayoutDef *d = find_layout_def(t);
            if (d) add_layout(ci, have_shape, d, out, &n, max);
        }
    } else if (!use_custom) {
        add_layout(ci, have_shape, find_layout_def("compact"), out, &n, max);
        if (have_shape && !aligned64(ci, have_shape)) {
            add_layout(ci, have_shape, find_layout_def("align64"), out, &n, max);
            add_layout(ci, have_shape, find_layout_def("align128"), out, &n, max);
            add_layout(ci, have_shape, find_layout_def("align256"), out, &n, max);
        }
    }
    if (n == 0) add_layout(ci, have_shape, find_layout_def("compact"), out, &n, max);
    return n;
}

static void pack_flat(_Float16 *dst, size_t bytes) {
    memset(dst, 0, bytes);
    size_t n = bytes / sizeof(_Float16);
    for (size_t i = 0; i < n; i++) dst[i] = pattern_value(i);
}

// Pack logical contiguous values into a (row-padded) physical surface.
static void pack_logical(_Float16 *dst, size_t bytes, const long *shape, int rank,
                         const long *strides, const _Float16 *logical) {
    memset(dst, 0, bytes);
    long n = numel(shape, rank);
    for (long i = 0; i < n; i++) {
        long rem = i, off = 0;
        for (int d = rank - 1; d >= 0; d--) {
            long c = rem % shape[d];
            rem /= shape[d];
            off += c * strides[d];
        }
        dst[off / 2] = logical[i];
    }
}

static void unpack_logical(_Float16 *logical, const _Float16 *src,
                           const long *shape, int rank, const long *strides) {
    long n = numel(shape, rank);
    for (long i = 0; i < n; i++) {
        long rem = i, off = 0;
        for (int d = rank - 1; d >= 0; d--) {
            long c = rem % shape[d];
            rem /= shape[d];
            off += c * strides[d];
        }
        logical[i] = src[off / 2];
    }
}

static void dump_candidate(const char *label, const char *layout,
                           IOSurfaceRef ioI, IOSurfaceRef ioO,
                           size_t inB, size_t outB) {
    const char *dir = getenv("OP_PROBE_DUMP_DIR");
    if (!dir || !*dir) return;
    NSString *d = [NSString stringWithUTF8String:dir];
    [[NSFileManager defaultManager] createDirectoryAtPath:d
        withIntermediateDirectories:YES attributes:nil error:nil];
    IOSurfaceLock(ioI, kIOSurfaceLockReadOnly, NULL);
    NSString *ip = [d stringByAppendingPathComponent:
        [NSString stringWithFormat:@"%s.%s.input.bin", label, layout]];
    [[NSData dataWithBytes:IOSurfaceGetBaseAddress(ioI) length:inB] writeToFile:ip atomically:YES];
    IOSurfaceUnlock(ioI, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(ioO, kIOSurfaceLockReadOnly, NULL);
    NSString *op = [d stringByAppendingPathComponent:
        [NSString stringWithFormat:@"%s.%s.output.bin", label, layout]];
    [[NSData dataWithBytes:IOSurfaceGetBaseAddress(ioO) length:outB] writeToFile:op atomically:YES];
    IOSurfaceUnlock(ioO, kIOSurfaceLockReadOnly, NULL);
}

static void dump_summary(const char *label, const CaseInfo *ci, int have_shape,
                         const CandResult *r, const Layout *layouts, int n,
                         const char *status, const char *numeric,
                         const char *chosen, const char *best,
                         IOSurfaceRef ioI, IOSurfaceRef ioO) {
    const char *dir = getenv("OP_PROBE_DUMP_DIR");
    if (!dir || !*dir) return;
    NSString *d = [NSString stringWithUTF8String:dir];
    [[NSFileManager defaultManager] createDirectoryAtPath:d
        withIntermediateDirectories:YES attributes:nil error:nil];
    NSString *p = [d stringByAppendingPathComponent:
        [NSString stringWithFormat:@"%s.layouts.txt", label]];
    FILE *f = fopen([p fileSystemRepresentation], "w");
    if (!f) return;
    fprintf(f, "status=%s numeric=%s chosen=%s best=%s op=%s\n",
            status, numeric, chosen, best, ci->op);
    if (ioI && ioO)
        fprintf(f, "iosurface in_bpr=%zu in_alloc=%zu out_bpr=%zu out_alloc=%zu\n",
                IOSurfaceGetBytesPerRow(ioI), IOSurfaceGetAllocSize(ioI),
                IOSurfaceGetBytesPerRow(ioO), IOSurfaceGetAllocSize(ioO));
    if (have_shape) {
        fprintf(f, "ishape=");
        for (int i = 0; i < ci->irk; i++) fprintf(f, "%s%ld", i ? "x" : "", ci->ish[i]);
        fprintf(f, " oshape=");
        for (int i = 0; i < ci->ork; i++) fprintf(f, "%s%ld", i ? "x" : "", ci->osh[i]);
        fprintf(f, "\n");
    }
    for (int i = 0; i < n; i++) {
        char is[128] = "NA", os[128] = "NA";
        if (have_shape) {
            fmt_strides(layouts[i].in_stride, ci->irk, is, sizeof is);
            fmt_strides(layouts[i].out_stride, ci->ork, os, sizeof os);
        }
        fprintf(f, "layout=%s in_row=%ld out_row=%ld in_total=%ld out_total=%ld "
                   "in_stride=%s out_stride=%s evaluated=%d maxdiff=%.9g\n",
                r[i].name, r[i].in_row, r[i].out_row, r[i].in_total, r[i].out_total,
                is, os, r[i].evaluated, r[i].maxdiff);
    }
    fclose(f);
}

// ---------------------------------------------------------------- main
int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: op_probe <case.mil> <weights.bin|-> <inBytes> <outBytes> [label]\n");
        return 2;
    }
    @autoreleasepool {
        const char *milPath = argv[1];
        const char *wPath = argv[2];
        size_t inB = (size_t)strtoull(argv[3], NULL, 10);
        size_t outB = (size_t)strtoull(argv[4], NULL, 10);
        sanitize_copy((argc > 5) ? argv[5] : milPath, g_label, sizeof g_label);

        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        NSError *e = nil;
        NSData *milData = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:milPath]];
        if (!milData) return report("NOMIL", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", -1, 3);
        NSData *wb = nil;
        if (strcmp(wPath, "-") != 0) {
            wb = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:wPath]];
            if (!wb) return report("NOWEIGHTS", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", -1, 3);
        }

        Class D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class I = NSClassFromString(@"_ANEInMemoryModel");
        Class AR = NSClassFromString(@"_ANERequest");
        Class AIO = NSClassFromString(@"_ANEIOSurfaceObject");
        id weightsArg = wb ? @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wb}} : @{};

        id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
            D, @selector(modelWithMILText:weights:optionsPlist:), milData, weightsArg, nil);
        if (!desc) return report("DESC_NIL", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", -1, 4);

        id mdl = ((id(*)(Class, SEL, id))objc_msgSend)(
            I, @selector(inMemoryModelWithDescriptor:), desc);
        // Temp setup must not be mistaken for a compile rejection: any
        // failure here is RUN_ERROR/15, before the compile stage.
        BOOL setup_ok = YES;
        NSError *setup_err = nil;
        id hx = nil;
        NSString *td = nil;
        NSFileManager *fm = [NSFileManager defaultManager];
        if (!mdl) {
            fprintf(stderr, "%s RUN_ERROR in-memory model is nil\n", g_label);
            setup_ok = NO;
        } else {
            hx = ((id(*)(id, SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        }
        if (!hx) {
            fprintf(stderr, "%s RUN_ERROR no hexStringIdentifier\n", g_label);
            setup_ok = NO;
        }
        if (setup_ok) {
            td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
            NSString *wdir = [td stringByAppendingPathComponent:@"weights"];
            if (![fm createDirectoryAtPath:wdir withIntermediateDirectories:YES
                                  attributes:nil error:&setup_err]) {
                fprintf(stderr, "%s SETUP_FAIL create %s: %s\n", g_label,
                        [wdir UTF8String], setup_err ? [[setup_err description] UTF8String] : "?");
                setup_ok = NO;
            }
        }
        if (setup_ok) {
            NSString *mp = [td stringByAppendingPathComponent:@"model.mil"];
            setup_err = nil;
            if (![milData writeToFile:mp options:NSDataWritingAtomic error:&setup_err]) {
                fprintf(stderr, "%s SETUP_FAIL write %s: %s\n", g_label,
                        [mp UTF8String], setup_err ? [[setup_err description] UTF8String] : "?");
                setup_ok = NO;
            }
        }
        if (setup_ok && wb) {
            NSString *wp = [td stringByAppendingPathComponent:@"weights/weight.bin"];
            setup_err = nil;
            if (![wb writeToFile:wp options:NSDataWritingAtomic error:&setup_err]) {
                fprintf(stderr, "%s SETUP_FAIL write %s: %s\n", g_label,
                        [wp UTF8String], setup_err ? [[setup_err description] UTF8String] : "?");
                setup_ok = NO;
            }
        }

        BOOL cok = NO, lok = NO, eok = NO;
        int stage = 0;  // 1=compile attempted, 2=load, 3=eval
        id wI = nil, wO = nil, req = nil;
        int rc = 0;
        int explicit_layout = 0, require_layout = 0;
        const char *status = "OK", *numeric = "NA";
        const char *chosen = "NA", *best = "NA";
        char in_row_s[32] = "NA", out_row_s[32] = "NA";
        char in_str_s[128] = "NA", out_str_s[128] = "NA";
        double maxdiff = -1.0;
        IOSurfaceRef ioI = NULL, ioO = NULL;
        char *txt = NULL;
        _Float16 *in_logical = NULL, *out_logical = NULL;
        CaseInfo ci;
        memset(&ci, 0, sizeof ci);
        int have_shape = 0;

        if (!setup_ok) {
            status = "RUN_ERROR"; rc = 15;
            goto done;
        }

        e = nil;
        stage = 1;
        cok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        if (!cok) {
            log_error("COMPILE_FAIL", e);
            status = "COMPILE_FAIL"; rc = 10;
            goto done;
        }

        e = nil;
        stage = 2;
        lok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!lok) {
            log_error("LOAD_FAIL", e);
            status = "LOAD_FAIL"; rc = 11;
            goto done;
        }

        txt = read_ctext(milPath);
        if (!txt) {
            fprintf(stderr, "%s RUN_ERROR cannot read MIL text\n", g_label);
            status = "RUN_ERROR"; rc = 15;
            goto done;
        }
        parse_case_info(txt, &ci);
        if (ci.kind != G_NONE && getenv("OP_PROBE_DUMP_ATTRS")) {
            const char *names[] = {"modelAttributes", "program", "model", "localModelPath", NULL};
            for (int i = 0; names[i]; i++) {
                SEL s = NSSelectorFromString([NSString stringWithUTF8String:names[i]]);
                if ([mdl respondsToSelector:s]) {
                    id v = ((id(*)(id, SEL))objc_msgSend)(mdl, s);
                    fprintf(stderr, "%s %s: %s\n", g_label, names[i],
                            v ? [[v description] UTF8String] : "(nil)");
                }
            }
        }
        have_shape = ci.irk > 0 && ci.ork > 0;
        if (have_shape) {
            long n = numel(ci.ish, ci.irk);
            in_logical = ( _Float16 * )malloc(sizeof(_Float16) * (size_t)n);
            if (!in_logical) {
                fprintf(stderr, "%s RUN_ERROR input logical allocation (%ld elems)\n", g_label, n);
                status = "RUN_ERROR"; rc = 15;
                goto done;
            }
            for (long i = 0; i < n; i++) in_logical[i] = pattern_value((size_t)i);
        }
        if (have_shape && ci.kind != G_NONE) {
            long n = numel(ci.osh, ci.ork);
            out_logical = ( _Float16 * )malloc(sizeof(_Float16) * (size_t)n);
            if (!out_logical) {
                fprintf(stderr, "%s RUN_ERROR output logical allocation (%ld elems)\n", g_label, n);
                status = "RUN_ERROR"; rc = 15;
                goto done;
            }
        }

        ioI = mk(inB);
        ioO = mk(outB);
        if (!ioI || !ioO) {
            fprintf(stderr, "%s IOSURFACE_FAIL inB=%zu outB=%zu\n", g_label, inB, outB);
            status = "IOSURFACE_FAIL"; rc = 14;
            goto done;
        }

        Layout layouts[8];
        CandResult cres[8];
        memset(cres, 0, sizeof cres);
        int nlay = collect_layouts(&ci, have_shape, layouts, 8);
        {
            const char *e1 = getenv("OP_PROBE_LAYOUTS");
            const char *e2 = getenv("OP_PROBE_LAYOUT");
            const char *ir = getenv("OP_PROBE_IN_ROW_BYTES");
            const char *ore = getenv("OP_PROBE_OUT_ROW_BYTES");
            explicit_layout = (e1 && *e1) || (e2 && *e2) || (ir && *ir) || (ore && *ore);
        }
        require_layout = getenv("OP_PROBE_REQUIRE_LAYOUT") != NULL;
        const char *passname = NULL;
        const char *bestname = "none";
        const char *triedname = NULL;
        int passli = -1, bestli = -1, triedli = -1;
        double bestdiff = INFINITY;
        int evaluated = 0;

        for (int li = 0; li < nlay; li++) {
            Layout *L = &layouts[li];
            cres[li].name = L->name;
            cres[li].in_row = L->in_row;
            cres[li].out_row = L->out_row;
            cres[li].in_total = L->in_total;
            cres[li].out_total = L->out_total;
            cres[li].evaluated = 0;
            cres[li].maxdiff = -1.0;

            if (have_shape && (L->in_total > (long)inB || L->out_total > (long)outB))
                continue;

            IOSurfaceLock(ioI, 0, NULL);
            {
                _Float16 *p = (_Float16 *)IOSurfaceGetBaseAddress(ioI);
                if (have_shape)
                    pack_logical(p, inB, ci.ish, ci.irk, L->in_stride, in_logical);
                else
                    pack_flat(p, inB);
            }
            IOSurfaceUnlock(ioI, 0, NULL);

            IOSurfaceLock(ioO, 0, NULL);
            {
                uint16_t *p = (uint16_t *)IOSurfaceGetBaseAddress(ioO);
                size_t n = outB / sizeof(uint16_t);
                for (size_t i = 0; i < n; i++) p[i] = 0x7E00;  // NaN sentinel
            }
            IOSurfaceUnlock(ioO, 0, NULL);

            // Build the request after packing so it sees this candidate's
            // input/output surfaces (and so each eval is independent).
            wI = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioI);
            wO = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioO);
            req = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
                AR, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

            e = nil;
            stage = 3;
            eok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
            if (!eok) {
                log_error("EVAL_FAIL", e);
                status = "EVAL_FAIL"; rc = 12;
                goto done;
            }
            evaluated++;
            cres[li].evaluated = 1;
            // Record the layout the private path actually accepted.  For
            // acceptance-only cases this is the only layout evidence we have.
            triedname = L->name;
            triedli = li;

            if (ci.kind != G_NONE) {
                IOSurfaceLock(ioO, kIOSurfaceLockReadOnly, NULL);
                unpack_logical(out_logical, (const _Float16 *)IOSurfaceGetBaseAddress(ioO),
                               ci.osh, ci.ork, L->out_stride);
                IOSurfaceUnlock(ioO, kIOSurfaceLockReadOnly, NULL);

                double d = golden_diff(&ci, in_logical, out_logical);
                if (isnan(d)) {
                    fprintf(stderr, "%s RUN_ERROR golden buffer allocation\n", g_label);
                    status = "RUN_ERROR"; rc = 15;
                    goto done;
                }
                cres[li].maxdiff = d;
                if (d < bestdiff) {
                    bestdiff = d;
                    bestname = L->name;
                    bestli = li;
                }
                if (getenv("OP_PROBE_DUMP_DIR") &&
                    (d > golden_tol(&ci) || getenv("OP_PROBE_DUMP_ALL")))
                    dump_candidate(g_label, L->name, ioI, ioO, inB, outB);
                if (d <= golden_tol(&ci)) {
                    passname = L->name;
                    passli = li;
                    break;
                }
            } else {
                if (getenv("OP_PROBE_DUMP_DIR") && getenv("OP_PROBE_DUMP_ALL"))
                    dump_candidate(g_label, L->name, ioI, ioO, inB, outB);
                break;
            }
        }

        if (ci.kind == G_NONE) {
            // No CPU golden: this is compile/load/eval acceptance only.  Report
            // the layout that was actually evaluated; do not claim a numeric
            // layout choice or verification.
            numeric = "ACCEPTANCE_ONLY";
            chosen = triedname ? triedname : "unknown";
            best = "n/a";
        } else if (passname) {
            maxdiff = bestdiff;
            best = passname;
            // An automatic multi-layout fit is only a candidate.  It becomes
            // PASS only with an explicit single layout (or when the rows are
            // naturally 64-byte aligned, where compact is the only layout).
            int single = (nlay == 1);
            int confirmed = single && (explicit_layout || aligned64(&ci, have_shape));
            if (confirmed) {
                numeric = "PASS";
                chosen = passname;
            } else {
                numeric = "CANDIDATE";
                chosen = "candidate";
                status = "NUMERIC_LAYOUT_CANDIDATE";
                rc = 0;
            }
        } else {
            // No candidate matched.  A mismatch is only an arithmetic claim
            // when the layout is confirmed: naturally 64-byte-aligned rows, a
            // single explicitly requested layout, or OP_PROBE_REQUIRE_LAYOUT.
            maxdiff = bestdiff;
            chosen = "unknown";
            best = bestname;
            int single = (nlay == 1);
            int confirmed = single && (explicit_layout || aligned64(&ci, have_shape));
            if (require_layout || confirmed) {
                numeric = "FAIL";
                status = "NUMERIC_FAIL";
                rc = 13;
            } else {
                numeric = "UNVERIFIED";
                status = "NUMERIC_UNVERIFIED";
                rc = 0;
            }
        }
        if (evaluated == 0) {
            numeric = "NA";
            status = "EVAL_FAIL";
            rc = 12;
        }

        // Report raw row bytes and element strides for the chosen/best layout.
        Layout *rep = NULL;
        if (passli >= 0) rep = &layouts[passli];
        else if (bestli >= 0) rep = &layouts[bestli];
        else if (ci.kind == G_NONE && triedli >= 0) rep = &layouts[triedli];
        if (rep) {
            snprintf(in_row_s, sizeof in_row_s, "%ld", rep->in_row);
            snprintf(out_row_s, sizeof out_row_s, "%ld", rep->out_row);
            fmt_strides(rep->in_stride, ci.irk, in_str_s, sizeof in_str_s);
            fmt_strides(rep->out_stride, ci.ork, out_str_s, sizeof out_str_s);
        }

        dump_summary(g_label, &ci, have_shape, cres, layouts, nlay,
                     status, numeric, chosen, best, ioI, ioO);

    done:
        if (in_logical) free(in_logical);
        if (out_logical) free(out_logical);
        if (txt) free(txt);
        if (mdl) {
            NSError *ue = nil;
            BOOL uok = ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
                mdl, @selector(unloadWithQoS:error:), 21, &ue);
            if (!uok) {
                if (ue) log_error("UNLOAD_FAIL", ue);
                else fprintf(stderr, "%s UNLOAD_FAIL: no NSError\n", g_label);
                status = "RUN_ERROR"; rc = 15;
            }
        }
        if (ioI) CFRelease(ioI);
        if (ioO) CFRelease(ioO);
        if (td) [fm removeItemAtPath:td error:nil];
        const char *ct = stage >= 1 ? tok(cok) : "NA";
        const char *lt = stage >= 2 ? tok(lok) : "NA";
        const char *et = stage >= 3 ? tok(eok) : "NA";
        return report(status, ct, lt, et, numeric, chosen, best,
                      in_row_s, out_row_s, in_str_s, out_str_s, maxdiff, rc);
    }
}
