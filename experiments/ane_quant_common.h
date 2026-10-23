// ane_quant_common.h — shared, deterministic helpers for the ANE quantization probes.
//
// This file is included textually by int8_quant_probe.m and constexpr_scale_probe.m.
// It deliberately does not touch the private ANE API by itself; the probes call
// aneRunCase() only after loading AppleNeuralEngine.
//
// Design rules:
//   * no arc4random / time-based data: every input and weight comes from a seeded
//     SplitMix64 stream, so two runs on the same machine see identical bytes;
//   * the BLOBFILE container is the standard CoreML v2 form:
//       64-byte file header: uint32 count, uint32 version
//       64-byte chunk header: 0xDEADBEEF, uint32 dtype tag,
//                             uint64 data_size, uint64 absolute data_offset
//     There is no bit-width field at +10 (the old probes wrote one there and
//     corrupted data_size).
//   * a failed case is reported through ANEResult; the caller decides whether the
//     failure is part of the required suite.
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>
#import <math.h>
#import <stdio.h>
#import <string.h>

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) {
    return (double)t * (double)g_tb.numer / (double)g_tb.denom / 1e6;
}

static int g_dumpMIL = 0;

// ---------------------------------------------------------------- CLI + case report
// Stable contract for the serial validation runner:
//   default      quick correctness + acceptance; no timing loops
//   --bench      add timing loops (still bounded)
//   --json       print one machine-readable JSON object instead of per-case lines
//   --quiet      per-case human lines only, no prose preamble
//   --selftest   offline arithmetic / MIL / blob checks; never opens ANE
// Status values are PASS, FAIL, SKIP, UNSUPPORTED.  Only FAIL makes the process
// exit nonzero.  UNSUPPORTED is reserved for a concrete exploratory case whose
// MIL the compiler actually rejected.  Setup failures (private API missing,
// descriptor nil, temp-dir/file writes), load/eval/unload failures and numeric
// mismatches are all FAIL.
typedef enum {
    ST_PASS = 0,
    ST_FAIL = 1,
    ST_SKIP = 2,
    ST_UNSUPPORTED = 3,
} ANEStatus;

static const char *aneStatusName(int s) {
    switch (s) {
        case ST_PASS: return "PASS";
        case ST_FAIL: return "FAIL";
        case ST_SKIP: return "SKIP";
        default: return "UNSUPPORTED";
    }
}

typedef struct {
    char name[128];
    char detail[640];
    int status;
    double ms;
    long elements, mism, rne;
} ANECaseRec;

typedef struct {
    ANECaseRec v[96];
    int n;
} ANECaseList;

static int g_json = 0;
static int g_quiet = 0;
static int g_bench = 0;

static void aneRecord(ANECaseList *l, const char *name, int status,
                      double ms, long elements, long mism, long rne,
                      const char *detail)
{
    if (l->n >= (int)(sizeof l->v / sizeof l->v[0])) return;
    ANECaseRec *c = &l->v[l->n++];
    snprintf(c->name, sizeof c->name, "%s", name);
    c->status = status;
    c->ms = ms;
    c->elements = elements;
    c->mism = mism;
    c->rne = rne;
    c->detail[0] = 0;
    if (detail && detail[0]) {
        size_t j = 0;
        for (size_t i = 0; detail[i] && j + 1 < sizeof c->detail; i++) {
            char ch = detail[i];
            if ((unsigned char)ch < 0x20 || ch == '"' || ch == '\\') ch = ' ';
            c->detail[j++] = ch;
        }
        c->detail[j] = 0;
    }
    if (!g_json) {
        if (elements > 0)
            printf("  %-52s %-11s %6ld/%-6ld mism  RNE=%-5ld",
                   name, aneStatusName(status), mism, elements, rne);
        else
            printf("  %-52s %-11s", name, aneStatusName(status));
        if (g_bench && ms > 0.0) printf(" %8.3f ms", ms);
        // Keep console lines bounded; the JSON detail is allowed to be longer.
        if (c->detail[0]) printf("  %.200s", c->detail);
        printf("\n");
        fflush(stdout);
    }
}

static void anePrintJSON(const char *suite, const char *mode,
                         const ANECaseList *l, long rneTotal, long zpAltTotal)
{
    long pass = 0, fail = 0, skip = 0, unsup = 0;
    for (int i = 0; i < l->n; i++) {
        switch (l->v[i].status) {
            case ST_PASS: pass++; break;
            case ST_FAIL: fail++; break;
            case ST_SKIP: skip++; break;
            default: unsup++; break;
        }
    }
    printf("{\"suite\":\"%s\",\"mode\":\"%s\",\"pass\":%ld,\"fail\":%ld,"
           "\"skip\":%ld,\"unsupported\":%ld,\"rne_mismatches\":%ld,"
           "\"zp_add_after_mismatches\":%ld,\"cases\":[",
           suite, mode, pass, fail, skip, unsup, rneTotal, zpAltTotal);
    for (int i = 0; i < l->n; i++) {
        const ANECaseRec *c = &l->v[i];
        printf("%s{\"name\":\"%s\",\"status\":\"%s\",\"elements\":%ld,"
               "\"mismatches\":%ld,\"rne_mismatches\":%ld,\"ms\":%.4f,"
               "\"detail\":\"%s\"}",
               i ? "," : "", c->name, aneStatusName(c->status),
               c->elements, c->mism, c->rne, c->ms, c->detail);
    }
    printf("]}\n");
}

// Returns 1 if argv contains `flag`.
static int aneHasFlag(int argc, char **argv, const char *flag) {
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], flag) == 0) return 1;
    return 0;
}

// ---------------------------------------------------------------- deterministic PRNG
typedef struct { uint64_t s; } Rng;

static void rngSeed(Rng *r, uint64_t seed) {
    r->s = seed ? seed : 0x9E3779B97F4A7C15ull;
}

static uint64_t rngNext(Rng *r) {
    uint64_t z = (r->s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Inclusive range; lo <= hi.
static int rngInt(Rng *r, int lo, int hi) {
    uint64_t span = (uint64_t)(hi - lo + 1);
    return lo + (int)(rngNext(r) % span);
}

// ---------------------------------------------------------------- BLOBFILE v2 writer
enum {
    ANE_BLOB_FP16  = 1,
    ANE_BLOB_INT8  = 4,
    ANE_BLOB_UINT4 = 0x0b,
};

@interface ANEBlob : NSObject
// Returns the byte offset of the chunk header; the payload starts at offset + 64.
- (NSUInteger)put:(const void *)payload bytes:(NSUInteger)n dtype:(uint32_t)dtype;
@property (nonatomic, readonly) NSData *data;
@end

@implementation ANEBlob {
    NSMutableData *_d;
    uint32_t _chunks;
}

- (instancetype)init {
    if ((self = [super init])) {
        _d = [NSMutableData data];
        _chunks = 0;
        uint8_t h[64] = {0};
        h[0] = 0x00;   // chunk count; incremented by every put:
        h[4] = 0x02;   // container version 2
        [_d appendBytes:h length:sizeof h];
    }
    return self;
}

- (NSUInteger)put:(const void *)payload bytes:(NSUInteger)n dtype:(uint32_t)dtype {
    while (_d.length % 64) {
        uint8_t z = 0;
        [_d appendBytes:&z length:1];
    }
    const NSUInteger off = _d.length;
    uint8_t h[64] = {0};
    h[0] = 0xEF; h[1] = 0xBE; h[2] = 0xAD; h[3] = 0xDE;
    uint32_t dt = dtype;
    memcpy(h + 4, &dt, sizeof dt);
    uint64_t sz = (uint64_t)n;
    memcpy(h + 8, &sz, sizeof sz);
    uint64_t at = (uint64_t)(off + sizeof h);
    memcpy(h + 16, &at, sizeof at);
    [_d appendBytes:h length:sizeof h];
    if (n) [_d appendBytes:payload length:n];
    // CoreML's StorageWriter records the actual number of chunks here; a
    // hard-coded 1 is wrong for every multi-chunk blob (graph chains, LUT +
    // weights + scales, ...).
    _chunks++;
    uint32_t cnt = _chunks;
    memcpy(_d.mutableBytes, &cnt, sizeof cnt);
    return off;
}

- (NSData *)data {
    uint32_t cnt = _chunks;
    if (_d.length >= sizeof cnt) memcpy(_d.mutableBytes, &cnt, sizeof cnt);
    return _d;
}
@end

// ---------------------------------------------------------------- IOSurface
// The private path has been observed to require at least 4096 bytes even when the
// logical tensor is smaller.  Everything here is a flat byte buffer.
static IOSurfaceRef aneSurface(size_t bytes) {
    size_t b = bytes < 4096 ? 4096 : bytes;
    b = (b + 4095) & ~(size_t)4095;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(b), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(b),
        (id)kIOSurfaceAllocSize:@(b), (id)kIOSurfacePixelFormat:@0});
}

// ---------------------------------------------------------------- error text
// The compiler's userInfo description can be dominated by a long temp path.
// Keep both ends: the useful token (InvalidMILProgram, ...) is usually near the
// start, while the numeric detail / trailing hash is at the end.  The old code
// kept only the tail and could cut the token away entirely.
static NSString *aneErrTail(NSError *e, NSUInteger maxLen) {
    NSString *d = e ? [[e userInfo] description] : @"?";
    if (!d || d.length == 0) d = e ? [e description] : @"?";
    if (!d) d = @"?";
    d = [d stringByReplacingOccurrencesOfString:@"\n" withString:@" "];
    d = [d stringByReplacingOccurrencesOfString:@"\r" withString:@" "];
    while ([d containsString:@"  "])
        d = [d stringByReplacingOccurrencesOfString:@"  " withString:@" "];
    if (d.length > maxLen && maxLen > 16) {
        NSUInteger head = maxLen * 2 / 5;
        NSUInteger tail = maxLen - head - 5;
        d = [NSString stringWithFormat:@"%@ ... %@",
             [d substringToIndex:head],
             [d substringFromIndex:d.length - tail]];
    }
    return d;
}

// ---------------------------------------------------------------- runner
typedef struct {
    // compileAttempted is set only when compileWithQoS was actually invoked.
    // compilerRejected is set only when that call returned NO.  Callers use
    // these to reserve UNSUPPORTED for a real MIL rejection of a concrete
    // exploratory case; setup, file-write, load, eval and unload failures are
    // infrastructure failures and must be FAIL.
    int compileAttempted, compilerRejected;
    int compileOK, loadOK, evalOK, unloadOK;
    double ms;          // best of three timing blocks; 0 when timedIters == 0
    char err[1024];
} ANEResult;

// One fp16/int8 input, one output.  `outCopy` is filled from the first successful
// evaluate, before any timing loop.  timedIters == 0 means "correctness only".
static ANEResult aneRunCase(NSString *mil, NSData *wb,
                            const void *in, size_t inBytes,
                            size_t outBytes, int timedIters,
                            void *outCopy, size_t outCopyBytes)
{
    ANEResult r;
    memset(&r, 0, sizeof r);
    r.unloadOK = 1;     // vacuously true unless a loaded model fails to unload

    @autoreleasepool {
        NSError *e = nil;
        if (g_dumpMIL) printf("---- MIL ----\n%s\n-------------\n", [mil UTF8String]);

        Class D   = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class I   = NSClassFromString(@"_ANEInMemoryModel");
        Class AR  = NSClassFromString(@"_ANERequest");
        Class AIO = NSClassFromString(@"_ANEIOSurfaceObject");
        if (!D || !I || !AR || !AIO) {
            snprintf(r.err, sizeof r.err, "SETUP_PRIVATE_API_MISSING");
            return r;
        }

        NSData *milData = [[mil dataUsingEncoding:NSUTF8StringEncoding] copy];
        if (!milData) {
            snprintf(r.err, sizeof r.err, "SETUP_MIL_ENCODE_FAIL");
            return r;
        }
        id weightsArg = wb ? @{@"@model_path/weights/weight.bin":
                               @{@"offset":@0, @"data":wb}} : @{};
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            D, @selector(modelWithMILText:weights:optionsPlist:),
            milData, weightsArg, nil);
        if (!desc) {
            snprintf(r.err, sizeof r.err, "SETUP_DESC_NIL");
            return r;
        }
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            I, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) {
            snprintf(r.err, sizeof r.err, "SETUP_MODEL_NIL");
            return r;
        }

        // `@model_path/` resolves against a directory named by this model's own
        // identifier, not an arbitrary name.
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        if (!hx) {
            snprintf(r.err, sizeof r.err, "SETUP_HEX_ID_NIL");
            return r;
        }
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        if (!td) {
            snprintf(r.err, sizeof r.err, "SETUP_TEMP_DIR_FAIL");
            return r;
        }
        NSFileManager *fm = [NSFileManager defaultManager];
        NSString *weightsDir = [td stringByAppendingPathComponent:@"weights"];
        if (![fm createDirectoryAtPath:weightsDir withIntermediateDirectories:YES
                            attributes:nil error:nil]) {
            snprintf(r.err, sizeof r.err, "SETUP_TEMP_DIR_FAIL %s", [td UTF8String]);
            return r;
        }
        if (![milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"]
                       atomically:YES]) {
            snprintf(r.err, sizeof r.err, "SETUP_WRITE_MIL_FAIL %s", [td UTF8String]);
            [fm removeItemAtPath:td error:nil];
            return r;
        }
        if (wb && ![wb writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"]
                        atomically:YES]) {
            snprintf(r.err, sizeof r.err, "SETUP_WRITE_WEIGHTS_FAIL %s", [td UTF8String]);
            [fm removeItemAtPath:td error:nil];
            return r;
        }

        r.compileAttempted = 1;
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            r.compilerRejected = 1;
            NSString *d = aneErrTail(e, 700);
            snprintf(r.err, sizeof r.err, "COMPILE_REJECT %s", d ? [d UTF8String] : "?");
            [fm removeItemAtPath:td error:nil];
            return r;
        }
        r.compileOK = 1;

        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
            NSString *d = aneErrTail(e, 700);
            snprintf(r.err, sizeof r.err, "LOAD_FAIL %s", d ? [d UTF8String] : "?");
            [fm removeItemAtPath:td error:nil];
            return r;
        }
        r.loadOK = 1;

        IOSurfaceRef ioI = aneSurface(inBytes);
        IOSurfaceRef ioO = aneSurface(outBytes);
        if (!ioI || !ioO) {
            snprintf(r.err, sizeof r.err, "SETUP_SURFACE_FAIL");
            // The model is loaded here; attempt the unload even on this path.
            if (!((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                    mdl, @selector(unloadWithQoS:error:), 21, &e))
                r.unloadOK = 0;
            if (ioI) CFRelease(ioI);
            if (ioO) CFRelease(ioO);
            [fm removeItemAtPath:td error:nil];
            return r;
        }
        if (in) {
            IOSurfaceLock(ioI, 0, NULL);
            memcpy(IOSurfaceGetBaseAddress(ioI), in, inBytes);
            IOSurfaceUnlock(ioI, 0, NULL);
        }
        IOSurfaceLock(ioO, 0, NULL);
        memset(IOSurfaceGetBaseAddress(ioO), 0, IOSurfaceGetAllocSize(ioO));
        IOSurfaceUnlock(ioO, 0, NULL);

        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            AIO, @selector(objectWithIOSurface:), ioI);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            AIO, @selector(objectWithIOSurface:), ioO);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

        // One untimed evaluate is the correctness / output-capture pass.
        if (!((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e)) {
            NSString *d = aneErrTail(e, 700);
            snprintf(r.err, sizeof r.err, "EVAL_FAIL %s", d ? [d UTF8String] : "?");
        } else {
            r.evalOK = 1;
            if (outCopy) {
                IOSurfaceLock(ioO, kIOSurfaceLockReadOnly, NULL);
                memcpy(outCopy, IOSurfaceGetBaseAddress(ioO),
                       outCopyBytes < outBytes ? outCopyBytes : outBytes);
                IOSurfaceUnlock(ioO, kIOSurfaceLockReadOnly, NULL);
            }
            if (timedIters > 0) {
                double best = 1e9;
                BOOL timingFailed = NO;
                for (int blk = 0; blk < 3 && !timingFailed; blk++) {
                    uint64_t t0 = mach_absolute_time();
                    for (int i = 0; i < timedIters; i++) {
                        if (!((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                                mdl, @selector(evaluateWithQoS:options:request:error:),
                                21, @{}, req, &e)) {
                            timingFailed = YES;
                            break;
                        }
                    }
                    if (timingFailed) break;
                    double ms = ticksToMs(mach_absolute_time() - t0) / (double)timedIters;
                    if (ms < best) best = ms;
                }
                if (timingFailed) {
                    // A timed evaluation failed after the correctness pass.  Do not
                    // report a timing: the case is not a valid PASS.
                    NSString *d = aneErrTail(e, 700);
                    r.evalOK = 0;
                    r.ms = 0.0;
                    snprintf(r.err, sizeof r.err, "TIMING_EVAL_FAIL %s",
                             d ? [d UTF8String] : "?");
                } else {
                    r.ms = best;
                }
            }
        }

        // If the model loaded, its unload result is part of the case status.
        if (r.loadOK) {
            if (!((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                    mdl, @selector(unloadWithQoS:error:), 21, &e)) {
                r.unloadOK = 0;
                if (r.err[0] == 0) {
                    NSString *d = aneErrTail(e, 700);
                    snprintf(r.err, sizeof r.err, "UNLOAD_FAIL %s", d ? [d UTF8String] : "?");
                }
            }
        }
        CFRelease(ioI);
        CFRelease(ioO);
        [fm removeItemAtPath:td error:nil];
    }
    return r;
}

// ---------------------------------------------------------------- reference arithmetic
// The ANE's int8 quantize has been observed to round exact ties away from zero on
// the audited machine.  That observation is the golden behaviour here; RNE is
// counted separately so a future machine that uses RNE is not silently mislabelled.
//
// With a nonzero zero_point the observed formula is
//     q = clamp(round_half_away(x / scale + zero_point))
// i.e. the zero_point is added BEFORE the tie is resolved.  Adding an integer
// zero_point can move a tie across zero, where round-half-away changes direction;
// therefore clamp(round_half_away(x/scale) + zero_point) is not equivalent at
// those sign-crossing ties.  qReferenceAddAfter keeps that second formula so the
// probes can report how many elements discriminate the two, rather than accepting
// unexplained mismatches.
//
// The RNE comparator is the CoreML specification formula
//     q = clamp(round_half_even(x / scale) + zero_point)
// NOT round_half_even(x/scale + zero_point).  For odd zero_point these differ at
// ties: x=0.0625, scale=0.125, zp=5 gives spec RNE 5 but observed 6.
static double roundHalfAway(double v) {
    return v >= 0.0 ? floor(v + 0.5) : ceil(v - 0.5);
}

static double roundHalfEven(double v) {
    double f = floor(v);
    double d = v - f;
    if (d < 0.5) return f;
    if (d > 0.5) return f + 1.0;
    return (fmod(f, 2.0) == 0.0) ? f : f + 1.0;
}

static int clampInt8(double v) {
    if (v < -128.0) v = -128.0;
    if (v > 127.0) v = 127.0;
    return (int)v;
}

__attribute__((unused))
static int qReference(double x, double scale, int zp) {
    return clampInt8(roundHalfAway(x / scale + (double)zp));
}

__attribute__((unused))
static int qReferenceAddAfter(double x, double scale, int zp) {
    return clampInt8(roundHalfAway(x / scale) + (double)zp);
}

__attribute__((unused))
static int qReferenceRNE(double x, double scale, int zp) {
    // CoreML spec formula: round the scaled value, then add zero_point.
    return clampInt8(roundHalfEven(x / scale) + (double)zp);
}

__attribute__((unused))
static double dqReference(int q, double scale, int zp) {
    return (double)(q - zp) * scale;
}

// ---------------------------------------------------------------- MIL boilerplate
static NSString *aneMilHeader(void) {
    return @"program(1.3)\n"
            "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n{\n";
}

static NSString *aneConvConsts(void) {
    return @"        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
            "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
            "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
            "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
            "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n";
}
