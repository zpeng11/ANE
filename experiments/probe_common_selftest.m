// probe_common_selftest.m — offline checks for ane_probe_common.h helpers.
//
// No ANE/private framework is loaded.  The BLOBFILE v2 writer is checked
// byte-for-byte against the documented container layout, and the op_probe
// source is checked for the acceptance-only reporting regression (no
// hardcoded `chosen = "compact"`).
//
// Usage:
//   ./probe_common_selftest [path/to/op_probe.m]
// The op_probe path defaults to "op_probe.m" in the current directory; if it
// cannot be read the source check is reported as SKIP (not a failure).
//
// Build: xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface \
//          -ldl -o /tmp/probe_common_selftest probe_common_selftest.m
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#import "ane_probe_common.h"

static int g_fail = 0;
static int g_checks = 0;

static void check(int cond, const char *what) {
    g_checks++;
    if (!cond) {
        printf("FAIL %s\n", what);
        g_fail++;
    } else {
        printf("ok   %s\n", what);
    }
}

// Little-endian readers; the container is explicitly little-endian and the
// payload is compared bytewise, so no host-endian assumptions are made.
static uint32_t rd32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t rd64(const uint8_t *p) {
    return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32);
}

static void test_blob(void) {
    // Two chunks of three fp16 values: payload 6 bytes, span 64+64 = 128.
    const _Float16 w[6] = { 1.0f, -2.5f, 0.25f, 3.0f, 0.5f, -0.75f };
    NSData *d = ane_probe_blob_fp16(w, 2, 3);
    check(d != nil, "blob_fp16 returns data");
    if (!d) return;

    const uint8_t *b = (const uint8_t *)d.bytes;
    NSUInteger n = d.length;
    check(n == 64 + 2 * 128, "blob length = header + 2 aligned chunks");
    if (n < 64 + 2 * 128) return;

    check(rd32(b + 0) == 2, "file header chunk count = 2");
    check(rd32(b + 4) == 2, "file header version = 2");

    for (int i = 0; i < 2; i++) {
        NSUInteger off = 64 + (NSUInteger)i * 128;
        const uint8_t *h = b + off;
        check(h[0] == 0xEF && h[1] == 0xBE && h[2] == 0xAD && h[3] == 0xDE,
              i == 0 ? "chunk0 magic 0xDEADBEEF" : "chunk1 magic 0xDEADBEEF");
        check(rd32(h + 4) == ANE_PROBE_BLOB_FP16,
              i == 0 ? "chunk0 dtype fp16" : "chunk1 dtype fp16");
        check(rd64(h + 8) == 6,
              i == 0 ? "chunk0 size = 6 bytes" : "chunk1 size = 6 bytes");
        check(rd64(h + 16) == (uint64_t)(off + 64),
              i == 0 ? "chunk0 offset points past header"
                     : "chunk1 offset points past header");
        check(memcmp(h + 24, "\0\0\0\0\0\0\0\0", 8) == 0,
              i == 0 ? "chunk0 padding-size field is zero"
                     : "chunk1 padding-size field is zero");
        check(memcmp(h + 64, (const uint8_t *)w + i * 6, 6) == 0,
              i == 0 ? "chunk0 payload bytes match" : "chunk1 payload bytes match");
        // The rest of the payload slot is zero padding.
        int padZero = 1;
        for (NSUInteger k = 64 + 6; k < 64 + 64; k++) if (h[k] != 0) padZero = 0;
        check(padZero, i == 0 ? "chunk0 padding is zero" : "chunk1 padding is zero");
    }

    check(ane_probe_blob_fp16(w, 0, 3) == nil, "zero chunks rejected");
    check(ane_probe_blob_fp16(w, 2, 0) == nil, "zero elements-per-chunk rejected");
    check(ane_probe_blob_fp16(NULL, 2, 3) == nil, "NULL weights rejected");

    // Overflow guards: every one of these must return nil without allocating,
    // so a wrapped size can never pair a small calloc with a large memcpy.
    check(ane_probe_align64(NSUIntegerMax) == 0, "align64 overflow returns sentinel");
    check(ane_probe_chunk_span(NSUIntegerMax) == 0, "chunk_span overflow returns 0");
    check(ane_probe_chunk_span(NSUIntegerMax - 64u) == 0,
          "chunk_span overflow near NSUIntegerMax returns 0");
    check(ane_probe_chunk_span(0) == 64, "chunk_span(0) is header only");
    check(ane_probe_blob_fp16(w, 1, NSUIntegerMax) == nil,
          "elemsPerChunk*sizeof overflow rejected");
    check(ane_probe_blob_fp16(w, 1, NSUIntegerMax / 2u + 1u) == nil,
          "elemsPerChunk above NSUIntegerMax/sizeof rejected");
    check(ane_probe_blob_fp16(w, 2147483647, NSUIntegerMax / 2u) == nil,
          "huge chunk count rejected");
}

static void test_timing(void) {
    check(ane_probe_ticks_to_us(0) == 0.0, "ticks_to_us(0) = 0");
    check(ane_probe_ticks_to_ms(0) == 0.0, "ticks_to_ms(0) = 0");
    double us = ane_probe_ticks_to_us(1000000);
    double ms = ane_probe_ticks_to_ms(1000000);
    check(isfinite(us) && us > 0.0, "ticks_to_us positive and finite");
    check(isfinite(ms) && ms > 0.0, "ticks_to_ms positive and finite");
    check(fabs(ms * 1000.0 - us) < 1e-6 * (us > 1.0 ? us : 1.0),
          "us and ms conversions agree");
}

static void test_surfaces(void) {
    // IOSurface is local to the process, but stay tolerant: if creation is
    // unavailable in this environment report SKIP rather than failing.
    IOSurfaceRef s = ane_probe_surface_create(1);
    if (!s) {
        printf("SKIP IOSurface creation unavailable in this environment\n");
        return;
    }
    check(IOSurfaceGetAllocSize(s) >= 1, "plain surface alloc >= requested");
    CFRelease(s);

    check(ane_probe_surface_create(0) == NULL, "zero-byte plain surface rejected");

    size_t alloc = 0;
    IOSurfaceRef a = ane_probe_surface_create_aligned(1, &alloc);
    if (!a) {
        printf("SKIP aligned IOSurface creation unavailable\n");
        return;
    }
    check(alloc >= 4096, "aligned surface alloc >= 4096");
    check((alloc % 4096) == 0, "aligned surface alloc is 4 KiB multiple");
    CFRelease(a);
    check(ane_probe_surface_create_aligned(0, &alloc) == NULL,
          "zero-byte aligned surface rejected");

    size_t allocE = 123;
    check(ane_probe_surface_create_aligned(SIZE_MAX, &allocE) == NULL,
          "SIZE_MAX aligned bytes rejected");
    check(allocE == 0, "rejected aligned surface clears allocOut");
    check(ane_probe_surface_create_aligned(SIZE_MAX - 100u, &allocE) == NULL,
          "SIZE_MAX-100 aligned bytes rejected");
}

// Source-level regression guard for the acceptance-only reporting fix.
static void test_op_probe_source(const char *path) {
    NSString *src = [NSString stringWithContentsOfFile:
        [NSString stringWithUTF8String:path] encoding:NSUTF8StringEncoding error:nil];
    if (!src) {
        printf("SKIP op_probe source not readable at %s\n", path);
        return;
    }
    check([src rangeOfString:@"chosen = \"compact\""].location == NSNotFound,
          "op_probe no longer hardcodes chosen=compact");
    check([src rangeOfString:@"triedname"].location != NSNotFound,
          "op_probe records the actually tried layout");
    check([src rangeOfString:@"ACCEPTANCE_ONLY"].location != NSNotFound,
          "op_probe keeps the acceptance-only marker");
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *probePath = argc > 1 ? argv[1] : "op_probe.m";
        printf("=== ane_probe_common offline selftest ===\n");
        test_blob();
        test_timing();
        test_surfaces();
        test_op_probe_source(probePath);
        printf("\n%d checks, %d failed\n", g_checks, g_fail);
        return g_fail ? 1 : 0;
    }
}
