// ane_probe_common.h — small, stable helpers shared by the ANE probes.
//
// Scope is deliberately narrow: only byte-identical infrastructure is here.
// Quantization semantics, JSON output and the case runner stay in
// ane_quant_common.h; this header must not grow into a framework.
//
// Everything is static inline and self-contained, so a probe can include it
// without adding a build dependency beyond this header.
//
// Contents:
//   * ANE_PROBE_MIL_HEADER — the CoreML MIL 1.3 buildInfo preamble used by
//     every in-memory probe (byte-identical across probes).
//   * lazy mach timebase conversion helpers (ticks -> us/ms)
//   * CoreML BLOBFILE v2 primitives and an FP16 multi-chunk writer
//   * flat IOSurface allocation helpers (plain and 4 KiB-rounded)
#ifndef ANE_PROBE_COMMON_H
#define ANE_PROBE_COMMON_H

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <dispatch/dispatch.h>
#import <mach/mach_time.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Standard MIL preamble.  Kept as an NSString literal so appendString: users
// can share it directly.  Probes that build a format string inline keep their
// own copy; the bytes are identical.
#define ANE_PROBE_MIL_HEADER \
    @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n"

// ---------------------------------------------------------------- timing
// The timebase is fixed for the process; initialise it exactly once so probes
// do not need a separate mach_timebase_info() call in main().  dispatch_once_f
// keeps this safe even if a future probe calls the helpers from a worker.
static inline void ane_probe_init_timebase(void *ctx) {
    mach_timebase_info((mach_timebase_info_data_t *)ctx);
}

static inline double ane_probe_ticks_to_us(uint64_t t) {
    static mach_timebase_info_data_t tb;
    static dispatch_once_t once;
    dispatch_once_f(&once, &tb, ane_probe_init_timebase);
    return (double)t * (double)tb.numer / (double)tb.denom / 1e3;
}

static inline double ane_probe_ticks_to_ms(uint64_t t) {
    static mach_timebase_info_data_t tb;
    static dispatch_once_t once;
    dispatch_once_f(&once, &tb, ane_probe_init_timebase);
    return (double)t * (double)tb.numer / (double)tb.denom / 1e6;
}

// ---------------------------------------------------------------- BLOBFILE v2
//   64-byte storage header: uint32 entry count @0, uint32 version=2 @4
//   per entry, 64-byte metadata: 0xDEADBEEF @0, dtype @4 (1 = FLOAT16,
//   coremltools BlobDataType), uint64 size @8, uint64 absolute data offset @16,
//   padding-size-in-bits @24 (0), followed by the 64-byte-aligned payload.
enum {
    ANE_PROBE_BLOB_FP16 = 1,
    ANE_PROBE_BLOB_INT8 = 4,
};

static inline NSUInteger ane_probe_align64(NSUInteger n) {
    // 0 is returned as an overflow sentinel (a real align64 is never 0 for
    // n > 0 because the result is a positive multiple of 64).
    if (n > NSUIntegerMax - 63u) return 0;
    return (n + 63u) & ~(NSUInteger)63u;
}

// Bytes occupied by one chunk: 64-byte header plus the aligned payload.
// Returns 0 when the size would overflow (chunk_span(0) is a valid 64).
static inline NSUInteger ane_probe_chunk_span(NSUInteger payloadBytes) {
    NSUInteger a = ane_probe_align64(payloadBytes);
    if (a == 0 && payloadBytes > 0) return 0;
    if (a > NSUIntegerMax - 64u) return 0;
    return 64u + a;
}

static inline void ane_probe_write_file_header(uint8_t *h, uint32_t chunkCount) {
    memset(h, 0, 64);
    memcpy(h + 0, &chunkCount, sizeof chunkCount);
    uint32_t version = 2;
    memcpy(h + 4, &version, sizeof version);
}

static inline void ane_probe_write_chunk_header(uint8_t *h, uint32_t dtype,
                                                uint64_t bytes, uint64_t absOffset) {
    memset(h, 0, 64);
    h[0] = 0xEF; h[1] = 0xBE; h[2] = 0xAD; h[3] = 0xDE;
    memcpy(h + 4, &dtype, sizeof dtype);
    memcpy(h + 8, &bytes, sizeof bytes);
    memcpy(h + 16, &absOffset, sizeof absOffset);
}

// FP16 weights, one chunk per nchunks, all chunks the same element count.
// Returns nil on invalid input or allocation failure.  Ownership: the returned
// NSData owns the malloc'd buffer (freeWhenDone).
static inline NSData *ane_probe_blob_fp16(const _Float16 *weights, int nchunks,
                                          NSUInteger elemsPerChunk) {
    if (nchunks < 1 || elemsPerChunk == 0 || !weights) return nil;
    // Reject every size that would wrap before touching the allocator; an
    // unchecked wrap here would pair a small calloc with a large memcpy.
    if (elemsPerChunk > NSUIntegerMax / sizeof(_Float16)) return nil;
    NSUInteger payload = elemsPerChunk * sizeof(_Float16);
    NSUInteger stride  = ane_probe_chunk_span(payload);
    if (stride == 0) return nil;
    if ((NSUInteger)nchunks > (NSUIntegerMax - 64u) / stride) return nil;
    NSUInteger total   = 64 + stride * (NSUInteger)nchunks;
    uint8_t *b = calloc(total, 1);
    if (!b) return nil;
    ane_probe_write_file_header(b, (uint32_t)nchunks);
    for (int i = 0; i < nchunks; i++) {
        uint8_t *h = b + 64 + (NSUInteger)i * stride;
        ane_probe_write_chunk_header(h, ANE_PROBE_BLOB_FP16, (uint64_t)payload,
                                     (uint64_t)(64 + (NSUInteger)i * stride + 64));
        memcpy(h + 64, weights + (NSUInteger)i * elemsPerChunk, payload);
    }
    return [NSData dataWithBytesNoCopy:b length:total freeWhenDone:YES];
}

// ---------------------------------------------------------------- IOSurface
// Flat byte buffer; the size is used exactly as given.  Returns NULL for 0.
static inline IOSurfaceRef ane_probe_surface_create(size_t bytes) {
    if (bytes == 0) return NULL;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

// Flat byte buffer with the requested size rounded up to a 4 KiB multiple.
// Use when the caller cares about allocator granularity rather than the exact
// logical tensor size.  *allocOut (optional) receives the actual allocation.
static inline IOSurfaceRef ane_probe_surface_create_aligned(size_t bytes,
                                                            size_t *allocOut) {
    if (allocOut) *allocOut = 0;
    if (bytes == 0 || bytes > SIZE_MAX - 4095u) return NULL;
    size_t b = (bytes + 4095u) & ~(size_t)4095u;
    IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(b), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(b),
        (id)kIOSurfaceAllocSize:@(b), (id)kIOSurfacePixelFormat:@0});
    if (s && allocOut) *allocOut = IOSurfaceGetAllocSize(s);
    return s;
}

#endif // ANE_PROBE_COMMON_H
