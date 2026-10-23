// surface_capacity.m — how does the ANE in-memory path react to IOSurfaces that are
// smaller/larger than the compiled tensor?
//
// Kernel: fp16 [1,CH,1,SP] x -> y = x + x (no weights), so every accepted eval
// has an exact CPU golden (2*x).  The probe sweeps input and output surface
// sizes below, at, and above the declared logical size and reports, for each
// case: logical bytes, requested allocation, and the actual IOSurface allocation.
// It does not assume a universal minimum (e.g. 4096 bytes); those numbers are
// reported so any such rule can be read off the data rather than hard-coded.
//
// Inputs are deterministic dyadic values, so the golden is exact in fp16.
// Numeric checks are only made over bytes the caller actually provided/allocated.
// Setup, compile, load, eval, and unload results are checked; the exact-size case
// is required to work.  Exit status is nonzero on infrastructure failures, a
// failed exact-size case, or a numeric mismatch.
//
// Build: xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//          -o /tmp/surface_capacity surface_capacity.m
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <string.h>
#import <math.h>
#import "ane_probe_common.h"

#define CH 512
#define SP 64
static const size_t kLogical = (size_t)CH*SP*sizeof(_Float16);

static NSString *genMIL(void){
    return [NSString stringWithFormat:
      @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n"
       "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
       "        tensor<fp16, [1, %d, 1, %d]> y = add(x = x, y = x)[name = string(\"a\")];\n"
       "    } -> (y);\n}\n", CH, SP, CH, SP];
}

static void fillInput(IOSurfaceRef s, const _Float16 *vals, size_t bytes){
    IOSurfaceLock(s, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(s), vals, bytes);
    IOSurfaceUnlock(s, 0, NULL);
}

// Compare out[0..n) with 2*ref[0..n).  Returns NO on lock failure.  *maxd gets the
// largest absolute difference; any non-finite observed or reference value is
// counted in *nonfinite and forces *maxd = NAN so a tolerance test cannot pass.
static BOOL checkOutput(IOSurfaceRef out, const _Float16 *ref, size_t n,
                        double *maxd, size_t *nonfinite){
    *maxd = 0.0;
    *nonfinite = 0;
    if(IOSurfaceLock(out, kIOSurfaceLockReadOnly, NULL) != KERN_SUCCESS) return NO;
    const _Float16 *p = (const _Float16 *)IOSurfaceGetBaseAddress(out);
    for(size_t i=0;i<n;i++){
        double got = (double)p[i];
        double expect = 2.0*(double)ref[i];
        if(!isfinite(got) || !isfinite(expect)){ (*nonfinite)++; continue; }
        double d = fabs(got - expect);
        if(d > *maxd) *maxd = d;
    }
    if(*nonfinite) *maxd = NAN;
    IOSurfaceUnlock(out, kIOSurfaceLockReadOnly, NULL);
    return YES;
}

int main(void){
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",RTLD_NOW);
    @autoreleasepool {
        const size_t elems = (size_t)CH*SP;
        _Float16 *in = malloc(elems*sizeof(_Float16));
        if(!in){ printf("input alloc failed\n"); return 1; }
        for(size_t i=0;i<elems;i++)
            in[i] = (_Float16)(((int)((i*13) % 17) - 8) * 0.25f);  // [-2, 2], exact in fp16

        NSError *e=nil;
        NSData *milData=[[genMIL() dataUsingEncoding:NSUTF8StringEncoding] copy];
        Class D=NSClassFromString(@"_ANEInMemoryModelDescriptor"),I=NSClassFromString(@"_ANEInMemoryModel");
        Class AR=NSClassFromString(@"_ANERequest"),AIO=NSClassFromString(@"_ANEIOSurfaceObject");
        // Empty weights dictionary, not nil: nil makes the descriptor factory return nil.
        id desc=((id(*)(Class,SEL,id,id,id))objc_msgSend)(D,@selector(modelWithMILText:weights:optionsPlist:),milData,@{},nil);
        if(!desc){ printf("descriptor NIL\n"); free(in); return 1; }
        id mdl=((id(*)(Class,SEL,id))objc_msgSend)(I,@selector(inMemoryModelWithDescriptor:),desc);
        if(!mdl){ printf("model NIL\n"); free(in); return 1; }
        id hx=((id(*)(id,SEL))objc_msgSend)(mdl,@selector(hexStringIdentifier));
        NSString *td=[NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm=[NSFileManager defaultManager];
        if(![fm createDirectoryAtPath:td withIntermediateDirectories:YES attributes:nil error:nil]
           || ![milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES]){
            printf("temp setup failed\n");
            [fm removeItemAtPath:td error:nil];
            free(in);
            return 1;
        }
        if(!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(compileWithQoS:options:error:),21,@{},&e)){
            printf("COMPILE FAIL: %s\n",e?[[e description]UTF8String]:"(no error)");
            [fm removeItemAtPath:td error:nil];
            free(in);
            return 1;
        }
        if(!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(loadWithQoS:options:error:),21,@{},&e)){
            printf("LOAD FAIL: %s\n",e?[[e description]UTF8String]:"(no error)");
            [fm removeItemAtPath:td error:nil];
            free(in);
            return 1;
        }

        printf("declared shape [1,%d,1,%d] fp16: logical = %zu bytes\n", CH, SP, kLogical);
        printf("(requested = kIOSurfaceAllocSize; alloc = IOSurfaceGetAllocSize; "
               "no minimum size is assumed)\n\n");

        typedef struct { const char *label; size_t inB, outB; int required; } Case;
        const Case cases[] = {
            { "input 0.5x, output exact", kLogical/2, kLogical,   0 },
            { "input exact, output exact", kLogical,   kLogical,   1 },
            { "input 2x,    output exact", kLogical*2, kLogical,   0 },
            { "input exact, output 0.5x", kLogical,    kLogical/2, 0 },
            { "input exact, output 2x",   kLogical,    kLogical*2, 0 },
        };
        const int nc=(int)(sizeof(cases)/sizeof(cases[0]));
        int fails=0;

        printf("  %-28s %9s %9s %9s %9s %9s %8s\n",
               "case", "logic in", "req in", "alloc in", "req out", "alloc out", "eval");
        for(int k=0;k<nc;k++){
            IOSurfaceRef is=ane_probe_surface_create(cases[k].inB), os=ane_probe_surface_create(cases[k].outB);
            if(!is || !os){
                printf("  %-28s surface allocation FAILED\n", cases[k].label);
                if(is) CFRelease(is);
                if(os) CFRelease(os);
                fails++;
                continue;
            }
            size_t inFill = cases[k].inB < kLogical ? cases[k].inB : kLogical;
            fillInput(is, in, inFill);

            id wI=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),is);
            id wO=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),os);
            id req=((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI],@[@0],@[wO],@[@0],nil,nil,@0);
            if(!wI || !wO || !req){
                printf("  %-28s request construction FAILED\n", cases[k].label);
                CFRelease(is); CFRelease(os);
                fails++;
                continue;
            }

            e=nil;
            BOOL ok=((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl,
                @selector(evaluateWithQoS:options:request:error:),21,@{},req,&e);
            printf("  %-28s %9zu %9zu %9zu %9zu %9zu %8s\n",
                   cases[k].label, kLogical, cases[k].inB, IOSurfaceGetAllocSize(is),
                   cases[k].outB, IOSurfaceGetAllocSize(os), ok?"OK":"FAIL");

            int bad=0;
            if(ok){
                // Only check outputs backed by both a written input region and an
                // allocated output region.  This keeps undersized cases bounded.
                size_t nOutElems = (cases[k].outB < kLogical ? cases[k].outB : kLogical)
                                   / sizeof(_Float16);
                size_t nInElems  = inFill / sizeof(_Float16);
                size_t nCheck = nOutElems < nInElems ? nOutElems : nInElems;
                double maxd=0;
                size_t nonfinite=0;
                if(nCheck==0){
                    printf("      numeric check: SKIP (no overlapping bytes)\n");
                } else if(!checkOutput(os, in, nCheck, &maxd, &nonfinite)){
                    printf("      output read FAILED\n");
                    bad=1;
                } else if(nonfinite){
                    printf("      numeric check over %zu elems: NON-FINITE observed/reference (%zu elems)\n",
                           nCheck, nonfinite);
                    bad=1;
                } else {
                    printf("      numeric check over %zu elems: %s (max |diff| = %.3g)\n",
                           nCheck, (maxd <= 1e-3) ? "PASS" : "MISMATCH", maxd);
                    if(!(maxd <= 1e-3)) bad=1;
                }
            } else {
                printf("      rejected: %s\n", e?[[e description]UTF8String]:"(no error)");
                if(cases[k].required) bad=1;
            }
            if(bad) fails++;
            CFRelease(is); CFRelease(os);
        }

        e=nil;
        BOOL uok=((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl,@selector(unloadWithQoS:error:),21,&e);
        if(!uok){ printf("\nunload FAIL: %s\n",e?[[e description]UTF8String]:"(no error)"); fails++; }
        [fm removeItemAtPath:td error:nil];
        free(in);
        printf("\n%s\n", fails?"FAILURES: some cases did not meet expectations":"all expected cases passed");
        return fails?1:0;
    }
}
