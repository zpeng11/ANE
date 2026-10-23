// dispatch_latency.m — minimum per-eval dispatch cost on ANE (latency floor for small ops).
//
// Every setup step, warmup eval, timed eval, and unload is checked.  A case whose
// compile/load fails, whose surfaces cannot be allocated, or whose evals return
// NO is reported as FAILED and is never printed as a timing.  The process exits
// nonzero if any case fails.
//
// The ANE maps [1,C,1,S] fp16 tensors with a 64-byte per-channel row stride, so
// the surface must hold C * align64(S*2) bytes even when S*2 is tiny (e.g. S=1:
// 64 bytes per channel).  This is a stride requirement, not a blanket
// "max(logical,4096)" floor; C=64,S=1 happens to need exactly 4096.
//
// Inputs are filled over the whole allocation with deterministic finite values
// so a padded layout still sees defined data.  This probe is timing-only: eval
// success is checked, numeric correctness is not asserted here.
//
// Build: xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//          -o /tmp/dispatch_latency dispatch_latency.m
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>
#import <string.h>
#import "ane_probe_common.h"

static size_t alignUp(size_t v, size_t a){ return (v + a - 1) & ~(a - 1); }
static size_t surfaceBytes(int ch, int sp){
    return (size_t)ch * alignUp((size_t)sp*sizeof(_Float16), 64);
}

// Fill the whole allocation, not just the logical tensor: with a padded channel
// stride the logical elements are not a contiguous prefix.  Timing-only probe,
// so input values just need to be deterministic and finite.
static BOOL fillSurface(IOSurfaceRef s){
    if(IOSurfaceLock(s, 0, NULL) != KERN_SUCCESS) return NO;
    size_t elems = IOSurfaceGetAllocSize(s) / sizeof(_Float16);
    _Float16 *p=(_Float16 *)IOSurfaceGetBaseAddress(s);
    for(size_t i=0;i<elems;i++) p[i]=(_Float16)(0.25f + (float)(i%7)*0.125f);
    IOSurfaceUnlock(s, 0, NULL);
    return YES;
}

// trivial program: out = x + x
static NSString *genMIL(int ch,int sp){
    return [NSString stringWithFormat:
      @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n"
       "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
       "        tensor<fp16, [1, %d, 1, %d]> c = add(x = x, y = x)[name = string(\"a\")];\n"
       "    } -> (c);\n}\n", ch,sp,ch,sp];
}

static int evalOnce(id mdl, id req, NSError **e){
    return ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        mdl,@selector(evaluateWithQoS:options:request:error:),21,@{},req,e) ? 0 : 1;
}

static int run(int ch,int sp,int iters){
    @autoreleasepool{
        NSError *e=nil;
        NSData *milData=nil;
        id desc=nil, mdl=nil, hx=nil, wI=nil, wO=nil, req=nil;
        NSString *td=nil;
        NSFileManager *fm=[NSFileManager defaultManager];
        Class D=NSClassFromString(@"_ANEInMemoryModelDescriptor"),I=NSClassFromString(@"_ANEInMemoryModel");
        Class AR=NSClassFromString(@"_ANERequest"),AIO=NSClassFromString(@"_ANEIOSurfaceObject");
        IOSurfaceRef i=NULL, o=NULL;
        size_t logical=(size_t)ch*sp*sizeof(_Float16), need=surfaceBytes(ch,sp), ai=0, ao=0;
        int rc=1;

        milData=[[genMIL(ch,sp) dataUsingEncoding:NSUTF8StringEncoding] copy];
        if(!milData){ printf("  [1,%d,1,%d] MIL allocation failed\n",ch,sp); goto done; }
        desc=((id(*)(Class,SEL,id,id,id))objc_msgSend)(D,@selector(modelWithMILText:weights:optionsPlist:),milData,@{},nil);
        if(!desc){ printf("  [1,%d,1,%d] descriptor NIL\n",ch,sp); goto done; }
        mdl=((id(*)(Class,SEL,id))objc_msgSend)(I,@selector(inMemoryModelWithDescriptor:),desc);
        if(!mdl){ printf("  [1,%d,1,%d] model NIL\n",ch,sp); goto done; }
        hx=((id(*)(id,SEL))objc_msgSend)(mdl,@selector(hexStringIdentifier));
        td=[NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        if(![fm createDirectoryAtPath:td withIntermediateDirectories:YES attributes:nil error:nil]
           || ![milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES]){
            printf("  [1,%d,1,%d] temp setup failed\n",ch,sp); goto done;
        }
        if(!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(compileWithQoS:options:error:),21,@{},&e)){
            printf("  [1,%d,1,%d] compile fail: %s\n",ch,sp,e?[[e description]UTF8String]:"(no error)"); goto done;
        }
        if(!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(loadWithQoS:options:error:),21,@{},&e)){
            printf("  [1,%d,1,%d] load fail: %s\n",ch,sp,e?[[e description]UTF8String]:"(no error)"); goto done;
        }

        i=ane_probe_surface_create_aligned(need,&ai); o=ane_probe_surface_create_aligned(need,&ao);
        if(!i || !o){ printf("  [1,%d,1,%d] IOSurface creation failed\n",ch,sp); goto done; }
        if(!fillSurface(i)){ printf("  [1,%d,1,%d] input fill failed\n",ch,sp); goto done; }

        wI=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),i);
        wO=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),o);
        req=((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI],@[@0],@[wO],@[@0],nil,nil,@0);
        if(!wI || !wO || !req){ printf("  [1,%d,1,%d] request construction failed\n",ch,sp); goto done; }

        for(int k=0;k<20;k++){
            e=nil;
            if(evalOnce(mdl,req,&e)){
                printf("  [1,%d,1,%d] warmup eval %d FAILED: %s\n",ch,sp,k,e?[[e description]UTF8String]:"(no error)");
                goto done;
            }
        }
        uint64_t t0=mach_absolute_time();
        for(int k=0;k<iters;k++){
            e=nil;
            if(evalOnce(mdl,req,&e)){
                printf("  [1,%d,1,%d] timed eval %d/%d FAILED: %s\n",ch,sp,k,iters,e?[[e description]UTF8String]:"(no error)");
                goto done;
            }
        }
        double d=ane_probe_ticks_to_us(mach_absolute_time()-t0)/iters;
        printf("  [1,%-5d,1,%-5d] logical=%7zu B need=%6zu B alloc=%7zu B  %8.2f us/eval (%d/%d OK)\n",
               ch,sp,logical,need,ao,d,iters,iters);
        rc=0;

    done:
        if(mdl){
            e=nil;
            if(!((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl,@selector(unloadWithQoS:error:),21,&e)){
                printf("  [1,%d,1,%d] unload FAIL: %s\n",ch,sp,e?[[e description]UTF8String]:"(no error)");
                rc=1;
            }
        }
        if(i) CFRelease(i);
        if(o) CFRelease(o);
        if(td) [fm removeItemAtPath:td error:nil];
        return rc;
    }
}

int main(void){
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",RTLD_NOW);
    printf("=== ANE per-eval dispatch floor (trivial add kernel) ===\n");
    printf("surface bytes = channels * align64(spatial*2); timing only, evals checked\n");
    int fails=0;
    fails += run(64,1,2000);
    fails += run(256,1,2000);
    fails += run(768,1,2000);
    fails += run(768,256,1000);
    fails += run(2048,256,500);
    fails += run(4096,256,200);
    printf("\n%s (%d failing case%s)\n", fails?"FAIL":"PASS", fails, fails==1?"":"s");
    return fails?1:0;
}
