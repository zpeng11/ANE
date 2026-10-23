// mapping_overhead.m — decompose the per-eval overhead.
//  A) trivial kernel, reuse request, plain eval loop            (baseline)
//  B) same but pre-map the IOSurfaces once (mapIOSurfacesWithRequest:cacheInference:)
//  C) kernel with 2 or 3 outputs instead of 1  -> does cost scale with tensor count?
//  D) measure the cost of the ObjC call path itself: eval a kernel whose input is tiny
//
// Every return value that can fail is checked.  Setup, compile, load, map, warmup
// evals, and timed evals that fail are reported and are never printed as a valid
// timing.  A premap timing is only reported if mapIOSurfacesWithRequest: itself
// succeeded; the matching unmap is always issued when the map succeeded, and
// surfaces/model are released/unloaded on every path.
//
// Surfaces use the ANE's 64-byte per-channel row stride: the allocation is
// channels * align64(spatial*2), not max(logical,4096).  Tiny S=1 kernels
// therefore need C*64 bytes, which is what fixes the 0x1d mapping failures.
// Inputs are filled over the whole allocation with deterministic finite values,
// since padded layouts are not contiguous.  Timing only; no numeric assertion.
//
// Build: xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//          -o /tmp/mapping_overhead mapping_overhead.m
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

static BOOL fillSurface(IOSurfaceRef s){
    if(IOSurfaceLock(s, 0, NULL) != KERN_SUCCESS) return NO;
    size_t elems = IOSurfaceGetAllocSize(s) / sizeof(_Float16);
    _Float16 *p=(_Float16 *)IOSurfaceGetBaseAddress(s);
    for(size_t i=0;i<elems;i++) p[i]=(_Float16)(0.25f + (float)(i%7)*0.125f);
    IOSurfaceUnlock(s, 0, NULL);
    return YES;
}

static NSString *genMIL(int ch,int sp,int nout){
    NSMutableString *m=[NSMutableString string];
    [m appendString:ANE_PROBE_MIL_HEADER];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n",ch,sp];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> a = add(x = x, y = x)[name = string(\"a\")];\n",ch,sp];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> b = mul(x = a, y = a)[name = string(\"b\")];\n",ch,sp];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> c = sub(x = b, y = a)[name = string(\"c\")];\n",ch,sp];
    if(nout==1) [m appendString:@"    } -> (a);\n}\n"];
    else if(nout==2) [m appendString:@"    } -> (a, b);\n}\n"];
    else [m appendString:@"    } -> (a, b, c);\n}\n"];
    return m;
}

static int evalOnce(id mdl, id req, NSError **e){
    return ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        mdl,@selector(evaluateWithQoS:options:request:error:),21,@{},req,e) ? 0 : 1;
}

static int run(const char *label,int ch,int sp,int nout,int iters,BOOL preMap){
    @autoreleasepool{
        NSError *e=nil;
        NSData *milData=nil;
        id desc=nil, mdl=nil, hx=nil, wI=nil, req=nil;
        NSMutableArray *outs=nil, *oidx=nil;
        NSString *td=nil;
        NSFileManager *fm=[NSFileManager defaultManager];
        Class D=NSClassFromString(@"_ANEInMemoryModelDescriptor"),I=NSClassFromString(@"_ANEInMemoryModel");
        Class AR=NSClassFromString(@"_ANERequest"),AIO=NSClassFromString(@"_ANEIOSurfaceObject");
        IOSurfaceRef in=NULL, os[3]={NULL,NULL,NULL};
        size_t logical=(size_t)ch*sp*sizeof(_Float16), need=surfaceBytes(ch,sp), ai=0, ao=0;
        BOOL mapped=NO;
        int rc=1;

        milData=[[genMIL(ch,sp,nout) dataUsingEncoding:NSUTF8StringEncoding] copy];
        if(!milData){ printf("  %-38s MIL allocation failed\n",label); goto done; }
        desc=((id(*)(Class,SEL,id,id,id))objc_msgSend)(D,@selector(modelWithMILText:weights:optionsPlist:),milData,@{},nil);
        if(!desc){ printf("  %-38s descriptor NIL\n",label); goto done; }
        mdl=((id(*)(Class,SEL,id))objc_msgSend)(I,@selector(inMemoryModelWithDescriptor:),desc);
        if(!mdl){ printf("  %-38s model NIL\n",label); goto done; }
        hx=((id(*)(id,SEL))objc_msgSend)(mdl,@selector(hexStringIdentifier));
        td=[NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        if(![fm createDirectoryAtPath:td withIntermediateDirectories:YES attributes:nil error:nil]
           || ![milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES]){
            printf("  %-38s temp setup failed\n",label); goto done;
        }
        if(!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(compileWithQoS:options:error:),21,@{},&e)){
            printf("  %-38s compile fail: %s\n",label,e?[[e description]UTF8String]:"(no error)"); goto done;
        }
        if(!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(loadWithQoS:options:error:),21,@{},&e)){
            printf("  %-38s load fail: %s\n",label,e?[[e description]UTF8String]:"(no error)"); goto done;
        }

        in=ane_probe_surface_create_aligned(need,&ai);
        if(!in){ printf("  %-38s input surface creation failed\n",label); goto done; }
        if(!fillSurface(in)){ printf("  %-38s input fill failed\n",label); goto done; }
        wI=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),in);
        if(!wI){ printf("  %-38s input wrapper failed\n",label); goto done; }

        outs=[NSMutableArray array]; oidx=[NSMutableArray array];
        for(int k=0;k<nout;k++){
            os[k]=ane_probe_surface_create_aligned(need,&ao);
            if(!os[k]){ printf("  %-38s output surface %d creation failed\n",label,k); goto done; }
            id w=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),os[k]);
            if(!w){ printf("  %-38s output wrapper %d failed\n",label,k); goto done; }
            [outs addObject:w]; [oidx addObject:@(k)];
        }
        req=((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI],@[@0],outs,oidx,nil,nil,@0);
        if(!req){ printf("  %-38s request construction failed\n",label); goto done; }

        if(preMap){
            e=nil;
            mapped=((BOOL(*)(id,SEL,id,BOOL,NSError**))objc_msgSend)(mdl,
                @selector(mapIOSurfacesWithRequest:cacheInference:error:),req,YES,&e);
            if(!mapped){
                printf("  %-38s premap FAILED: %s (no timing reported)\n",label,e?[[e description]UTF8String]:"(no error)");
                goto done;
            }
        }

        for(int k=0;k<20;k++){
            e=nil;
            if(evalOnce(mdl,req,&e)){
                printf("  %-38s warmup eval %d FAILED: %s\n",label,k,e?[[e description]UTF8String]:"(no error)");
                goto done;
            }
        }
        uint64_t t0=mach_absolute_time();
        for(int k=0;k<iters;k++){
            e=nil;
            if(evalOnce(mdl,req,&e)){
                printf("  %-38s timed eval %d/%d FAILED: %s\n",label,k,iters,e?[[e description]UTF8String]:"(no error)");
                goto done;
            }
        }
        printf("  %-38s logical=%7zu need=%6zu alloc=%7zu %7.2f us/eval (%d/%d OK)\n",
               label,logical,need,ao,ane_probe_ticks_to_us(mach_absolute_time()-t0)/iters,iters,iters);
        rc=0;

    done:
        if(mapped){
            if([mdl respondsToSelector:@selector(unmapIOSurfacesWithRequest:)])
                ((void(*)(id,SEL,id))objc_msgSend)(mdl,@selector(unmapIOSurfacesWithRequest:),req);
            else
                printf("  %-38s WARNING: no unmapIOSurfacesWithRequest: selector\n",label);
        }
        if(mdl){
            e=nil;
            if(!((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl,@selector(unloadWithQoS:error:),21,&e)){
                printf("  %-38s unload FAIL: %s\n",label,e?[[e description]UTF8String]:"(no error)");
                rc=1;
            }
        }
        if(in) CFRelease(in);
        for(int k=0;k<nout;k++) if(os[k]) CFRelease(os[k]);
        if(td) [fm removeItemAtPath:td error:nil];
        return rc;
    }
}

int main(void){
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",RTLD_NOW);
    printf("=== decomposing the per-eval overhead ===\n");
    printf("surface bytes = channels * align64(spatial*2); timing only, evals checked\n");
    int fails=0;
    fails += run("A) trivial, 1 output, no premap",        256,1,1,2000,NO);
    fails += run("B) trivial, 1 output, premap",           256,1,1,2000,YES);
    fails += run("C) trivial, 2 outputs",                  256,1,2,2000,NO);
    fails += run("C) trivial, 3 outputs",                  256,1,3,2000,NO);
    fails += run("D) real work, 1 output (768x256)",       768,256,1,1000,NO);
    fails += run("D) real work, 1 output, premap",         768,256,1,1000,YES);
    printf("\n%s (%d failing case%s)\n", fails?"FAIL":"PASS", fails, fails==1?"":"s");
    return fails?1:0;
}
