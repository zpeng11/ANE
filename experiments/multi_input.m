// multi_input.m — multi-input request behaviour with numeric verification.
//
// Cases:
//   1. 1 input,  no weights, weights argument nil  -> descriptor is expected nil
//      (observed API trap: pass @{} even when the MIL has no BLOBFILE)
//   2. 1 input,  no weights, weights argument {}   -> y = x + x
//   3. 2 inputs, no weights, weights argument {}   -> y = x + y
//   4. 2 inputs + compiled BLOBFILE weights        -> y = conv(x + y, W)
//
// Every accepted case is checked against an exact deterministic CPU golden, so
// each input's contribution is verified numerically rather than only by "EVAL OK".
// The weight blob uses the standard v2 layout (dtype @4, uint64 size @8,
// uint64 offset @16).  All setup/compile/load/eval/unload results are checked and
// the process exits nonzero if any expected case fails or any golden mismatches.
//
// Build: xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//          -o /tmp/multi_input multi_input.m
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <string.h>
#import <math.h>
#import "ane_probe_common.h"

#define CH 64
#define SP 32
#define TOL 1e-2

static NSString *genMIL(int nin, BOOL withWeights){
    NSMutableString *m=[NSMutableString string];
    [m appendString:ANE_PROBE_MIL_HEADER];
    if(nin==1) [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", CH, SP];
    else       [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x, tensor<fp16, [1, %d, 1, %d]> y) {\n", CH, SP, CH, SP];

    if(nin==1){
        [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> s = add(x = x, y = x)[name = string(\"sum\")];\n", CH, SP];
    } else {
        [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> s = add(x = x, y = y)[name = string(\"sum\")];\n", CH, SP];
    }

    if(withWeights){
        [m appendString:@"        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n"
            @"        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
            @"        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
            @"        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
            @"        int32 gp = const()[name = string(\"gp\"), val = int32(1)];\n"];
        [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n",
            CH, CH, CH, CH];
        [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> c = conv(dilations = dl, groups = gp, pad = pd, pad_type = pt, strides = st, weight = W, x = s)[name = string(\"conv\")];\n", CH, SP];
        [m appendString:@"    } -> (c);\n}\n"];
    } else {
        [m appendString:@"    } -> (s);\n}\n"];
    }
    return m;
}

static BOOL writeSurface(IOSurfaceRef s, const _Float16 *src, size_t elems){
    if(IOSurfaceLock(s, 0, NULL) != KERN_SUCCESS) return NO;
    memcpy(IOSurfaceGetBaseAddress(s), src, elems*sizeof(_Float16));
    IOSurfaceUnlock(s, 0, NULL);
    return YES;
}

static BOOL readSurface(IOSurfaceRef s, _Float16 *dst, size_t elems){
    if(IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL) != KERN_SUCCESS) return NO;
    memcpy(dst, IOSurfaceGetBaseAddress(s), elems*sizeof(_Float16));
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
    return YES;
}

typedef enum { G_NONE, G_DOUBLE, G_SUM, G_CONV } Golden;

static int tryCase(const char *label, int nin, BOOL withWeights, BOOL nilWeights,
                   Golden golden, const _Float16 *x, const _Float16 *y,
                   const _Float16 *w, size_t elems)
{
    @autoreleasepool {
        NSError *e=nil;
        NSData *milData=nil, *wb=nil;
        id desc=nil, mdl=nil, hx=nil, wI0=nil, wI1=nil, wO=nil, req=nil, weightsArg=nil;
        NSMutableArray *ins=nil, *idx=nil;
        NSString *td=nil, *wdir=nil;
        Class D=NSClassFromString(@"_ANEInMemoryModelDescriptor"),I=NSClassFromString(@"_ANEInMemoryModel");
        Class AR=NSClassFromString(@"_ANERequest"),AIO=NSClassFromString(@"_ANEIOSurfaceObject");
        NSFileManager *fm=[NSFileManager defaultManager];
        IOSurfaceRef sI0=NULL, sI1=NULL, sO=NULL;
        int rc=1;

        milData=[[genMIL(nin,withWeights) dataUsingEncoding:NSUTF8StringEncoding] copy];
        // This MIL has exactly one BLOBFILE entry, so one chunk is the actual count.
        if(withWeights) wb=ane_probe_blob_fp16(w,1,(size_t)CH*CH);
        if(!milData || (withWeights && !wb)){ printf("  %-40s host allocation failed\n",label); goto done; }
        weightsArg = nilWeights ? nil
                     : (withWeights ? @{@"@model_path/weights/weight.bin":@{@"offset":@0,@"data":wb}} : @{});

        desc=((id(*)(Class,SEL,id,id,id))objc_msgSend)(D,@selector(modelWithMILText:weights:optionsPlist:),
            milData,weightsArg,nil);
        if(nilWeights){
            if(desc){
                printf("  %-40s UNEXPECTED: descriptor non-nil with nil weights argument\n",label);
            } else {
                printf("  %-40s PASS (descriptor nil as expected)\n",label);
                rc=0;
            }
            goto done;
        }
        if(!desc){ printf("  %-40s descriptor NIL\n",label); goto done; }
        mdl=((id(*)(Class,SEL,id))objc_msgSend)(I,@selector(inMemoryModelWithDescriptor:),desc);
        if(!mdl){ printf("  %-40s model NIL\n",label); goto done; }
        hx=((id(*)(id,SEL))objc_msgSend)(mdl,@selector(hexStringIdentifier));
        td=[NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        wdir=[td stringByAppendingPathComponent:@"weights"];
        if(![fm createDirectoryAtPath:wdir withIntermediateDirectories:YES attributes:nil error:nil]
           || ![milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES]
           || (wb && ![wb writeToFile:[wdir stringByAppendingPathComponent:@"weight.bin"] atomically:YES])){
            printf("  %-40s temp setup failed\n",label); goto done;
        }
        if(!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(compileWithQoS:options:error:),21,@{},&e)){
            printf("  %-40s COMPILE FAIL: %s\n",label,e?[[e description]UTF8String]:"(no error)"); goto done;
        }
        if(!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(loadWithQoS:options:error:),21,@{},&e)){
            printf("  %-40s LOAD FAIL: %s\n",label,e?[[e description]UTF8String]:"(no error)"); goto done;
        }

        size_t bytes=elems*sizeof(_Float16);
        sI0=ane_probe_surface_create(bytes);
        if(nin==2) sI1=ane_probe_surface_create(bytes);
        sO=ane_probe_surface_create(bytes);
        if(!sI0 || (nin==2 && !sI1) || !sO){ printf("  %-40s IOSurface creation failed\n",label); goto done; }
        if(!writeSurface(sI0,x,elems) || (nin==2 && !writeSurface(sI1,y,elems))){
            printf("  %-40s IOSurface write failed\n",label); goto done;
        }

        wI0=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),sI0);
        wO=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),sO);
        ins=[NSMutableArray arrayWithObject:wI0];
        idx=[NSMutableArray arrayWithObject:@0];
        if(nin==2){
            wI1=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),sI1);
            [ins addObject:wI1]; [idx addObject:@1];
        }
        req=((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            ins,idx,@[wO],@[@0],nil,nil,@0);
        if(!wI0 || (nin==2 && !wI1) || !wO || !req){ printf("  %-40s request construction failed\n",label); goto done; }

        e=nil;
        if(!((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl,
              @selector(evaluateWithQoS:options:request:error:),21,@{},req,&e)){
            printf("  %-40s EVAL FAIL: %s\n",label,e?[[e description]UTF8String]:"(no error)"); goto done;
        }

        _Float16 *out=malloc(bytes);
        if(!out){ printf("  %-40s output alloc failed\n",label); goto done; }
        if(!readSurface(sO,out,elems)){
            printf("  %-40s IOSurface read failed\n",label); free(out); goto done;
        }
        double maxd=0;
        size_t nonfinite=0;
        for(size_t i=0;i<elems;i++){
            double expect;
            if(golden==G_DOUBLE) expect=2.0*(double)x[i];
            else if(golden==G_SUM) expect=(double)x[i]+(double)y[i];
            else {
                // conv(sum, W): out[co,s] = sum_ci W[co,ci]*(x[ci,s]+y[ci,s])
                size_t co=i/SP, s=i%SP;
                double sum=0;
                for(int ci=0;ci<CH;ci++)
                    sum += (double)w[co*CH+ci]*((double)x[ci*SP+s]+(double)y[ci*SP+s]);
                expect=sum;
            }
            double got=(double)out[i];
            if(!isfinite(got) || !isfinite(expect)){ nonfinite++; continue; }
            double d=fabs(got-expect);
            if(d>maxd) maxd=d;
        }
        free(out);
        if(nonfinite){
            maxd=NAN;   // sentinel: never satisfy a <= tolerance test
            printf("  %-40s MISMATCH (non-finite observed/reference: %zu elems)\n",label,nonfinite);
        } else if(maxd<=TOL){
            printf("  %-40s PASS (max |ANE - CPU| = %.4g)\n",label,maxd);
            rc=0;
        } else {
            printf("  %-40s MISMATCH (max |ANE - CPU| = %.4g)\n",label,maxd);
        }

    done:
        if(mdl){
            e=nil;
            if(!((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl,@selector(unloadWithQoS:error:),21,&e)){
                printf("  %-40s unload FAIL: %s\n",label,e?[[e description]UTF8String]:"(no error)");
                rc=1;
            }
        }
        if(sI0) CFRelease(sI0);
        if(sI1) CFRelease(sI1);
        if(sO) CFRelease(sO);
        if(td) [fm removeItemAtPath:td error:nil];
        return rc;
    }
}

int main(void){
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",RTLD_NOW);
    const size_t elems=(size_t)CH*SP, wElems=(size_t)CH*CH;
    _Float16 *x=malloc(elems*sizeof(_Float16));
    _Float16 *y=malloc(elems*sizeof(_Float16));
    _Float16 *w=malloc(wElems*sizeof(_Float16));
    if(!x || !y || !w){ printf("host allocation failed\n"); free(x); free(y); free(w); return 1; }
    for(size_t i=0;i<elems;i++){
        x[i]=(_Float16)(((int)((i*13)%9)-4)*0.25f);   // [-1, 1]
        y[i]=(_Float16)(((int)((i*29)%5)-2)*0.25f);   // [-0.5, 0.5]
    }
    for(size_t i=0;i<wElems;i++)
        w[i]=(_Float16)(((int)((i*7)%5)-2)*0.125f);   // [-0.25, 0.25]

    printf("=== multi-input request probe with CPU goldens ===\n");
    int fails=0;
    fails += tryCase("1-input, no weights, nil weightsArg", 1, NO,  YES, G_NONE,   x,y,w,elems);
    fails += tryCase("1-input, no weights, {} weightsArg",  1, NO,  NO,  G_DOUBLE, x,y,w,elems);
    fails += tryCase("2-input, no weights, {} weightsArg",  2, NO,  NO,  G_SUM,    x,y,w,elems);
    fails += tryCase("2-input + BLOBFILE weights (conv)",   2, YES, NO,  G_CONV,   x,y,w,elems);

    free(x); free(y); free(w);
    printf("\n%s (%d failing case%s)\n", fails?"FAIL":"PASS", fails, fails==1?"":"s");
    return fails?1:0;
}
