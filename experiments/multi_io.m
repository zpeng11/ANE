// multi_io.m — two differently-shaped inputs, two outputs, with both
// inputs observably contributing and both outputs checked against a CPU golden.
//
//   a: [1,1,SP,IC]  fp16 activation
//   b: [1,1,IC,OC]  fp16 weight (different shape from a)
//   y = matmul(a, b)          -> [1,1,SP,OC]
//   z = add(x=y, y=y)         -> [1,1,SP,OC]
//
// This is the repo's dynamic-weight pattern: the weight matrix is a runtime
// input, not a compiled constant.  Neither input is decorative: y depends on a
// and b, z depends on y; the CPU golden is the corresponding 1x1/matrix product.
//
// All setup/compile/load/eval/unload results are checked.  Any failure (or a
// numeric mismatch) makes the process exit nonzero.  Surfaces are released and
// the temporary model directory is removed on every path.
//
// Build: xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//          -o /tmp/multi_io multi_io.m
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <string.h>
#import <math.h>
#import "ane_probe_common.h"

#define SP 64
#define IC 64
#define OC 96
#define TOL 1e-2

static NSString *genMIL(void){
    return [NSString stringWithFormat:
      @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n"
       "    func main<ios18>(tensor<fp16, [1, 1, %d, %d]> a, tensor<fp16, [1, 1, %d, %d]> b) {\n"
       "        bool bx = const()[name = string(\"bx\"), val = bool(false)];\n"
       "        bool by = const()[name = string(\"by\"), val = bool(false)];\n"
       "        tensor<fp16, [1, 1, %d, %d]> y = matmul(transpose_x = bx, transpose_y = by, x = a, y = b)[name = string(\"mm\")];\n"
       "        tensor<fp16, [1, 1, %d, %d]> z = add(x = y, y = y)[name = string(\"add\")];\n"
       "    } -> (y, z);\n}\n", SP, IC, IC, OC, SP, OC, SP, OC];
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

int main(void){
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",RTLD_NOW);
    @autoreleasepool {
        const size_t aElems=(size_t)SP*IC, bElems=(size_t)IC*OC, outElems=(size_t)SP*OC;
        NSError *e=nil;
        NSData *milData=nil;
        id desc=nil, mdl=nil, hx=nil, wA=nil, wB=nil, wY=nil, wZ=nil, req=nil;
        NSString *td=nil;
        Class D=NSClassFromString(@"_ANEInMemoryModelDescriptor"),I=NSClassFromString(@"_ANEInMemoryModel");
        Class AR=NSClassFromString(@"_ANERequest"),AIO=NSClassFromString(@"_ANEIOSurfaceObject");
        NSFileManager *fm=[NSFileManager defaultManager];
        IOSurfaceRef sA=NULL, sB=NULL, sY=NULL, sZ=NULL;
        size_t aA=0,aB=0,aY=0,aZ=0;
        _Float16 *a=malloc(aElems*sizeof(_Float16));
        _Float16 *b=malloc(bElems*sizeof(_Float16));
        _Float16 *y=malloc(outElems*sizeof(_Float16));
        _Float16 *z=malloc(outElems*sizeof(_Float16));
        float *gy=malloc(outElems*sizeof(float));
        int rc=1;

        if(!a || !b || !y || !z || !gy){ printf("host allocation failed\n"); goto done; }

        // Deterministic, exactly representable inputs/weights.
        for(size_t i=0;i<aElems;i++) a[i]=(_Float16)(((int)((i*13)%9)-4)*0.25f);   // [-1,1]
        for(size_t i=0;i<bElems;i++) b[i]=(_Float16)(((int)((i*7)%5)-2)*0.125f);   // [-0.25,0.25]

        // CPU golden: y[s,oc] = sum_ic a[s,ic]*b[ic,oc]; z = 2y.
        for(size_t s=0;s<SP;s++){
            for(size_t oc=0;oc<OC;oc++){
                float sum=0.f;
                for(size_t ic=0;ic<IC;ic++)
                    sum += (float)a[s*IC+ic]*(float)b[ic*OC+oc];
                gy[s*OC+oc]=sum;
            }
        }

        milData=[[genMIL() dataUsingEncoding:NSUTF8StringEncoding] copy];
        if(!milData){ printf("MIL allocation failed\n"); goto done; }

        desc=((id(*)(Class,SEL,id,id,id))objc_msgSend)(D,@selector(modelWithMILText:weights:optionsPlist:),milData,@{},nil);
        if(!desc){ printf("descriptor NIL\n"); goto done; }
        mdl=((id(*)(Class,SEL,id))objc_msgSend)(I,@selector(inMemoryModelWithDescriptor:),desc);
        if(!mdl){ printf("model NIL\n"); goto done; }
        hx=((id(*)(id,SEL))objc_msgSend)(mdl,@selector(hexStringIdentifier));
        td=[NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        if(![fm createDirectoryAtPath:td withIntermediateDirectories:YES attributes:nil error:nil]
           || ![milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES]){
            printf("temp setup failed\n"); goto done;
        }
        if(!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(compileWithQoS:options:error:),21,@{},&e)){
            printf("COMPILE FAIL: %s\n",e?[[e description]UTF8String]:"(no error)"); goto done;
        }
        if(!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(loadWithQoS:options:error:),21,@{},&e)){
            printf("LOAD FAIL: %s\n",e?[[e description]UTF8String]:"(no error)"); goto done;
        }
        printf("compiled+loaded: a[1,1,%d,%d] * b[1,1,%d,%d] -> (y,z)[1,1,%d,%d]\n",SP,IC,IC,OC,SP,OC);

        sA=ane_probe_surface_create(aElems*sizeof(_Float16));
        sB=ane_probe_surface_create(bElems*sizeof(_Float16));
        sY=ane_probe_surface_create(outElems*sizeof(_Float16));
        sZ=ane_probe_surface_create(outElems*sizeof(_Float16));
        if(sA) aA=IOSurfaceGetAllocSize(sA);
        if(sB) aB=IOSurfaceGetAllocSize(sB);
        if(sY) aY=IOSurfaceGetAllocSize(sY);
        if(sZ) aZ=IOSurfaceGetAllocSize(sZ);
        if(!sA || !sB || !sY || !sZ){ printf("IOSurface creation failed\n"); goto done; }
        printf("surfaces requested/alloc: a=%zu/%zu B b=%zu/%zu B y=%zu/%zu B z=%zu/%zu B\n",
               aElems*sizeof(_Float16),aA, bElems*sizeof(_Float16),aB,
               outElems*sizeof(_Float16),aY, outElems*sizeof(_Float16),aZ);
        if(!writeSurface(sA,a,aElems) || !writeSurface(sB,b,bElems)){
            printf("IOSurface write failed\n"); goto done;
        }

        wA=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),sA);
        wB=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),sB);
        wY=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),sY);
        wZ=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),sZ);
        req=((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wA,wB],@[@0,@1],@[wY,wZ],@[@0,@1],nil,nil,@0);
        if(!wA || !wB || !wY || !wZ || !req){ printf("request construction failed\n"); goto done; }

        e=nil;
        if(!((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl,
              @selector(evaluateWithQoS:options:request:error:),21,@{},req,&e)){
            printf("EVAL FAIL: %s\n",e?[[e description]UTF8String]:"(no error)"); goto done;
        }
        if(!readSurface(sY,y,outElems) || !readSurface(sZ,z,outElems)){
            printf("IOSurface read failed\n"); goto done;
        }

        double dy=0, dz=0;
        size_t nonfinite=0;
        for(size_t i=0;i<outElems;i++){
            double g=(double)gy[i], oy=(double)y[i], oz=(double)z[i];
            if(!isfinite(g) || !isfinite(oy) || !isfinite(oz)){ nonfinite++; continue; }
            double d1=fabs(oy-g);
            double d2=fabs(oz-2.0*g);
            if(d1>dy) dy=d1;
            if(d2>dz) dz=d2;
        }
        if(nonfinite){
            dy=dz=NAN;   // sentinel: never satisfy a <= tolerance test
            printf("outputs y/z: NON-FINITE observed/reference in %zu/%zu elems -> FAIL\n",
                   nonfinite,outElems);
            printf("numeric mismatch\n");
            goto done;
        }
        printf("output y: max |ANE - CPU| = %.4g  -> %s\n", dy, dy<=TOL?"PASS":"FAIL");
        printf("output z: max |ANE - CPU| = %.4g  -> %s\n", dz, dz<=TOL?"PASS":"FAIL");
        if(!(dy<=TOL) || !(dz<=TOL)){ printf("numeric mismatch\n"); goto done; }

        rc=0;
    done:
        if(mdl){
            e=nil;
            if(!((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl,@selector(unloadWithQoS:error:),21,&e)){
                printf("unload FAIL: %s\n",e?[[e description]UTF8String]:"(no error)");
                rc=1;
            }
        }
        if(sA) CFRelease(sA);
        if(sB) CFRelease(sB);
        if(sY) CFRelease(sY);
        if(sZ) CFRelease(sZ);
        if(td) [fm removeItemAtPath:td error:nil];
        free(a); free(b); free(y); free(z); free(gy);
        printf("%s\n", rc==0?"PASS":"FAIL");
        return rc;
    }
}
