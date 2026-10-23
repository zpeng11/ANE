// io_dtype_bench.m — fp32-with-cast vs fp16-direct I/O on the same conv chain.
// Scope note: this chain is compute-bound, so I/O dtype is expected to have little
// effect here. Use an I/O-bound shape to study dtype sensitivity; do not treat this
// configuration as a general rule.
//
// PERFORMANCE-ONLY probe: weights are synthetic and deterministic; outputs are not
// validated. A failed eval aborts the run instead of being reported as a timing.
//
// Build: xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl -o io_dtype_bench io_dtype_bench.m
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>
#import <math.h>
#import <string.h>
#import "ane_probe_common.h"

// ---- v2 BLOBFILE container (see ane_probe_common.h) -------------------------

static uint32_t g_rng = 0x9E3779B9u;
static uint32_t rngNext(void){ g_rng = g_rng*1664525u + 1013904223u; return g_rng; }

static NSData *blobFP16(int ch,int depth){
    NSUInteger wsize=(NSUInteger)ch*ch*2, cs=ane_probe_chunk_span(wsize), total=64+cs*(NSUInteger)depth;
    uint8_t *b=calloc(total,1); if(!b) return nil;
    ane_probe_write_file_header(b, (uint32_t)depth);   // one fp16 chunk per layer
    float scale=sqrtf(3.0f/(float)ch);
    for(int i=0;i<depth;i++){
        NSUInteger off=64+(NSUInteger)i*cs;
        ane_probe_write_chunk_header(b+off, ANE_PROBE_BLOB_FP16, (uint64_t)wsize, (uint64_t)(off+64));
        _Float16*d=(_Float16*)(b+off+64);
        for(NSUInteger j=0;j<wsize/2;j++){
            float u=(float)((rngNext()>>8)&0xFFFFu)/65535.0f;
            d[j]=(_Float16)((2.0f*u-1.0f)*scale);
        }
    }
    return [NSData dataWithBytesNoCopy:b length:total freeWhenDone:YES];
}

// fp32io=YES -> declare tensor<fp32,...> I/O and cast inside (inmem_peak style)
// fp32io=NO  -> declare tensor<fp16,...> I/O straight through (ane_int8_bench style)
static NSString *genMIL(int ch,int sp,int depth,BOOL fp32io){
    NSMutableString *m=[NSMutableString string];
    [m appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<%s, [1, %d, 1, %d]> x) {\n", fp32io?"fp32":"fp16", ch, sp];
    [m appendString:@"            string c_pad_type_0 = const()[name = string(\"c_pad_type_0\"), val = string(\"valid\")];\n"
        @"            tensor<int32, [2]> c_strides_0 = const()[name = string(\"c_strides_0\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"            tensor<int32, [4]> c_pad_0 = const()[name = string(\"c_pad_0\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"            tensor<int32, [2]> c_dilations_0 = const()[name = string(\"c_dilations_0\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"            int32 c_groups_0 = const()[name = string(\"c_groups_0\"), val = int32(1)];\n"];
    NSString *prev;
    if(fp32io){
        [m appendString:@"            string xf16 = const()[name = string(\"xf16\"), val = string(\"fp16\")];\n"];
        [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> x0 = cast(dtype = xf16, x = x)[name = string(\"cast_in\")];\n", ch, sp];
        prev=@"x0";
    } else prev=@"x";
    NSUInteger cs=ane_probe_chunk_span((NSUInteger)ch*ch*2);
    for(int i=0;i<depth;i++){
        [m appendFormat:@"            tensor<fp16, [%d, %d, 1, 1]> W%d = const()[name = string(\"W%d\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n",
            ch,ch,i,i,ch,ch,(unsigned long)(64+i*cs)];
        NSString *o=[NSString stringWithFormat:@"c%d",i];
        [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> %@ = conv(dilations = c_dilations_0, groups = c_groups_0, pad = c_pad_0, pad_type = c_pad_type_0, strides = c_strides_0, weight = W%d, x = %@)[name = string(\"%@\")];\n",ch,sp,o,i,prev,o];
        prev=o;
    }
    if(fp32io){
        [m appendString:@"            string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"];
        [m appendFormat:@"            tensor<fp32, [1, %d, 1, %d]> c = cast(dtype = to_fp32, x = %@)[name = string(\"cast_out\")];\n", ch, sp, prev];
        [m appendString:@"        } -> (c);\n}\n"];
    } else {
        [m appendFormat:@"        } -> (%@);\n}\n", prev];
    }
    return m;
}

static double bench(int ch,int sp,int depth,BOOL fp32io){
    @autoreleasepool{
        NSError *e=nil;
        NSData *milData=[[genMIL(ch,sp,depth,fp32io) dataUsingEncoding:NSUTF8StringEncoding] copy];
        NSData *wb=blobFP16(ch,depth);
        if(!wb)return -1;
        Class D=NSClassFromString(@"_ANEInMemoryModelDescriptor"),I=NSClassFromString(@"_ANEInMemoryModel");
        Class AR=NSClassFromString(@"_ANERequest"),AIO=NSClassFromString(@"_ANEIOSurfaceObject");
        id desc=((id(*)(Class,SEL,id,id,id))objc_msgSend)(D,@selector(modelWithMILText:weights:optionsPlist:),milData,@{@"@model_path/weights/weight.bin":@{@"offset":@0,@"data":wb}},nil);
        if(!desc)return -1;
        id mdl=((id(*)(Class,SEL,id))objc_msgSend)(I,@selector(inMemoryModelWithDescriptor:),desc);
        id hx=((id(*)(id,SEL))objc_msgSend)(mdl,@selector(hexStringIdentifier));
        NSString *td=[NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm=[NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [wb writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
        if(!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(compileWithQoS:options:error:),21,@{},&e)){[fm removeItemAtPath:td error:nil];return -3;}
        if(!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(loadWithQoS:options:error:),21,@{},&e)){[fm removeItemAtPath:td error:nil];return -4;}
        NSUInteger bytes=(NSUInteger)ch*sp*(fp32io?4:2);
        IOSurfaceRef ioI=IOSurfaceCreate((__bridge CFDictionaryRef)@{(id)kIOSurfaceWidth:@(bytes),(id)kIOSurfaceHeight:@1,(id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(bytes),(id)kIOSurfaceAllocSize:@(bytes),(id)kIOSurfacePixelFormat:@0});
        IOSurfaceRef ioO=IOSurfaceCreate((__bridge CFDictionaryRef)@{(id)kIOSurfaceWidth:@(bytes),(id)kIOSurfaceHeight:@1,(id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(bytes),(id)kIOSurfaceAllocSize:@(bytes),(id)kIOSurfacePixelFormat:@0});
        if(!ioI || !ioO){
            if(ioI)CFRelease(ioI); if(ioO)CFRelease(ioO);
            [fm removeItemAtPath:td error:nil]; return -5;
        }
        id wI=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),ioI);
        id wO=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),ioO);
        id req=((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(AR,@selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),@[wI],@[@0],@[wO],@[@0],nil,nil,@0);
        BOOL ok=YES;
        for(int i=0;i<10 && ok;i++)
            ok=((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl,@selector(evaluateWithQoS:options:request:error:),21,@{},req,&e);
        int it=100; uint64_t t0=0;
        if(ok){
            t0=mach_absolute_time();
            for(int i=0;i<it;i++){
                if(!((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl,@selector(evaluateWithQoS:options:request:error:),21,@{},req,&e)){ok=NO;break;}
            }
        }
        NSString *evalErr = ok ? nil : (e ? [e description] : nil);
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl,@selector(unloadWithQoS:error:),21,&e);
        CFRelease(ioI);CFRelease(ioO);[fm removeItemAtPath:td error:nil];
        if(!ok){printf("  eval FAIL: %s\n",evalErr?[evalErr UTF8String]:"?");return -6;}
        return ane_probe_ticks_to_ms(mach_absolute_time()-t0)/it;
    }
}

int main(void){

    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",RTLD_NOW);
    printf("=== fp32 vs fp16 direct I/O (performance-only; 128x conv 512ch sp64) ===\n\n");
    int ch=512,sp=64,depth=128;
    printf("%-22s %10s %10s\n","I/O dtype","ms/eval","vs fp32");
    double m32=bench(ch,sp,depth,YES);
    double m16=bench(ch,sp,depth,NO);
    if(m32>0) printf("%-22s %8.4f ms %9s\n","fp32 in/out (cast)",m32,"1.000x");
    else      printf("%-22s FAIL(%.0f)\n","fp32 in/out (cast)",m32);
    if(m16>0){
        if(m32>0) printf("%-22s %8.4f ms %8.3fx\n","fp16 in/out",m16,m32/m16);
        else      printf("%-22s %8.4f ms %9s\n","fp16 in/out",m16,"-");
    } else printf("%-22s FAIL(%.0f)\n","fp16 in/out",m16);
    return (m32>0 && m16>0) ? 0 : 1;
}
