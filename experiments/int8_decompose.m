// int8_decompose.m — decompose the INT8 W8A8 speedup into its two ingredients.
// The repo attributes the speedup to int8 *activations* halving L2 SRAM traffic
// (quantize/dequantize between tiles), with int8 weights stored via
// constexpr_affine_dequantize. This isolates which change actually buys the speed.
//
//   (a) fp16 weights, fp16 activations            <- baseline
//   (b) int8 weights, fp16 activations            <- weight compression only
//   (c) fp16 weights, int8 activation cache       <- activation traffic only
//   (d) int8 weights, int8 activation cache       <- full W8A8
//
// PERFORMANCE-ONLY probe: weights are synthetic and deterministic; outputs are not
// validated. ms/eval and speedup ratios are meaningful, numerics are not. The sparse
// int8 pattern is chosen for stable per-layer gain, not to model real quantization.
//
// Build: xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl -o int8_decompose int8_decompose.m
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
static const float kInt8Scale = 0.125f;   // matches fp16(0x1p-3) in the MIL below

static uint32_t g_rng = 0x9E3779B9u;
static uint32_t rngNext(void){ g_rng = g_rng*1664525u + 1013904223u; return g_rng; }

// Sparse {-1,0,+1} pattern. With dequant scale 0.125, choosing p = 1/(scale^2*ch)
// gives std(w_dequant) = 1/sqrt(ch), i.e. unit gain through the 1x1 conv chain.
static int8_t rngInt8Weight(int ch){
    float p = 1.0f/(kInt8Scale*kInt8Scale*(float)ch);
    if(p >= 1.0f) return (rngNext() & 1u) ? 1 : -1;   // dense +/-1 for small ch
    // UINT32_MAX is not exactly representable as float; stay below 2^32.
    uint32_t threshold = (uint32_t)(p * 4294967040.0f);
    if(rngNext() < threshold) return (rngNext() & 1u) ? 1 : -1;
    return 0;
}

static NSData *blobInt8(int ch,int depth){
    NSUInteger wsize=(NSUInteger)ch*ch, cs=ane_probe_chunk_span(wsize), total=64+cs*(NSUInteger)depth;
    uint8_t *b=calloc(total,1); if(!b) return nil;
    ane_probe_write_file_header(b, (uint32_t)depth);   // one int8 chunk per layer
    for(int i=0;i<depth;i++){
        NSUInteger off=64+(NSUInteger)i*cs;
        ane_probe_write_chunk_header(b+off, ANE_PROBE_BLOB_INT8, (uint64_t)wsize, (uint64_t)(off+64));
        int8_t*d=(int8_t*)(b+off+64);
        for(NSUInteger j=0;j<wsize;j++) d[j]=rngInt8Weight(ch);
    }
    return [NSData dataWithBytesNoCopy:b length:total freeWhenDone:YES];
}
static NSData *blobFP16(int ch,int depth){
    NSUInteger wsize=(NSUInteger)ch*ch*2, cs=ane_probe_chunk_span(wsize), total=64+cs*(NSUInteger)depth;
    uint8_t *b=calloc(total,1); if(!b) return nil;
    ane_probe_write_file_header(b, (uint32_t)depth);   // one fp16 chunk per layer
    float scale=sqrtf(3.0f/(float)ch);
    for(int i=0;i<depth;i++){
        NSUInteger off=64+(NSUInteger)i*cs;
        ane_probe_write_chunk_header(b+off, ANE_PROBE_BLOB_FP16, (uint64_t)wsize, (uint64_t)(off+64));
        _Float16*d=(_Float16*)(b+off+64);
        for(NSUInteger j=0;j<(NSUInteger)ch*ch;j++){
            float u=(float)((rngNext()>>8)&0xFFFFu)/65535.0f;
            d[j]=(_Float16)((2.0f*u-1.0f)*scale);
        }
    }
    return [NSData dataWithBytesNoCopy:b length:total freeWhenDone:YES];
}

static NSString *genMIL(int ch,int sp,int depth,BOOL iw,BOOL aq){
    NSMutableString *m=[NSMutableString string];
    [m appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, %d, %d]> x) {\n",ch,sp,sp];
    [m appendString:@"        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        @"        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"];
    if (aq) [m appendString:@"        fp16 q_scale = const()[name = string(\"q_scale\"), val = fp16(0x1p-3)];\n"
                        @"        string q_dtype = const()[name = string(\"q_dtype\"), val = string(\"int8\")];\n"
                        @"        fp16 dq_scale = const()[name = string(\"dq_scale\"), val = fp16(0x1p-3)];\n"];
    NSUInteger cs = iw ? ane_probe_chunk_span((NSUInteger)ch*ch) : ane_probe_chunk_span((NSUInteger)ch*ch*2);
    NSString *prev=@"x";
    for(int i=0;i<depth;i++){
        if(iw){
            [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> W%d = constexpr_affine_dequantize()[axis = int32(0), name = string(\"W%d\"), quantized_data = tensor<int8, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu))), scale = fp16(0x1p-3), zero_point = int8(0)];\n",
                ch,ch,i,i,ch,ch,(unsigned long)(64+i*cs)];
        } else {
            [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> W%d = const()[name = string(\"W%d\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n",
                ch,ch,i,i,ch,ch,(unsigned long)(64+i*cs)];
        }
        NSString *co=[NSString stringWithFormat:@"c%d",i];
        [m appendFormat:@"        tensor<fp16, [1, %d, %d, %d]> %@ = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W%d, x = %@)[name = string(\"%@\")];\n",ch,sp,sp,co,i,prev,co];
        if(aq && i<depth-1){
            NSString *q=[NSString stringWithFormat:@"q%d",i], *dq=[NSString stringWithFormat:@"dq%d",i];
            [m appendFormat:@"        tensor<int8, [1, %d, %d, %d]> %@ = quantize(input = %@, output_dtype = q_dtype, scale = q_scale)[name = string(\"%@\")];\n",ch,sp,sp,q,co,q];
            [m appendFormat:@"        tensor<fp16, [1, %d, %d, %d]> %@ = dequantize(input = %@, scale = dq_scale)[name = string(\"%@\")];\n",ch,sp,sp,dq,q,dq];
            prev=dq;
        } else prev=co;
    }
    [m appendFormat:@"    } -> (%@);\n}\n",prev];
    return m;
}

static double bench(int ch,int sp,int depth,BOOL iw,BOOL aq,const char*lbl){
    @autoreleasepool{
        NSError *e=nil;
        NSData *milData=[[genMIL(ch,sp,depth,iw,aq) dataUsingEncoding:NSUTF8StringEncoding] copy];
        NSData *wb= iw?blobInt8(ch,depth):blobFP16(ch,depth);
        if(!wb){printf("  %s: weight blob alloc FAIL\n",lbl);return -1;}
        Class D=NSClassFromString(@"_ANEInMemoryModelDescriptor"),I=NSClassFromString(@"_ANEInMemoryModel");
        Class AR=NSClassFromString(@"_ANERequest"),AIO=NSClassFromString(@"_ANEIOSurfaceObject");
        id desc=((id(*)(Class,SEL,id,id,id))objc_msgSend)(D,@selector(modelWithMILText:weights:optionsPlist:),milData,@{@"@model_path/weights/weight.bin":@{@"offset":@0,@"data":wb}},nil);
        if(!desc){printf("  %s: desc FAIL\n",lbl);return -1;}
        id mdl=((id(*)(Class,SEL,id))objc_msgSend)(I,@selector(inMemoryModelWithDescriptor:),desc);
        id hx=((id(*)(id,SEL))objc_msgSend)(mdl,@selector(hexStringIdentifier));
        NSString *td=[NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm=[NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [wb writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
        if(!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(compileWithQoS:options:error:),21,@{},&e)){printf("  %s: compile FAIL: %s\n",lbl,e?[[e description]UTF8String]:"?");[fm removeItemAtPath:td error:nil];return -3;}
        if(!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(loadWithQoS:options:error:),21,@{},&e)){printf("  %s: load FAIL\n",lbl);[fm removeItemAtPath:td error:nil];return -4;}
        NSUInteger bytes=(NSUInteger)ch*sp*sp*2;
        IOSurfaceRef ioI=IOSurfaceCreate((__bridge CFDictionaryRef)@{(id)kIOSurfaceWidth:@(bytes),(id)kIOSurfaceHeight:@1,(id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(bytes),(id)kIOSurfaceAllocSize:@(bytes),(id)kIOSurfacePixelFormat:@0});
        IOSurfaceRef ioO=IOSurfaceCreate((__bridge CFDictionaryRef)@{(id)kIOSurfaceWidth:@(bytes),(id)kIOSurfaceHeight:@1,(id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(bytes),(id)kIOSurfaceAllocSize:@(bytes),(id)kIOSurfacePixelFormat:@0});
        if(!ioI || !ioO){
            if(ioI)CFRelease(ioI); if(ioO)CFRelease(ioO);
            printf("  %s: surface alloc FAIL\n",lbl);
            [fm removeItemAtPath:td error:nil]; return -5;
        }
        id wI=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),ioI);
        id wO=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),ioO);
        id req=((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(AR,@selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),@[wI],@[@0],@[wO],@[@0],nil,nil,@0);
        BOOL ok=YES;
        for(int i=0;i<10 && ok;i++)
            ok=((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl,@selector(evaluateWithQoS:options:request:error:),21,@{},req,&e);
        int it=30; uint64_t t0=0;
        if(ok){
            t0=mach_absolute_time();
            for(int i=0;i<it;i++){
                if(!((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl,@selector(evaluateWithQoS:options:request:error:),21,@{},req,&e)){ok=NO;break;}
            }
        }
        NSString *evalErr = ok ? nil : (e ? [e description] : nil);
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl,@selector(unloadWithQoS:error:),21,&e);
        CFRelease(ioI);CFRelease(ioO);[fm removeItemAtPath:td error:nil];
        if(!ok){printf("  %s: eval FAIL: %s\n",lbl,evalErr?[evalErr UTF8String]:"?");return -6;}
        return ane_probe_ticks_to_ms(mach_absolute_time()-t0)/it;
    }
}

int main(void){

    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",RTLD_NOW);
    printf("=== INT8 W8A8 decomposition (performance-only; 128x conv 512ch 64x64) ===\n\n");
    int ch=512,sp=64,depth=128;
    double gop=2.0*ch*ch*sp*sp*depth/1e9;
    struct { BOOL iw, aq; const char *l; } v[] = {
        {NO ,NO ,"a) fp16 W + fp16 A"},
        {YES,NO ,"b) int8 W + fp16 A"},
        {NO ,YES,"c) fp16 W + int8 A"},
        {YES,YES,"d) int8 W + int8 A (W8A8)"},
    };
    printf("%-28s %10s %9s %9s\n","variant","ms/eval","TOPS","vs base");
    double base=0;
    for(int i=0;i<4;i++){
        double ms=bench(ch,sp,depth,v[i].iw,v[i].aq,v[i].l);
        if(ms>0){
            double t=gop/ms;
            if(i==0) base=ms;
            if(base>0) printf("%-28s %8.2f ms %8.2f %8.2fx\n",v[i].l,ms,t,base/ms);
            else       printf("%-28s %8.2f ms %8.2f %8s\n",v[i].l,ms,t,"-");
        }
        else printf("%-28s FAIL(%.0f)\n",v[i].l,ms);
        fflush(stdout);
    }
    return 0;
}
