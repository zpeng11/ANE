// sustained.m — run one ANE kernel config for many iterations, print per-block avg.
// Purpose: distinguish "first-run boost" from sustained-throughput on a fanless M4 Air.
//
// PERFORMANCE-ONLY probe: weights are synthetic and deterministic; output values are
// not validated. A failed eval aborts the run instead of being reported as a timing.
//
// Build: xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl -o sustained sustained.m
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>
#import <math.h>
#import <string.h>

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

// ---- v2 BLOBFILE container (64-byte chunk header, little-endian) -------------
//   +0 magic 0xDEADBEEF, +4 uint32 dtype (fp16=1, int8=4),
//   +8 uint64 payload bytes, +16 uint64 absolute payload offset.
// The 64-byte file header is: +0 uint32 chunk_count, +4 uint32 version (2).
static const uint32_t kBLOBDtypeFP16 = 1;

static NSUInteger align64(NSUInteger n) { return (n + 63u) & ~(NSUInteger)63u; }
static NSUInteger chunkSpan(NSUInteger payloadBytes) { return 64u + align64(payloadBytes); }

static void writeFileHeader(uint8_t *h, uint32_t chunkCount) {
    memset(h, 0, 64);
    memcpy(h + 0, &chunkCount, sizeof chunkCount);
    uint32_t version = 2;
    memcpy(h + 4, &version, sizeof version);
}

static void writeChunkHeader(uint8_t *h, uint32_t dtype, uint64_t bytes, uint64_t absOffset) {
    memset(h, 0, 64);
    h[0] = 0xEF; h[1] = 0xBE; h[2] = 0xAD; h[3] = 0xDE;
    memcpy(h + 4, &dtype, sizeof dtype);
    memcpy(h + 8, &bytes, sizeof bytes);
    memcpy(h + 16, &absOffset, sizeof absOffset);
}

static uint32_t g_rng = 0x9E3779B9u;
static uint32_t rngNext(void) { g_rng = g_rng * 1664525u + 1013904223u; return g_rng; }

static NSData *buildWeightBlob(int ch, int depth) {
    NSUInteger payload = (NSUInteger)ch * (NSUInteger)ch * 2;
    NSUInteger chunkSize = chunkSpan(payload);
    NSUInteger total = 64 + chunkSize * (NSUInteger)depth;
    uint8_t *buf = calloc(total, 1);
    if (!buf) return nil;
    writeFileHeader(buf, (uint32_t)depth);
    float scale = sqrtf(3.0f / (float)ch);   // unit-gain 1x1 conv: no explosion/underflow
    for (int i = 0; i < depth; i++) {
        NSUInteger off = 64 + (NSUInteger)i * chunkSize;
        writeChunkHeader(buf + off, kBLOBDtypeFP16, (uint64_t)payload, (uint64_t)(off + 64));
        _Float16 *f = (_Float16 *)(buf + off + 64);
        for (NSUInteger j = 0; j < payload / 2; j++) {
            float u = (float)((rngNext() >> 8) & 0xFFFFu) / 65535.0f;
            f[j] = (_Float16)((2.0f * u - 1.0f) * scale);
        }
    }
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}
static NSString *genMIL(int ch,int sp,int depth){
    NSMutableString *m=[NSMutableString string];
    [m appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n",ch,sp];
    [m appendString:@"            string c_pad_type_0 = const()[name = string(\"c_pad_type_0\"), val = string(\"valid\")];\n            tensor<int32, [2]> c_strides_0 = const()[name = string(\"c_strides_0\"), val = tensor<int32, [2]>([1, 1])];\n            tensor<int32, [4]> c_pad_0 = const()[name = string(\"c_pad_0\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n            tensor<int32, [2]> c_dilations_0 = const()[name = string(\"c_dilations_0\"), val = tensor<int32, [2]>([1, 1])];\n            int32 c_groups_0 = const()[name = string(\"c_groups_0\"), val = int32(1)];\n            string x_to_fp16_dtype_0 = const()[name = string(\"x_to_fp16_dtype_0\"), val = string(\"fp16\")];\n"];
    [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> x_to_fp16 = cast(dtype = x_to_fp16_dtype_0, x = x)[name = string(\"cast_in\")];\n",ch,sp];
    NSUInteger cs=chunkSpan((NSUInteger)ch*(NSUInteger)ch*2); NSString *prev=@"x_to_fp16";
    for(int i=0;i<depth;i++){
        [m appendFormat:@"            tensor<fp16, [%d, %d, 1, 1]> W%d = const()[name = string(\"W%d\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n",ch,ch,i,i,ch,ch,(unsigned long)(64+i*cs)];
        NSString *o=[NSString stringWithFormat:@"c%d",i];
        [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> %@ = conv(dilations = c_dilations_0, groups = c_groups_0, pad = c_pad_0, pad_type = c_pad_type_0, strides = c_strides_0, weight = W%d, x = %@)[name = string(\"%@\")];\n",ch,sp,o,i,prev,o];
        prev=o;
    }
    [m appendString:@"            string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"];
    [m appendFormat:@"            tensor<fp32, [1, %d, 1, %d]> c = cast(dtype = to_fp32, x = %@)[name = string(\"cast_out\")];\n",ch,sp,prev];
    [m appendString:@"        } -> (c);\n}\n"];
    return m;
}

int main(int argc,char**argv){
    mach_timebase_info(&g_tb);
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",RTLD_NOW);
    int ch=argc>1?atoi(argv[1]):512, sp=argc>2?atoi(argv[2]):64, depth=argc>3?atoi(argv[3]):128;
    int total=argc>4?atoi(argv[4]):600, block=argc>5?atoi(argv[5]):50;
    if(ch<=0||sp<=0||depth<=0||total<=0||block<=0||block>total){
        fprintf(stderr,"usage: %s [ch] [sp] [depth] [evals] [block] (positive; block <= evals)\n",argv[0]);
        return 1;
    }
    NSError *e=nil; NSData *milData=[[genMIL(ch,sp,depth) dataUsingEncoding:NSUTF8StringEncoding] copy];
    NSData *wb=buildWeightBlob(ch,depth);
    if(!wb){ fprintf(stderr,"weight blob alloc FAIL\n"); return 1; }
    Class D=NSClassFromString(@"_ANEInMemoryModelDescriptor"),I=NSClassFromString(@"_ANEInMemoryModel");
    Class AR=NSClassFromString(@"_ANERequest"),AIO=NSClassFromString(@"_ANEIOSurfaceObject");
    id desc=((id(*)(Class,SEL,id,id,id))objc_msgSend)(D,@selector(modelWithMILText:weights:optionsPlist:),milData,@{@"@model_path/weights/weight.bin":@{@"offset":@0,@"data":wb}},nil);
    if(!desc){ fprintf(stderr,"descriptor NIL%s%s\n", e?" : ":"", e?[[e description]UTF8String]:""); return 1; }
    id mdl=((id(*)(Class,SEL,id))objc_msgSend)(I,@selector(inMemoryModelWithDescriptor:),desc);
    id hx=((id(*)(id,SEL))objc_msgSend)(mdl,@selector(hexStringIdentifier));
    NSString *td=[NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm=[NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [wb writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
    if(!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(compileWithQoS:options:error:),21,@{},&e)){printf("compile FAIL %s\n",e?[[e description]UTF8String]:"?");[fm removeItemAtPath:td error:nil];return 1;}
    if(!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(loadWithQoS:options:error:),21,@{},&e)){printf("load FAIL %s\n",e?[[e description]UTF8String]:"?");[fm removeItemAtPath:td error:nil];return 1;}
    NSUInteger bytes=ch*sp*4;
    IOSurfaceRef ioI=IOSurfaceCreate((__bridge CFDictionaryRef)@{(id)kIOSurfaceWidth:@(bytes),(id)kIOSurfaceHeight:@1,(id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(bytes),(id)kIOSurfaceAllocSize:@(bytes),(id)kIOSurfacePixelFormat:@0});
    IOSurfaceRef ioO=IOSurfaceCreate((__bridge CFDictionaryRef)@{(id)kIOSurfaceWidth:@(bytes),(id)kIOSurfaceHeight:@1,(id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(bytes),(id)kIOSurfaceAllocSize:@(bytes),(id)kIOSurfacePixelFormat:@0});
    if(!ioI || !ioO){
        if(ioI)CFRelease(ioI); if(ioO)CFRelease(ioO);
        printf("surface alloc FAIL\n");[fm removeItemAtPath:td error:nil];return 1;
    }
    id wI=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),ioI);
    id wO=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),ioO);
    id req=((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(AR,@selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),@[wI],@[@0],@[wO],@[@0],nil,nil,@0);
    double gf=2.0*ch*ch*sp*depth/1e9;
    printf("=== sustained (performance-only): %dx conv %dch sp%d, %d evals, block=%d ===\n",depth,ch,sp,total,block);
    printf("%-14s %9s %8s\n","block","ms/eval","TFLOPS");
    for(int b=0;b<total/block;b++){
        uint64_t t0=mach_absolute_time();
        for(int i=0;i<block;i++){
            if(!((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl,@selector(evaluateWithQoS:options:request:error:),21,@{},req,&e)){
                printf("block %d: EVAL FAIL at iter %d: %s\n",b,i,e?[[e description]UTF8String]:"?");
                ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl,@selector(unloadWithQoS:error:),21,&e);
                CFRelease(ioI);CFRelease(ioO);[fm removeItemAtPath:td error:nil];
                return 2;
            }
        }
        double ms=ticksToMs(mach_absolute_time()-t0)/block;
        printf("%-14d %8.3f ms %7.2f\n",b,ms,gf/ms); fflush(stdout);
    }
    if(!((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl,@selector(unloadWithQoS:error:),21,&e))
        printf("warning: unload FAIL: %s\n",e?[[e description]UTF8String]:"?");
    CFRelease(ioI);CFRelease(ioO);[fm removeItemAtPath:td error:nil];
    return 0;
}
