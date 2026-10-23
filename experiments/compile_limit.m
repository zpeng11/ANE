// compile_limit.m — diagnose per-process model build/compile/load resource behavior.
// Each iteration builds, compiles, loads, evaluates, and (without --retain) unloads
// one small model. --retain keeps models resident so you can observe where load
// eventually fails. This is a diagnostic for the current process/build; it does not
// assert a fixed compile or resident-model limit.
//
// This is intentionally expensive (compiles up to N models). Do not use it as a
// routine smoke test; the bounded probes are sustained.m / cpu_gemm_bench.
//
// Build: xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl -o compile_limit compile_limit.m
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <math.h>
#import <string.h>

// ---- v2 BLOBFILE container (64-byte chunk header, little-endian) -------------
//   +0 magic 0xDEADBEEF, +4 uint32 dtype (fp16=1, int8=4),
//   +8 uint64 payload bytes, +16 uint64 absolute payload offset.
// The 64-byte file header is: +0 uint32 chunk_count, +4 uint32 version (2).
static const uint32_t kBLOBDtypeFP16 = 1;

static NSUInteger align64(NSUInteger n) { return (n + 63u) & ~(NSUInteger)63u; }

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

static NSData *buildWeightBlob(int ch){
    NSUInteger payload = (NSUInteger)ch*ch*2;
    NSUInteger chunkSize = 64 + align64(payload), total = 64 + chunkSize;
    uint8_t *buf = calloc(total,1);
    if(!buf) return nil;
    writeFileHeader(buf, 1u);   // single weight chunk
    writeChunkHeader(buf + 64, kBLOBDtypeFP16, (uint64_t)payload, 128);
    _Float16 *f=(_Float16*)(buf+128);
    float scale = sqrtf(3.0f/(float)ch);
    for (NSUInteger j=0;j<payload/2;j++){
        float u=(float)((rngNext()>>8)&0xFFFFu)/65535.0f;
        f[j]=(_Float16)((2.0f*u-1.0f)*scale);
    }
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

static NSString *genMIL(int ch,int sp){
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n"
         "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
         "            string c_pad_type_0 = const()[name = string(\"c_pad_type_0\"), val = string(\"valid\")];\n"
         "            tensor<int32, [2]> c_strides_0 = const()[name = string(\"c_strides_0\"), val = tensor<int32, [2]>([1, 1])];\n"
         "            tensor<int32, [4]> c_pad_0 = const()[name = string(\"c_pad_0\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
         "            tensor<int32, [2]> c_dilations_0 = const()[name = string(\"c_dilations_0\"), val = tensor<int32, [2]>([1, 1])];\n"
         "            int32 c_groups_0 = const()[name = string(\"c_groups_0\"), val = int32(1)];\n"
         "            string x_to_fp16_dtype_0 = const()[name = string(\"x_to_fp16_dtype_0\"), val = string(\"fp16\")];\n"
         "            tensor<fp16, [1, %d, 1, %d]> xf = cast(dtype = x_to_fp16_dtype_0, x = x)[name = string(\"cast_in\")];\n"
         "            tensor<fp16, [%d, %d, 1, 1]> W0 = const()[name = string(\"W0\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
         "            tensor<fp16, [1, %d, 1, %d]> c0 = conv(dilations = c_dilations_0, groups = c_groups_0, pad = c_pad_0, pad_type = c_pad_type_0, strides = c_strides_0, weight = W0, x = xf)[name = string(\"c0\")];\n"
         "            string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
         "            tensor<fp32, [1, %d, 1, %d]> c = cast(dtype = to_fp32, x = c0)[name = string(\"cast_out\")];\n"
         "        } -> (c);\n}\n", ch, sp, ch, sp, ch, ch, ch, ch, ch, sp, ch, sp];
}

int main(int argc, char **argv) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",RTLD_NOW);
    int maxTry = argc>1?atoi(argv[1]):400;
    BOOL retainModels = (argc>2 && strcmp(argv[2],"--retain")==0);
    Class D=NSClassFromString(@"_ANEInMemoryModelDescriptor"), I=NSClassFromString(@"_ANEInMemoryModel");
    Class AR=NSClassFromString(@"_ANERequest"), AIO=NSClassFromString(@"_ANEIOSurfaceObject");
    printf("=== ANE per-process compile+load+eval probe (max %d, retain=%s) ===\n", maxTry, retainModels?"YES":"no");
    NSMutableArray *keep = [NSMutableArray array];   // retain models to emulate leak
    int ok=0;
    for (int i=0;i<maxTry;i++){
        @autoreleasepool {
            NSError *e=nil;
            NSData *mil=[[genMIL(512,64) dataUsingEncoding:NSUTF8StringEncoding] copy];
            NSData *wb=buildWeightBlob(512);
            if(!wb){ printf("iter %d: weight blob alloc FAIL\n",i); break; }
            id desc=((id(*)(Class,SEL,id,id,id))objc_msgSend)(D,@selector(modelWithMILText:weights:optionsPlist:),mil,@{@"@model_path/weights/weight.bin":@{@"offset":@0,@"data":wb}},nil);
            if(!desc){ printf("iter %d: descriptor nil\n",i); break; }
            id mdl=((id(*)(Class,SEL,id))objc_msgSend)(I,@selector(inMemoryModelWithDescriptor:),desc);
            id hx=((id(*)(id,SEL))objc_msgSend)(mdl,@selector(hexStringIdentifier));
            NSString *td=[NSTemporaryDirectory() stringByAppendingPathComponent:hx];
            NSFileManager *fm=[NSFileManager defaultManager];
            [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
            [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
            [wb writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
            BOOL cok=((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(compileWithQoS:options:error:),21,@{},&e);
            if(!cok){ printf("FAIL at compile #%d\n  %s\n", i, e?[[e description]UTF8String]:"?"); [fm removeItemAtPath:td error:nil]; break; }
            BOOL lok=((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(loadWithQoS:options:error:),21,@{},&e);
            if(!lok){ printf("FAIL at load #%d\n  %s\n", i, e?[[e description]UTF8String]:"?"); [fm removeItemAtPath:td error:nil]; break; }
            NSUInteger bytes=512*64*4;
            IOSurfaceRef ioI=IOSurfaceCreate((__bridge CFDictionaryRef)@{(id)kIOSurfaceWidth:@(bytes),(id)kIOSurfaceHeight:@1,(id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(bytes),(id)kIOSurfaceAllocSize:@(bytes),(id)kIOSurfacePixelFormat:@0});
            IOSurfaceRef ioO=IOSurfaceCreate((__bridge CFDictionaryRef)@{(id)kIOSurfaceWidth:@(bytes),(id)kIOSurfaceHeight:@1,(id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(bytes),(id)kIOSurfaceAllocSize:@(bytes),(id)kIOSurfacePixelFormat:@0});
            if(!ioI || !ioO){
                if(ioI)CFRelease(ioI); if(ioO)CFRelease(ioO);
                printf("FAIL at surface alloc #%d\n", i);
                [fm removeItemAtPath:td error:nil]; break;
            }
            id wI=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),ioI);
            id wO=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),ioO);
            id req=((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(AR,@selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),@[wI],@[@0],@[wO],@[@0],nil,nil,@0);
            BOOL eok=((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl,@selector(evaluateWithQoS:options:request:error:),21,@{},req,&e);
            CFRelease(ioI); CFRelease(ioO);
            if(!eok){
                printf("FAIL at eval #%d\n  %s\n", i, e?[[e description]UTF8String]:"?");
                [fm removeItemAtPath:td error:nil]; break;
            }
            if (retainModels) [keep addObject:mdl];          // keep model resident (simulates leak)
            else if(!((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl,@selector(unloadWithQoS:error:),21,&e))
                printf("warning: unload FAIL at #%d: %s\n", i, e?[[e description]UTF8String]:"?");
            [fm removeItemAtPath:td error:nil];
            ok++;
            if (i==0 || (i+1)%25==0) { printf("c+l+e %d ok\n", i+1); fflush(stdout); }
        }
    }
    printf("RESULT: %d successful compiles before failure/limit\n", ok);
    return 0;
}
