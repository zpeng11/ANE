// static_shape_compile.m — static-shape compile+load cost on the ANE in-memory path.
//
// This probe uses only fully-specified static shapes.  It deliberately does not
// test symbolic/dynamic dimensions ("?" / "S" in the MIL text are not valid
// shapes for this compiler) and it makes no claim about dynamic-shape support.
// What it measures is how compile+load time scales with the spatial dimension
// for otherwise identical static programs.
//
// Each case builds a fresh MIL program plus a standard v2 FP16 weight blob with
// deterministic weights.  Setup, compile, load, and unload results are checked;
// a failing case is reported and makes the process exit nonzero.
//
// Build: xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//          -o /tmp/static_shape_compile static_shape_compile.m
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>
#import <string.h>
#import "ane_probe_common.h"

// Deterministic weights: dyadic values in [-0.5, 0.5], exactly representable.
static void fillWeights(_Float16 *w, NSUInteger n){
    for(NSUInteger i = 0; i < n; i++)
        w[i] = (_Float16)(((int)((i*37) % 17) - 8) * 0.0625f);
}

static NSString *genMIL(int ch, int sp, int depth){
    NSMutableString *m=[NSMutableString string];
    [m appendString:ANE_PROBE_MIL_HEADER];
    [m appendFormat:@"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n", ch, sp];
    [m appendString:@"            string c_pad_type_0 = const()[name = string(\"c_pad_type_0\"), val = string(\"valid\")];\n"
        @"            tensor<int32, [2]> c_strides_0 = const()[name = string(\"c_strides_0\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"            tensor<int32, [4]> c_pad_0 = const()[name = string(\"c_pad_0\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"            tensor<int32, [2]> c_dilations_0 = const()[name = string(\"c_dilations_0\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"            int32 c_groups_0 = const()[name = string(\"c_groups_0\"), val = int32(1)];\n"
        @"            string x_to_fp16_dtype_0 = const()[name = string(\"x_to_fp16_dtype_0\"), val = string(\"fp16\")];\n"];
    [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> xf = cast(dtype = x_to_fp16_dtype_0, x = x)[name = string(\"cast_in\")];\n", ch, sp];
    NSUInteger cs = 64 + (NSUInteger)ch*ch*2;
    NSString *prev=@"xf";
    for(int i=0;i<depth;i++){
        [m appendFormat:@"            tensor<fp16, [%d, %d, 1, 1]> W%d = const()[name = string(\"W%d\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n",
            ch,ch,i,i,ch,ch,(unsigned long)(64 + (NSUInteger)i*cs)];
        NSString *o=[NSString stringWithFormat:@"c%d",i];
        [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> %@ = conv(dilations = c_dilations_0, groups = c_groups_0, pad = c_pad_0, pad_type = c_pad_type_0, strides = c_strides_0, weight = W%d, x = %@)[name = string(\"%@\")];\n",
            ch,sp,o,i,prev,o];
        prev=o;
    }
    [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> c = cast(dtype = x_to_fp16_dtype_0, x = %@)[name = string(\"cast_out\")];\n", ch, sp, prev];
    [m appendString:@"        } -> (c);\n}\n"];
    return m;
}

// Compile+load one static shape.  Returns 0 on success, nonzero on any failure;
// on success *msOut gets the measured compile+load time.
static int tryBuild(NSString *milStr, int ch, int sp, int depth, double *msOut){
    *msOut = -1.0;
    @autoreleasepool{
        NSError *e=nil;
        NSData *milData=[milStr dataUsingEncoding:NSUTF8StringEncoding];
        NSUInteger elems=(NSUInteger)ch*ch;
        _Float16 *w=malloc(elems*(NSUInteger)depth*sizeof(_Float16));
        if(!w){ printf("  [1,%d,1,%d] alloc failed\n",ch,sp); return 1; }
        fillWeights(w, elems*(NSUInteger)depth);
        // depth weight matrices, matching the `depth` BLOBFILE offsets in genMIL.
        NSData *wb=ane_probe_blob_fp16(w, depth, elems);
        free(w);
        if(!wb){ printf("  [1,%d,1,%d] blob build failed\n",ch,sp); return 1; }

        Class D=NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class I=NSClassFromString(@"_ANEInMemoryModel");
        id desc=((id(*)(Class,SEL,id,id,id))objc_msgSend)(D,@selector(modelWithMILText:weights:optionsPlist:),
            milData, @{@"@model_path/weights/weight.bin":@{@"offset":@0,@"data":wb}}, nil);
        if(!desc){ printf("  [1,%d,1,%d] descriptor NIL\n",ch,sp); return 1; }
        id mdl=((id(*)(Class,SEL,id))objc_msgSend)(I,@selector(inMemoryModelWithDescriptor:),desc);
        if(!mdl){ printf("  [1,%d,1,%d] model NIL\n",ch,sp); return 1; }
        id hx=((id(*)(id,SEL))objc_msgSend)(mdl,@selector(hexStringIdentifier));
        NSString *td=[NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm=[NSFileManager defaultManager];
        NSString *wdir=[td stringByAppendingPathComponent:@"weights"];
        BOOL setup=[fm createDirectoryAtPath:wdir withIntermediateDirectories:YES attributes:nil error:nil]
                && [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES]
                && [wb writeToFile:[wdir stringByAppendingPathComponent:@"weight.bin"] atomically:YES];
        if(!setup){
            printf("  [1,%d,1,%d] temp setup failed\n",ch,sp);
            [fm removeItemAtPath:td error:nil];
            return 1;
        }

        uint64_t t0=mach_absolute_time();
        BOOL cok=((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(compileWithQoS:options:error:),21,@{},&e);
        if(!cok){
            printf("  [1,%d,1,%d] COMPILE FAIL: %s\n",ch,sp,e?[[e description]UTF8String]:"(no error)");
            [fm removeItemAtPath:td error:nil];
            return 1;
        }
        BOOL lok=((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,@selector(loadWithQoS:options:error:),21,@{},&e);
        *msOut=ane_probe_ticks_to_ms(mach_absolute_time()-t0);
        if(!lok){
            printf("  [1,%d,1,%d] LOAD FAIL: %s\n",ch,sp,e?[[e description]UTF8String]:"(no error)");
            [fm removeItemAtPath:td error:nil];
            return 1;
        }
        BOOL uok=((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl,@selector(unloadWithQoS:error:),21,&e);
        if(!uok){
            printf("  [1,%d,1,%d] unload FAIL: %s\n",ch,sp,e?[[e description]UTF8String]:"(no error)");
            [fm removeItemAtPath:td error:nil];
            return 1;
        }
        [fm removeItemAtPath:td error:nil];
        return 0;
    }
}

int main(void){
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",RTLD_NOW);

    const int ch=512, depth=8;
    const int sps[]={16,32,64,128,256,512,1024};
    const int ns=(int)(sizeof(sps)/sizeof(sps[0]));

    printf("=== static-shape compile+load cost (fresh MIL + v2 blob per case) ===\n");
    printf("channels=%d, depth=%d conv layers, deterministic weights\n\n", ch, depth);
    printf("  %-12s %14s %8s\n", "shape", "compile+load", "status");

    int fails=0, ok=0;
    for(int i=0;i<ns;i++){
        double ms=-1;
        if(tryBuild(genMIL(ch,sps[i],depth), ch, sps[i], depth, &ms)==0){
            printf("  [1,%d,1,%-4d] %9.1f ms %8s\n", ch, sps[i], ms, "OK");
            ok++;
        } else {
            printf("  [1,%d,1,%-4d] %9s %8s\n", ch, sps[i], "-", "FAIL");
            fails++;
        }
    }
    printf("\n%d/%d shapes compiled and loaded\n", ok, ns);
    return fails ? 1 : 0;
}
