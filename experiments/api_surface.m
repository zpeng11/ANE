// api_surface.m — dump the full private-API surface that is usable for inference.
// Enumerates methods on the descriptor/model/request classes and device-info props,
// so we can see exactly which knobs exist (memory pool, perf stats, late latch, AOT cache...).
// Build: xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl -o api_surface api_surface.m
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>

static void dumpClass(const char *name){
    Class c = NSClassFromString([NSString stringWithUTF8String:name]);
    if(!c){ printf("\n### %s : NOT FOUND\n", name); return; }
    printf("\n### %s  (super=%s)\n", name, class_getName(class_getSuperclass(c)));
    unsigned int n=0;
    Method *ms = class_copyMethodList(c,&n);
    printf("  instance methods (%u):\n", n);
    for(unsigned i=0;i<n;i++){
        const char *sn = sel_getName(method_getName(ms[i]));
        if(sn[0]=='.' || strcmp(sn,"dealloc")==0) continue;
        printf("    - %s   [%s]\n", sn, method_getTypeEncoding(ms[i]));
    }
    free(ms);
    Method *cm = class_copyMethodList(object_getClass(c), &n);
    printf("  class methods (%u):\n", n);
    for(unsigned i=0;i<n;i++){
        const char *sn = sel_getName(method_getName(cm[i]));
        printf("    + %s   [%s]\n", sn, method_getTypeEncoding(cm[i]));
    }
    free(cm);
}

int main(int argc, char **argv){
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",RTLD_NOW);
    if(argc>1){
        dumpClass(argv[1]);
        fflush(stdout);
        return 0;
    }
    printf("=== AppleNeuralEngine private API surface (inference-relevant) ===\n");
    dumpClass("_ANEInMemoryModelDescriptor");
    dumpClass("_ANEInMemoryModel");
    dumpClass("_ANERequest");
    dumpClass("_ANEIOSurfaceObject");
    dumpClass("_ANEProgramIOSurfacesMapper");
    dumpClass("_ANEDeviceInfo");

    printf("\n=== _ANEDeviceInfo values ===\n");
    Class DI = NSClassFromString(@"_ANEDeviceInfo");
    if(DI){
        const char *props[] = {"aneSubType","aneArchitectureType","numANECores","isVirtualMachine",
            "productName","boardType","isANECompileSupported","aneBoardType","hasANE","aneVersion",
            "isPrecompiledModelSupported","supportsFloat16","supportsInt8", NULL};
        for(int i=0; props[i]; i++){
            SEL s = NSSelectorFromString([NSString stringWithUTF8String:props[i]]);
            if([DI respondsToSelector:s]){
                id v = ((id(*)(Class,SEL))objc_msgSend)(DI, s);
                printf("  %-30s = %s\n", props[i], v?[[v description] UTF8String]:"(nil)");
            } else printf("  %-30s = (no such selector)\n", props[i]);
        }
    }
    return 0;
}
