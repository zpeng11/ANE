#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
int main(void){
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",RTLD_NOW);
    Class DI = NSClassFromString(@"_ANEDeviceInfo");
    if(!DI){ printf("_ANEDeviceInfo not found\n"); return 1; }
    const char *sel_names[] = {"aneSubType","aneSubTypeVariant","aneSubTypeProductVariant","aneArchitectureType",
        "numANEs","numANECores","hasANE","productName","buildVersion","isVirtualMachine","aneBoardType",
        "isInternalBuild","isExcessivePowerDrainWhenIdle","precompiledModelChecksDisabled", NULL};
    for(int i=0; sel_names[i]; i++){
        SEL s = NSSelectorFromString([NSString stringWithUTF8String:sel_names[i]]);
        if([DI respondsToSelector:s]){
            const char *te = method_getTypeEncoding(class_getClassMethod(DI, s));
            if(te && (te[0]=='I'||te[0]=='q'||te[0]=='Q'||te[0]=='i'||te[0]=='B'))
                printf("  %-32s = %lld\n", sel_names[i], (long long)((long long(*)(Class,SEL))objc_msgSend)(DI,s));
            else { id v=((id(*)(Class,SEL))objc_msgSend)(DI,s); printf("  %-32s = %s\n", sel_names[i], v?[[v description]UTF8String]:"(nil)"); }
        } else printf("  %-32s = n/a\n", sel_names[i]);
    }
    return 0;
}
