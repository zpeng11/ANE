// cpuhog.c — real saturating CPU load (NEON FMA chain) for ANE contention tests.
// Runs until killed (default SIGTERM/SIGINT disposition); load_sensitivity.sh owns
// the lifetime. The volatile sink prevents the compiler from deleting the chain.
// Build: xcrun clang -O2 -o cpuhog cpuhog.c
#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>

// External linkage (not static) so the optimizer cannot drop the FMA chain and
// -Wunused-but-set-variable stays quiet.
volatile float cpuhog_sink;

int main(void){
    float32x4_t a=vdupq_n_f32(1.0000001f), b=vdupq_n_f32(0.9999999f);
    for(;;){
        for(int i=0;i<100000;i++){
            a=vfmaq_f32(a,b,vdupq_n_f32(1e-7f));
            b=vfmaq_f32(b,a,vdupq_n_f32(1e-7f));
        }
        cpuhog_sink=vaddvq_f32(a)+vaddvq_f32(b);
    }
    return 0;
}
