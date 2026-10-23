// cpu_gemm_bench.m — measure Accelerate sgemm throughput on this CPU.
// The dynamic training loop does its dW gradients with cblas_sgemm; if the CPU is
// contended/slow, ms/step balloons.
//
// Unit equation: FLOPs = 2*m*n*k, so GFLOPS = 2*m*n*k / (ms * 1e6).
//
// The 280 GFLOPS denominator is a NEON-only FP32 reference for 4 P-cores, not a
// measured limit. Accelerate's sgemm may use the AMX matrix units, so efficiency
// >100% is expected.
//
// PERFORMANCE-ONLY probe: synthetic matrices, throughput only.
//
// Build: xcrun clang -O2 -framework Accelerate -o cpu_gemm_bench cpu_gemm_bench.m
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>
#include <Accelerate/Accelerate.h>

static mach_timebase_info_data_t tb;
static double ms_since(uint64_t t){ return (double)(mach_absolute_time()-t)*tb.numer/tb.denom/1e6; }

int main(void){
    mach_timebase_info(&tb);
    // sizes matching the training dW sgemms: [out, in] x [in, seq] with seq=256
    struct { int m, n, k; const char *label; } cases[] = {
        {2048, 256, 768, "W1/W3 dW  (2048x256 @ 768)"},
        {768,  256, 2048,"W2 dW     (768x256 @ 2048)"},
        {768,  256, 768, "Wq/Wk/Wv/Wo dW (768x256 @ 768)"},
        {2048, 2048, 2048,"square 2048"},
    };
    printf("=== Accelerate sgemm throughput ===\n");
    printf("%-30s %8s %10s %9s\n","case","ms","GFLOPS","eff/NEON");
    for (int c=0;c<4;c++){
        int m=cases[c].m,n=cases[c].n,k=cases[c].k;
        float *A=malloc((size_t)m*k*4), *B=malloc((size_t)k*n*4), *C=malloc((size_t)m*n*4);
        if(!A || !B || !C){ printf("%-30s alloc FAIL\n",cases[c].label); free(A);free(B);free(C); continue; }
        for (size_t i=0;i<(size_t)m*k;i++) A[i]=0.001f*(i%97);
        for (size_t i=0;i<(size_t)k*n;i++) B[i]=0.001f*(i%89);
        int it = 20;
        // warmup
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.0f,A,k,B,n,0.0f,C,n);
        uint64_t t0=mach_absolute_time();
        for(int i=0;i<it;i++)
            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.0f,A,k,B,n,0.0f,C,n);
        double ms=ms_since(t0)/it;
        double gflops = 2.0*m*n*k/(ms*1e6);
        printf("%-30s %7.3f  %8.1f  %8.1f\n",cases[c].label,ms,gflops,gflops/280.0*100);
        free(A);free(B);free(C);
    }
    return 0;
}
