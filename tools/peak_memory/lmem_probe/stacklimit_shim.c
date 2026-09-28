// LD_PRELOAD shim: report the max CUDA stack-size limit a process's context reached.
// The driver raises cudaLimitStackSize to the largest per-thread local frame of any
// kernel launched in the context and reserves (limit x maxThreadsPerSM x SMs) of
// device memory for it (lmem_probe.cu demonstrates this). The shim samples the limit
// after every synchronizing runtime call it wraps (the runtime is already unloading
// by the time a destructor runs) and writes the max at exit. No perf counters needed.
//   gcc -O2 -shared -fPIC stacklimit_shim.c -o libstacklimit.so -ldl
//   STACKLIMIT_OUT=/path.json LD_PRELOAD=./libstacklimit.so <cmd>
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
typedef int (*getlimit_t)(size_t*, int);
static size_t max_limit = 0;
static int samples = 0;
static void sample(void) {
    static getlimit_t f = NULL;
    if (!f) f = (getlimit_t)dlsym(RTLD_NEXT, "cudaDeviceGetLimit");
    size_t v = 0;
    if (f && f(&v, 0) == 0) { if (v > max_limit) max_limit = v; samples++; }
}
#define WRAP1(name, T1) \
    int name(T1 a) { static int (*r)(T1) = NULL; if (!r) r = dlsym(RTLD_NEXT, #name); \
                     int rc = r(a); sample(); return rc; }
#define WRAP0(name) \
    int name(void) { static int (*r)(void) = NULL; if (!r) r = dlsym(RTLD_NEXT, #name); \
                     int rc = r(); sample(); return rc; }
WRAP0(cudaDeviceSynchronize)
WRAP1(cudaStreamSynchronize, void*)
WRAP1(cudaEventSynchronize, void*)
WRAP1(cudaFree, void*)
int cudaMemcpy(void* d, const void* s, size_t n, int k) {
    static int (*r)(void*, const void*, size_t, int) = NULL;
    if (!r) r = dlsym(RTLD_NEXT, "cudaMemcpy");
    int rc = r(d, s, n, k); sample(); return rc;
}
static void __attribute__((destructor)) report(void) {
    const char* p = getenv("STACKLIMIT_OUT");
    FILE* fh = p ? fopen(p, "w") : stderr;
    if (!fh) return;
    fprintf(fh, "{\"max_stack_limit_bytes\": %zu, \"samples\": %d}\n", max_limit, samples);
    if (p) fclose(fh);
}
