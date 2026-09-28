// Local-memory (stack) reservation microbenchmark for the peak-memory study.
// Launches one kernel whose per-thread stack frame is ~STACK_BYTES (a runtime-
// indexed local array the compiler cannot promote to registers) and holds, so an
// external NVML sampler can read the resident footprint. The driver reserves
// local memory for the whole device when such a kernel is launched:
//   ~ per-thread frame x max resident threads/SM x SM count (never shrinks).
//   nvcc -O3 -cudart shared -arch=sm_90 -DSTACK_BYTES=4352 lmem_probe.cu -o lmem_4352
#include <cstdio>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>
#ifndef STACK_BYTES
#define STACK_BYTES 0
#endif
__global__ void k(int* out, int idx) {
#if STACK_BYTES > 0
    volatile int buf[STACK_BYTES / 4];
    for (int i = 0; i < STACK_BYTES / 4; ++i) buf[i] = i ^ idx;
    if (out) out[threadIdx.x] = buf[(idx + threadIdx.x) % (STACK_BYTES / 4)];
#else
    if (out) out[threadIdx.x] = idx;
#endif
}
int main() {
    cudaFree(0);
    int* d; cudaMalloc(&d, 1024 * sizeof(int));
    k<<<1, 256>>>(d, 3);
    cudaDeviceSynchronize();
    size_t lim = 0; cudaDeviceGetLimit(&lim, cudaLimitStackSize);
    cudaFuncAttributes a; cudaFuncGetAttributes(&a, k);
    int sms = 0, tps = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    cudaDeviceGetAttribute(&tps, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
    std::printf("{\"stack_bytes_req\":%d,\"localSizeBytes\":%zu,\"stack_limit\":%zu,\"sms\":%d,\"threads_per_sm\":%d}\n",
                STACK_BYTES, a.localSizeBytes, lim, sms, tps);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    return 0;
}
