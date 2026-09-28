// Empty-CUDA-context baseline for the peak-memory study.
//
// Creates the primary context (cudaFree(0)), launches one trivial kernel so the
// module is loaded, then holds for a few ms so the NVML sampler sees the steady
// resident footprint. Its NVML peak is the per-process floor every CUDA tool on
// this host/driver pays before allocating anything; the study reports it as a
// reference line, it is not subtracted from any arm.
//
//   nvcc -O2 -cudart shared -arch=sm_90 ctx_baseline.cu -o ctx_baseline
#include <cstdio>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>

__global__ void touch(int* p) { if (p) *p = 1; }

int main() {
    if (cudaFree(0) != cudaSuccess) { std::fprintf(stderr, "no CUDA context\n"); return 2; }
    touch<<<1, 1>>>(nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) { std::fprintf(stderr, "launch failed\n"); return 2; }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::printf("ctx_baseline ok\n");
    return 0;
}
