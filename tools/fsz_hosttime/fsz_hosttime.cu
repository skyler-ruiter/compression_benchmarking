// Host-wall-clock benchmark for FSZ, to answer the question the shipped `fsz`
// tool cannot: how much time is spent outside the CUDA-event bracket.
//
// The stock CLI reports device time only (a cudaEvent pair around the kernel),
// which is what benchkit records as device_ms. FZGM reports BOTH a device
// figure (dag_elapsed_ms) and a "Host elapsed" wall time around the same calls,
// and the ratio between them has been badly misleading before (D33: FZGM's
// split-mode compress ran 3.53x host-over-device). Comparing FSZ's device_ms
// against FZGM's device_ms is therefore only half an answer.
//
// This measures, per repetition, std::chrono around
//   fsz::compress(...)  +  cudaStreamSynchronize(...)
// which is the same bracket FZGM's "Host elapsed" uses: launch cost, any
// host-side work the call does, and the wait for the device to finish. All
// allocation, the workspace, and the H2D copy are hoisted out, exactly as they
// are on the FZGM side. It also records a device event time inside the same
// iteration so host and device come from the SAME launch, not from two runs.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_90 fsz_hosttime.cu -o fsz_hosttime \
//        -I$HOME/compressors/FSZ/include -L$HOME/compressors/FSZ/build -lfsz
#include "fsz/fsz.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void check_cuda(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(e));
        std::exit(1);
    }
}

static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv) {
    const char* path = nullptr;
    std::string eb_mode = "rel";
    double eb_val = 1e-3;
    int reps = 10, warmup = 3;

    // -eb takes two tokens (abs|rel, value), matching the stock `fsz` CLI so the
    // adapter can pass the same argv shape to either binary.
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-i") && i + 1 < argc)         path = argv[++i];
        else if (!std::strcmp(argv[i], "-eb") && i + 2 < argc) {
            eb_mode = argv[++i];
            eb_val  = std::atof(argv[++i]);
        }
        else if (!std::strcmp(argv[i], "-r") && i + 1 < argc)    reps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-t") && i + 1 < argc) {
            if (std::strcmp(argv[i + 1], "f32") != 0) {
                std::fprintf(stderr, "fsz_hosttime: only -t f32 is supported\n");
                return 2;
            }
            ++i;
        }
        else if (!std::strcmp(argv[i], "-d")) {
            // dims are irrelevant here (FSZ tiles the flattened array); skip them
            while (i + 1 < argc && argv[i + 1][0] != '-') ++i;
        }
    }
    if (!path) {
        std::fprintf(stderr,
                     "usage: -i file.f32 [-t f32] [-d D1 D2 D3] "
                     "[-eb abs|rel V] [-r reps]\n");
        return 2;
    }
    if (eb_mode != "abs" && eb_mode != "rel") {
        std::fprintf(stderr, "fsz_hosttime: -eb mode must be abs or rel\n");
        return 2;
    }

    // ---- load ----
    std::FILE* fh = std::fopen(path, "rb");
    if (!fh) { std::fprintf(stderr, "cannot open %s\n", path); return 2; }
    std::fseek(fh, 0, SEEK_END);
    const std::size_t bytes = (std::size_t)std::ftell(fh);
    std::fseek(fh, 0, SEEK_SET);
    const std::size_t n = bytes / sizeof(float);
    std::vector<float> h_in(n);
    if (std::fread(h_in.data(), sizeof(float), n, fh) != n) {
        std::fprintf(stderr, "short read on %s\n", path); return 2;
    }
    std::fclose(fh);

    // rel bound is a fraction of (max - min), matching the fsz CLI and canonical rel_range
    float vmin = h_in[0], vmax = h_in[0];
    for (std::size_t i = 1; i < n; ++i) { vmin = std::min(vmin, h_in[i]); vmax = std::max(vmax, h_in[i]); }
    const float eb = (eb_mode == "abs")
                       ? (float)eb_val
                       : (float)(eb_val * ((double)vmax - (double)vmin));

    // ---- device setup, all hoisted out of the timed region ----
    float *d_in = nullptr, *d_out = nullptr;
    unsigned char* d_cmp = nullptr;
    check_cuda(cudaMalloc(&d_in,  n * sizeof(float)),            "alloc d_in");
    check_cuda(cudaMalloc(&d_out, n * sizeof(float)),            "alloc d_out");
    check_cuda(cudaMalloc(&d_cmp, fsz::max_compressed_bytes(n)), "alloc d_cmp");
    check_cuda(cudaMemcpy(d_in, h_in.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice),               "memcpy h2d");
    fsz::Workspace ws(n);

    cudaEvent_t e0, e1;
    check_cuda(cudaEventCreate(&e0), "eventcreate");
    check_cuda(cudaEventCreate(&e1), "eventcreate");

    fsz_compress_result_t cres{};
    for (int i = 0; i < warmup; ++i) {
        cres = fsz::compress(d_in, d_cmp, n, eb, ws);
        fsz::decompress(d_out, d_cmp, n, eb, cres, ws);
    }
    check_cuda(cudaDeviceSynchronize(), "warmup");

    std::vector<double> c_host, c_dev, d_host, d_dev;
    for (int i = 0; i < reps; ++i) {
        {   // compress: host wall and device event around the SAME launch
            auto t0 = std::chrono::steady_clock::now();
            cudaEventRecord(e0);
            cres = fsz::compress(d_in, d_cmp, n, eb, ws);
            cudaEventRecord(e1);
            check_cuda(cudaStreamSynchronize(0), "sync compress");
            auto t1 = std::chrono::steady_clock::now();
            float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1);
            c_dev.push_back(ms);
            c_host.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        {   // decompress
            auto t0 = std::chrono::steady_clock::now();
            cudaEventRecord(e0);
            fsz::decompress(d_out, d_cmp, n, eb, cres, ws);
            cudaEventRecord(e1);
            check_cuda(cudaStreamSynchronize(0), "sync decompress");
            auto t1 = std::chrono::steady_clock::now();
            float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1);
            d_dev.push_back(ms);
            d_host.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
    }

    // Bound check, so the adapter keeps the same native_quality cross-check it
    // gets from the stock CLI. d_out holds the last decompress of the loop.
    std::vector<float> h_out(n);
    check_cuda(cudaMemcpy(h_out.data(), d_out, n * sizeof(float),
                          cudaMemcpyDeviceToHost), "memcpy d2h");
    double max_err = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double e = std::fabs((double)h_in[i] - (double)h_out[i]);
        if (e > max_err) max_err = e;
    }
    const bool ok = (max_err <= (double)eb * 1.01);

    const double MB = (double)bytes / 1e6;   // decimal, matching benchkit's GB/s
    const double cd = median(c_dev), ch = median(c_host);
    const double dd = median(d_dev), dh = median(d_host);

    // One line per phase-pair. Columns mirror the stock CLI's `csv,` line where
    // they overlap, then add the host figures it cannot report.
    std::printf("hostcsv,%s,%zu,%zu,%.6e,%.6e,%.4f,%.4f,%.4f,%.4f,"
                "%.2f,%.2f,%.2f,%.2f,%.3f,%.3f,%s\n",
                path, n, (std::size_t)cres.cmp_size,
                (double)eb, max_err,
                cd, ch, dd, dh,
                MB / cd, MB / ch, MB / dd, MB / dh,
                ch / cd, dh / dd, ok ? "pass" : "fail");

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_cmp);
    return 0;
}
