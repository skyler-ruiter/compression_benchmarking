// probe_selftest.cu — a small, known CUDA allocation sequence whose peak live
// device-memory footprint is analytically computable. Run this binary under
// libcudamemprobe.so (see run_selftest.sh) and diff its reported
// `peak_device_bytes` / `untracked_frees` against the values this program
// prints to stdout.
//
// Design: we mirror, host-side, exactly what a correct probe should compute
// (add on alloc, subtract on free, track running max) using the ACTUAL byte
// counts CUDA reports back to us (important for cudaMallocPitch, whose true
// size is pitch*height, not width*height). The phases after the first are
// sized so they never exceed the phase-1 peak, so the overall expected peak
// is unambiguous even though we don't hardcode the driver's pitch alignment.
//
// Sequence (MB = 1<<20):
//   Phase 1 (plain cudaMalloc/cudaFree):
//     A = malloc(100MB)                 live=100MB   peak=100MB
//     B = malloc(50MB)                  live=150MB   peak=150MB
//     free(A)                            live=50MB
//     C = malloc(200MB)                 live=250MB   peak=250MB  <- global peak
//     free(B); free(C)                   live=0
//   Phase 2 (managed):
//     D = mallocManaged(40MB); free(D)   live touches 40MB, peak stays 250MB
//   Phase 3 (pitched):
//     mallocPitch(~0.95MB row, 30 rows); free
//       true bytes = actual_pitch * 30, well under 250MB either way
//   Phase 4 (stream-ordered async):
//     E = mallocAsync(60MB, stream); sync; freeAsync(E, stream); sync
//   Phase 5 (deliberate unmatched free):
//     cudaFree() on a pointer that was never allocated by us -> the wrapper
//     must count this as one untracked free and must not go negative.
//
// Expected: EXPECTED_PEAK = 250MB exactly (phase 1 dominates every later
// phase), EXPECTED_UNTRACKED_FREES = 1 (only the deliberate bogus free).

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static const size_t MB = 1ull << 20;

// Running host-side model of what a correct probe should be tracking.
static size_t g_expected_live = 0;
static size_t g_expected_peak = 0;
static size_t g_expected_untracked = 0;

static void track_alloc(size_t bytes) {
    g_expected_live += bytes;
    if (g_expected_live > g_expected_peak) g_expected_peak = g_expected_live;
}
static void track_free(size_t bytes) {
    g_expected_live -= bytes;
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(_err));                                  \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

int main() {
    // Make sure a device/context exists before we start counting; the CUDA
    // context itself is not a cudaMalloc and must not be counted by the probe.
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0)); // force context init, cheaply, before phase 1

    // ---- Phase 1: plain cudaMalloc / cudaFree --------------------------------
    void *A = nullptr, *B = nullptr, *C = nullptr;
    CUDA_CHECK(cudaMalloc(&A, 100 * MB));
    track_alloc(100 * MB);
    fprintf(stderr, "[selftest] after A: live=%zu peak=%zu\n", g_expected_live, g_expected_peak);

    CUDA_CHECK(cudaMalloc(&B, 50 * MB));
    track_alloc(50 * MB);
    fprintf(stderr, "[selftest] after B: live=%zu peak=%zu\n", g_expected_live, g_expected_peak);

    CUDA_CHECK(cudaFree(A));
    track_free(100 * MB);
    fprintf(stderr, "[selftest] after free(A): live=%zu peak=%zu\n", g_expected_live, g_expected_peak);

    CUDA_CHECK(cudaMalloc(&C, 200 * MB));
    track_alloc(200 * MB);
    fprintf(stderr, "[selftest] after C: live=%zu peak=%zu\n", g_expected_live, g_expected_peak);

    CUDA_CHECK(cudaFree(B));
    track_free(50 * MB);
    CUDA_CHECK(cudaFree(C));
    track_free(200 * MB);
    fprintf(stderr, "[selftest] end of phase 1: live=%zu peak=%zu\n", g_expected_live, g_expected_peak);

    // ---- Phase 2: cudaMallocManaged ------------------------------------------
    void *D = nullptr;
    CUDA_CHECK(cudaMallocManaged(&D, 40 * MB, cudaMemAttachGlobal));
    track_alloc(40 * MB);
    CUDA_CHECK(cudaFree(D));
    track_free(40 * MB);
    fprintf(stderr, "[selftest] end of phase 2 (managed): live=%zu peak=%zu\n", g_expected_live, g_expected_peak);

    // ---- Phase 3: cudaMallocPitch ---------------------------------------------
    // Use an intentionally non-page-aligned row width so the driver's real
    // pitch padding is exercised; we use the ACTUAL returned pitch (not the
    // requested width) to compute the true byte count, per the README.
    void *P = nullptr;
    size_t pitch = 0;
    const size_t width_bytes = 1000 * 1024 + 37; // deliberately odd/unaligned
    const size_t height = 30;
    CUDA_CHECK(cudaMallocPitch(&P, &pitch, width_bytes, height));
    size_t pitch_total_bytes = pitch * height;
    fprintf(stderr, "[selftest] pitch alloc: requested width=%zu pitch=%zu height=%zu true_bytes=%zu\n",
            width_bytes, pitch, height, pitch_total_bytes);
    track_alloc(pitch_total_bytes);
    CUDA_CHECK(cudaFree(P));
    track_free(pitch_total_bytes);
    fprintf(stderr, "[selftest] end of phase 3 (pitched): live=%zu peak=%zu\n", g_expected_live, g_expected_peak);

    // ---- Phase 4: stream-ordered cudaMallocAsync / cudaFreeAsync ---------------
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    void *E = nullptr;
    CUDA_CHECK(cudaMallocAsync(&E, 60 * MB, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream)); // make sure the alloc has landed
    track_alloc(60 * MB);
    fprintf(stderr, "[selftest] after async alloc: live=%zu peak=%zu\n", g_expected_live, g_expected_peak);

    CUDA_CHECK(cudaFreeAsync(E, stream));
    // The probe (per README) counts the free at call time, slightly ahead of
    // the stream actually retiring it; mirror that here.
    track_free(60 * MB);
    CUDA_CHECK(cudaStreamSynchronize(stream)); // let the free actually settle
    CUDA_CHECK(cudaStreamDestroy(stream));
    fprintf(stderr, "[selftest] end of phase 4 (async): live=%zu peak=%zu\n", g_expected_live, g_expected_peak);

    // ---- Phase 5: deliberate unmatched free ------------------------------------
    // A pointer value that was never returned by any cuda*Alloc* call in this
    // process. cudaFree() on it is expected to return an error internally,
    // but the probe wrapper must still observe the call and count it as an
    // untracked free (and must not corrupt its live/peak accounting).
    void *bogus = reinterpret_cast<void *>(0x1);
    cudaError_t bogus_rc = cudaFree(bogus);
    (void)bogus_rc; // intentionally ignored: this call is expected to fail
    // Clear whatever sticky error cudaFree(bogus) may have left, so it doesn't
    // get misattributed to the (unrelated) synchronize below.
    cudaGetLastError();
    g_expected_untracked += 1;

    CUDA_CHECK(cudaDeviceSynchronize());

    printf("EXPECTED_PEAK=%zu\n", g_expected_peak);
    printf("EXPECTED_UNTRACKED_FREES=%zu\n", g_expected_untracked);
    return 0;
}
