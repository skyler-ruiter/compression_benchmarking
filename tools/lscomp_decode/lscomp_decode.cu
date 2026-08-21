// Minimal standalone decoder for lsCOMP bitstreams.  The upstream example CLI
// only decodes the stream it has just compressed in the same process, which is
// insufficient for a benchkit adapter whose compressed artifact must stand alone.
#include <cuda_runtime.h>
#include <lsCOMP_entry.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void ck(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T>
int decode(const char* input, const char* output, uint3 dims) {
    std::FILE* f = std::fopen(input, "rb");
    if (!f) { std::perror(input); return 2; }
    std::fseek(f, 0, SEEK_END);
    const size_t nbytes = static_cast<size_t>(std::ftell(f));
    std::fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> hcmp(nbytes);
    if (std::fread(hcmp.data(), 1, nbytes, f) != nbytes) return 2;
    std::fclose(f);

    const size_t n = static_cast<size_t>(dims.x) * dims.y * dims.z;
    unsigned char* dcmp = nullptr;
    T* dout = nullptr;
    ck(cudaMalloc(&dcmp, nbytes), "cudaMalloc compressed");
    ck(cudaMalloc(&dout, n * sizeof(T)), "cudaMalloc output");
    ck(cudaMemcpy(dcmp, hcmp.data(), nbytes, cudaMemcpyHostToDevice), "copy compressed");
    const uint4 bins = make_uint4(1, 1, 1, 1);
    if constexpr (sizeof(T) == 2)
        lsCOMP_decompression_uint16_bsize64(reinterpret_cast<uint16_t*>(dout), dcmp,
                                            nbytes, dims, bins, 1.0f, 0);
    else
        lsCOMP_decompression_uint32_bsize64(reinterpret_cast<uint32_t*>(dout), dcmp,
                                            nbytes, dims, bins, 1.0f, 0);
    ck(cudaDeviceSynchronize(), "decompress");
    std::vector<T> hout(n);
    ck(cudaMemcpy(hout.data(), dout, n * sizeof(T), cudaMemcpyDeviceToHost), "copy output");
    cudaFree(dcmp); cudaFree(dout);
    f = std::fopen(output, "wb");
    if (!f) { std::perror(output); return 2; }
    const bool ok = std::fwrite(hout.data(), sizeof(T), n, f) == n;
    std::fclose(f);
    return ok ? 0 : 2;
}

int main(int argc, char** argv) {
    const char *input = nullptr, *output = nullptr;
    std::string dtype;
    uint3 dims = make_uint3(0, 0, 0);
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-i") && i + 1 < argc) input = argv[++i];
        else if (!std::strcmp(argv[i], "-o") && i + 1 < argc) output = argv[++i];
        else if (!std::strcmp(argv[i], "-t") && i + 1 < argc) dtype = argv[++i];
        else if (!std::strcmp(argv[i], "-d") && i + 3 < argc) {
            dims.x = std::strtoul(argv[++i], nullptr, 10);
            dims.y = std::strtoul(argv[++i], nullptr, 10);
            dims.z = std::strtoul(argv[++i], nullptr, 10);
        }
    }
    if (!input || !output || (dtype != "u16" && dtype != "u32") ||
        !dims.x || !dims.y || !dims.z) {
        std::fprintf(stderr, "usage: lscomp_decode -i stream -o integers -t u16|u32 -d slow mid fast\n");
        return 2;
    }
    return dtype == "u16" ? decode<uint16_t>(input, output, dims)
                           : decode<uint32_t>(input, output, dims);
}
