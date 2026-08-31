#include "cuSZp_entry_f32.h"
#include "cuSZp_kernels_f32.h"
#include <cub/cub.cuh>

// Scratch-buffer cache: cuSZp_{compress,decompress}_{plain,outlier}_f32 originally
// cudaMalloc'd + cudaFree'd their 3 small scratch arrays (d_cmpOffset/d_locOffset/
// d_flag) on *every call*. That's cheap on bare-metal GPUs but on some platforms
// (observed on a GPU-passthrough cloud VM) cudaMalloc/cudaFree carry large, highly
// variable latency (seen up to several hundred ms for a single call) that dwarfs the
// actual kernel time (~145us for a 26MB field) and dominates any per-call timing.
// These caches grow-allocate once per distinct cmpOffSize and are reused across
// calls — same correctness (still memset to 0 every call), no more per-call
// malloc/free. Freed only at process exit, which is fine for this CLI; a caller
// embedding this as a long-running service with widely varying, one-off sizes
// would want an explicit release API instead (not needed here).
//
// Compress no longer uses this (see CuszpSplitScratch below) -- only decompress,
// which still runs the fused multi-warp decoder, needs the lookback arrays.
namespace {
struct CuszpScratch {
    unsigned int* d_cmpOffset = nullptr;
    unsigned int* d_locOffset = nullptr;
    int* d_flag = nullptr;
    size_t capacity = 0;

    void ensure(size_t cmpOffSize) {
        if (cmpOffSize <= capacity) return;
        if (d_cmpOffset) cudaFree(d_cmpOffset);
        if (d_locOffset) cudaFree(d_locOffset);
        if (d_flag) cudaFree(d_flag);
        cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int) * cmpOffSize);
        cudaMalloc((void**)&d_locOffset, sizeof(unsigned int) * cmpOffSize);
        cudaMalloc((void**)&d_flag, sizeof(int) * cmpOffSize);
        capacity = cmpOffSize;
    }
};
CuszpScratch g_decompress_plain_f32;
CuszpScratch g_decompress_outlier_f32;

// Scratch for the split count->scan->pack compress pipeline: the per-real-block
// exclusive payload-byte-offset array, plus CUB's own scan temp storage. Cached
// the same way as CuszpScratch above and for the same reason.
struct CuszpSplitScratch {
    unsigned int* d_blockOffsets = nullptr;
    size_t blockOffsetsCapacity = 0;
    void* d_tempStorage = nullptr;
    size_t tempStorageCapacity = 0;

    void ensureBlockOffsets(size_t n) {
        if (n <= blockOffsetsCapacity) return;
        if (d_blockOffsets) cudaFree(d_blockOffsets);
        cudaMalloc((void**)&d_blockOffsets, sizeof(unsigned int) * n);
        blockOffsetsCapacity = n;
    }
    void ensureTempStorage(size_t bytes) {
        if (bytes <= tempStorageCapacity) return;
        if (d_tempStorage) cudaFree(d_tempStorage);
        cudaMalloc(&d_tempStorage, bytes);
        tempStorageCapacity = bytes;
    }
};
CuszpSplitScratch g_split_plain_f32;
CuszpSplitScratch g_split_outlier_f32;

// Transform functors used to derive each real block's payload byte count directly
// from the rate/encoding header byte K1 already wrote to cmpData -- no separate
// counts array. Composed with cub::CountingInputIterator<int> so that scanning
// numRealBlocks+1 "items" yields both every real block's exclusive offset (outputs
// 0..numRealBlocks-1) and the grand total payload size (output numRealBlocks) in
// one DeviceScan call; the dummy index==numRealBlocks input is never summed into
// anything we read, so its value (0) is irrelevant.
struct RateToBytesPlain {
    const unsigned char* cmpData;
    size_t numRealBlocks;
    __host__ __device__ __forceinline__
    unsigned int operator()(const int& i) const {
        if ((size_t)i >= numRealBlocks) return 0u;
        unsigned char rate = cmpData[i];
        return rate ? (4u + (unsigned int)rate * 4u) : 0u;
    }
};

struct RateToBytesOutlier {
    const unsigned char* cmpData;
    size_t numRealBlocks;
    __host__ __device__ __forceinline__
    unsigned int operator()(const int& i) const {
        if ((size_t)i >= numRealBlocks) return 0u;
        unsigned char enc = cmpData[i];
        int encoding_selection = enc >> 7;
        int outlier_byte_num = ((enc & 0x60) >> 5) + 1;
        int rate = enc & 0x1f;
        if (encoding_selection) return (unsigned int)(4 + outlier_byte_num + rate * 4);
        return rate ? (unsigned int)(4 + rate * 4) : 0u;
    }
};
}  // namespace

/** ************************************************************************
 * @brief cuSZp end-to-end compression API for device pointers (split pipeline)
 *        Compression is executed in GPU as three passes: a dependency-free count
 *        kernel (K1), a cub::DeviceScan over the header bytes K1 wrote, and a
 *        dependency-free pack kernel (K2) writing each block's payload at its
 *        now-known offset. See cuSZp_kernels_f32.h for the design rationale.
 *        Original data is stored as device pointers (in GPU).
 *        Compressed data is stored back as device pointers (in GPU), and is
 *        wire-compatible with cuSZp_decompress_plain_f32 below (same header
 *        layout/size as the fused multi-warp encoder it replaces).
 *
 * @param   d_oriData       original data (device pointer)
 * @param   d_cmpBytes      compressed data (device pointer)
 * @param   nbEle           original data size (number of floating point)
 * @param   cmpSize         compressed data size (number of unsigned char)
 * @param   errorBound      user-defined error bound
 * @param   stream          CUDA stream for executing compression kernel
 * *********************************************************************** */
void cuSZp_compress_plain_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound, cudaStream_t stream)
{
    // Header size, computed identically to the fused multi-warp encoder so the two
    // remain wire-compatible (decompress can't tell which compressor produced its
    // input). rate_ofs is always a multiple of 1024, hence of 4 -- see cuSZp_kernels_f32.h.
    const unsigned int rate_ofs = (nbEle+cmp_tblock_size*cmp_chunk-1)/(cmp_tblock_size*cmp_chunk)*(cmp_tblock_size*cmp_chunk)/32;
    const size_t numRealBlocks = (nbEle + 31) / 32;

    // K1 (count): covers the full padded header range so the decoder still finds a
    // valid (zero) rate byte for every padding block beyond numRealBlocks it reads.
    dim3 countBlock(split_tblock_size);
    dim3 countGrid((rate_ofs + split_tblock_size - 1) / split_tblock_size);
    cuSZp_count_kernel_plain_f32<<<countGrid, countBlock, 0, stream>>>(d_oriData, d_cmpBytes, errorBound, nbEle, rate_ofs);

    // Scan: exclusive payload-byte offset per real block (+ grand total as the last element).
    g_split_plain_f32.ensureBlockOffsets(numRealBlocks + 1);
    cub::CountingInputIterator<int> countItr(0);
    RateToBytesPlain xform{d_cmpBytes, numRealBlocks};
    cub::TransformInputIterator<unsigned int, RateToBytesPlain, cub::CountingInputIterator<int>> transformItr(countItr, xform);

    size_t tempBytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, tempBytes, transformItr, g_split_plain_f32.d_blockOffsets, numRealBlocks + 1, stream);
    g_split_plain_f32.ensureTempStorage(tempBytes);
    cub::DeviceScan::ExclusiveSum(g_split_plain_f32.d_tempStorage, tempBytes, transformItr, g_split_plain_f32.d_blockOffsets, numRealBlocks + 1, stream);

    // K2 (pack): only real blocks carry payload.
    dim3 packBlock(split_tblock_size);
    dim3 packGrid((numRealBlocks + split_tblock_size - 1) / split_tblock_size);
    cuSZp_pack_kernel_plain_f32<<<packGrid, packBlock, 0, stream>>>(d_oriData, d_cmpBytes, g_split_plain_f32.d_blockOffsets, errorBound, nbEle, numRealBlocks, rate_ofs);

    unsigned int totalPayloadBytes;
    cudaMemcpy(&totalPayloadBytes, g_split_plain_f32.d_blockOffsets + numRealBlocks, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    *cmpSize = (size_t)rate_ofs + totalPayloadBytes;
}

 /** ************************************************************************
 * @brief cuSZp end-to-end decompression API for device pointers
 *        Decompression is executed in GPU.
 *        Compressed data is stored as device pointers (in GPU).
 *        Reconstructed data is stored as device pointers (in GPU).
 *        P.S. Reconstructed data and original data have the same shape.
 *
 * @param   d_decData       reconstructed data (device pointer)
 * @param   d_cmpBytes      compressed data (device pointer)
 * @param   nbEle           reconstructed data size (number of floating point)
 * @param   cmpSize         compressed data size (number of unsigned char)
 * @param   errorBound      user-defined error bound
 * @param   stream          CUDA stream for executing compression kernel
 * *********************************************************************** */
void cuSZp_decompress_plain_f32(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, float errorBound, cudaStream_t stream)
{
    // Data blocking.
    int bsize = dec_tblock_size;
    int gsize = (nbEle + bsize * dec_chunk - 1) / (bsize * dec_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU decompression (cached, see CuszpScratch above).
    g_decompress_plain_f32.ensure(cmpOffSize);
    unsigned int* d_cmpOffset = g_decompress_plain_f32.d_cmpOffset;
    unsigned int* d_locOffset = g_decompress_plain_f32.d_locOffset;
    int* d_flag = g_decompress_plain_f32.d_flag;
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // cuSZp GPU decompression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuSZp_decompress_kernel_plain_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_decData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);
}

/** ************************************************************************
 * @brief cuSZp end-to-end compression API for device pointers (split pipeline)
 *        See cuSZp_compress_plain_f32 above for the pipeline description; this
 *        is the outlier-mode counterpart (variable-length, non-4-byte-aligned
 *        payloads -- K2 reuses the fused encoder's vec_ofs realignment branches
 *        unchanged).
 *
 * @param   d_oriData       original data (device pointer)
 * @param   d_cmpBytes      compressed data (device pointer)
 * @param   nbEle           original data size (number of floating point)
 * @param   cmpSize         compressed data size (number of unsigned char)
 * @param   errorBound      user-defined error bound
 * @param   stream          CUDA stream for executing compression kernel
 * *********************************************************************** */
void cuSZp_compress_outlier_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound, cudaStream_t stream)
{
    const unsigned int rate_ofs = (nbEle+cmp_tblock_size*cmp_chunk-1)/(cmp_tblock_size*cmp_chunk)*(cmp_tblock_size*cmp_chunk)/32;
    const size_t numRealBlocks = (nbEle + 31) / 32;

    dim3 countBlock(split_tblock_size);
    dim3 countGrid((rate_ofs + split_tblock_size - 1) / split_tblock_size);
    cuSZp_count_kernel_outlier_f32<<<countGrid, countBlock, 0, stream>>>(d_oriData, d_cmpBytes, errorBound, nbEle, rate_ofs);

    g_split_outlier_f32.ensureBlockOffsets(numRealBlocks + 1);
    cub::CountingInputIterator<int> countItr(0);
    RateToBytesOutlier xform{d_cmpBytes, numRealBlocks};
    cub::TransformInputIterator<unsigned int, RateToBytesOutlier, cub::CountingInputIterator<int>> transformItr(countItr, xform);

    size_t tempBytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, tempBytes, transformItr, g_split_outlier_f32.d_blockOffsets, numRealBlocks + 1, stream);
    g_split_outlier_f32.ensureTempStorage(tempBytes);
    cub::DeviceScan::ExclusiveSum(g_split_outlier_f32.d_tempStorage, tempBytes, transformItr, g_split_outlier_f32.d_blockOffsets, numRealBlocks + 1, stream);

    dim3 packBlock(split_tblock_size);
    dim3 packGrid((numRealBlocks + split_tblock_size - 1) / split_tblock_size);
    cuSZp_pack_kernel_outlier_f32<<<packGrid, packBlock, 0, stream>>>(d_oriData, d_cmpBytes, g_split_outlier_f32.d_blockOffsets, errorBound, nbEle, numRealBlocks, rate_ofs);

    unsigned int totalPayloadBytes;
    cudaMemcpy(&totalPayloadBytes, g_split_outlier_f32.d_blockOffsets + numRealBlocks, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    *cmpSize = (size_t)rate_ofs + totalPayloadBytes;
}

 /** ************************************************************************
 * @brief cuSZp end-to-end decompression API for device pointers
 *        Decompression is executed in GPU.
 *        Compressed data is stored as device pointers (in GPU).
 *        Reconstructed data is stored as device pointers (in GPU).
 *        P.S. Reconstructed data and original data have the same shape.
 *
 * @param   d_decData       reconstructed data (device pointer)
 * @param   d_cmpBytes      compressed data (device pointer)
 * @param   nbEle           reconstructed data size (number of floating point)
 * @param   cmpSize         compressed data size (number of unsigned char)
 * @param   errorBound      user-defined error bound
 * @param   stream          CUDA stream for executing compression kernel
 * *********************************************************************** */
void cuSZp_decompress_outlier_f32(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, float errorBound, cudaStream_t stream)
{
    // Data blocking.
    int bsize = dec_tblock_size;
    int gsize = (nbEle + bsize * dec_chunk - 1) / (bsize * dec_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU decompression (cached, see CuszpScratch above).
    g_decompress_outlier_f32.ensure(cmpOffSize);
    unsigned int* d_cmpOffset = g_decompress_outlier_f32.d_cmpOffset;
    unsigned int* d_locOffset = g_decompress_outlier_f32.d_locOffset;
    int* d_flag = g_decompress_outlier_f32.d_flag;
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // cuSZp GPU decompression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuSZp_decompress_kernel_outlier_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_decData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);
}
