#include "cuSZp_entry_f64.h"
#include "cuSZp_kernels_f64.h"

// Scratch-buffer cache: see cuSZp_entry_f32.cu for the full rationale. Same fix,
// f64 side.
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
CuszpScratch g_compress_plain_f64;
CuszpScratch g_decompress_plain_f64;
CuszpScratch g_compress_outlier_f64;
CuszpScratch g_decompress_outlier_f64;
}  // namespace

/** ************************************************************************
 * @brief cuSZp end-to-end compression API for device pointers
 *        Compression is executed in GPU.
 *        Original data is stored as device pointers (in GPU).
 *        Compressed data is stored back as device pointers (in GPU).
 *
 * @param   d_oriData       original data (device pointer)
 * @param   d_cmpBytes      compressed data (device pointer)
 * @param   nbEle           original data size (number of floating point)
 * @param   cmpSize         compressed data size (number of unsigned char)
 * @param   errorBound      user-defined error bound
 * @param   stream          CUDA stream for executing compression kernel
 * *********************************************************************** */
void cuSZp_compress_plain_f64(double* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, double errorBound, cudaStream_t stream)
{
    // Data blocking.
    int bsize = cmp_tblock_size;
    int gsize = (nbEle + bsize * cmp_chunk - 1) / (bsize * cmp_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression (cached, see CuszpScratch above).
    g_compress_plain_f64.ensure(cmpOffSize);
    unsigned int* d_cmpOffset = g_compress_plain_f64.d_cmpOffset;
    unsigned int* d_locOffset = g_compress_plain_f64.d_locOffset;
    int* d_flag = g_compress_plain_f64.d_flag;
    unsigned int glob_sync;
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuSZp_compress_kernel_plain_f64<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);

    // Obtain compression ratio and move data back to CPU.
    cudaMemcpy(&glob_sync, d_cmpOffset+cmpOffSize-1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    *cmpSize = (size_t)glob_sync + (nbEle+cmp_tblock_size*cmp_chunk-1)/(cmp_tblock_size*cmp_chunk)*(cmp_tblock_size*cmp_chunk)/32;
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
void cuSZp_decompress_plain_f64(double* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, double errorBound, cudaStream_t stream)
{
    // Data blocking.
    int bsize = dec_tblock_size;
    int gsize = (nbEle + bsize * dec_chunk - 1) / (bsize * dec_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU decompression (cached, see CuszpScratch above).
    g_decompress_plain_f64.ensure(cmpOffSize);
    unsigned int* d_cmpOffset = g_decompress_plain_f64.d_cmpOffset;
    unsigned int* d_locOffset = g_decompress_plain_f64.d_locOffset;
    int* d_flag = g_decompress_plain_f64.d_flag;
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // cuSZp GPU decompression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuSZp_decompress_kernel_plain_f64<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_decData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);
}

/** ************************************************************************
 * @brief cuSZp end-to-end compression API for device pointers
 *        Compression is executed in GPU.
 *        Original data is stored as device pointers (in GPU).
 *        Compressed data is stored back as device pointers (in GPU).
 *
 * @param   d_oriData       original data (device pointer)
 * @param   d_cmpBytes      compressed data (device pointer)
 * @param   nbEle           original data size (number of floating point)
 * @param   cmpSize         compressed data size (number of unsigned char)
 * @param   errorBound      user-defined error bound
 * @param   stream          CUDA stream for executing compression kernel
 * *********************************************************************** */
void cuSZp_compress_outlier_f64(double* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, double errorBound, cudaStream_t stream)
{
    // Data blocking.
    int bsize = cmp_tblock_size;
    int gsize = (nbEle + bsize * cmp_chunk - 1) / (bsize * cmp_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression (cached, see CuszpScratch above).
    g_compress_outlier_f64.ensure(cmpOffSize);
    unsigned int* d_cmpOffset = g_compress_outlier_f64.d_cmpOffset;
    unsigned int* d_locOffset = g_compress_outlier_f64.d_locOffset;
    int* d_flag = g_compress_outlier_f64.d_flag;
    unsigned int glob_sync;
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuSZp_compress_kernel_outlier_f64<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);

    // Obtain compression ratio and move data back to CPU.
    cudaMemcpy(&glob_sync, d_cmpOffset+cmpOffSize-1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    *cmpSize = (size_t)glob_sync + (nbEle+cmp_tblock_size*cmp_chunk-1)/(cmp_tblock_size*cmp_chunk)*(cmp_tblock_size*cmp_chunk)/32;
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
void cuSZp_decompress_outlier_f64(double* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, double errorBound, cudaStream_t stream)
{
    // Data blocking.
    int bsize = dec_tblock_size;
    int gsize = (nbEle + bsize * dec_chunk - 1) / (bsize * dec_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU decompression (cached, see CuszpScratch above).
    g_decompress_outlier_f64.ensure(cmpOffSize);
    unsigned int* d_cmpOffset = g_decompress_outlier_f64.d_cmpOffset;
    unsigned int* d_locOffset = g_decompress_outlier_f64.d_locOffset;
    int* d_flag = g_decompress_outlier_f64.d_flag;
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // cuSZp GPU decompression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuSZp_decompress_kernel_outlier_f64<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_decData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);
}
