#include "cuSZp_entry_1D_f32.h"
#include "cuSZp_kernels_1D_f32.h"

// Scratch-buffer cache: cuSZp originally cudaMalloc'd + cudaFree'd 3 small scratch
// arrays (d_cmpOffset/d_locOffset/d_flag) on every compress/decompress call. Cheap
// on bare-metal GPUs, but on some platforms (observed on a GPU-passthrough cloud
// VM) cudaMalloc/cudaFree carry large, highly variable latency (seen up to several
// hundred ms for a single call) that dwarfs actual kernel time and dominates any
// per-call timing. One cache per function below, grow-allocated once per distinct
// cmpOffSize and reused across calls (still memset to 0 every call — correctness
// unaffected). Freed only at process exit, fine for this CLI.
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
CuszpScratch g_scratch_cuSZp_compress_1D_fixed_f32;
CuszpScratch g_scratch_cuSZp_decompress_1D_fixed_f32;
CuszpScratch g_scratch_cuSZp_compress_1D_plain_f32;
CuszpScratch g_scratch_cuSZp_decompress_1D_plain_f32;
CuszpScratch g_scratch_cuSZp_compress_1D_outlier_f32;
CuszpScratch g_scratch_cuSZp_decompress_1D_outlier_f32;
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
void cuSZp_compress_1D_fixed_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound, cudaStream_t stream)
{
    // Data blocking.
    int bsize = tblock_size;
    int gsize = (nbEle + bsize * thread_chunk - 1) / (bsize * thread_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    g_scratch_cuSZp_compress_1D_fixed_f32.ensure(cmpOffSize);
    unsigned int* d_cmpOffset = g_scratch_cuSZp_compress_1D_fixed_f32.d_cmpOffset;
    unsigned int* d_locOffset = g_scratch_cuSZp_compress_1D_fixed_f32.d_locOffset;
    int* d_flag = g_scratch_cuSZp_compress_1D_fixed_f32.d_flag;
    unsigned int glob_sync;
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuSZp_compress_kernel_1D_fixed_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);

    // Obtain compression ratio and move data back to CPU.  
    cudaMemcpy(&glob_sync, d_cmpOffset+cmpOffSize-1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    *cmpSize = (size_t)glob_sync + (nbEle+tblock_size*thread_chunk-1)/(tblock_size*thread_chunk)*(tblock_size*thread_chunk)/32;

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
void cuSZp_decompress_1D_fixed_f32(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, float errorBound, cudaStream_t stream)
{
    // Data blocking.
    int bsize = tblock_size;
    int gsize = (nbEle + bsize * thread_chunk - 1) / (bsize * thread_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU decompression.
    g_scratch_cuSZp_decompress_1D_fixed_f32.ensure(cmpOffSize);
    unsigned int* d_cmpOffset = g_scratch_cuSZp_decompress_1D_fixed_f32.d_cmpOffset;
    unsigned int* d_locOffset = g_scratch_cuSZp_decompress_1D_fixed_f32.d_locOffset;
    int* d_flag = g_scratch_cuSZp_decompress_1D_fixed_f32.d_flag;
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // cuSZp GPU decompression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuSZp_decompress_kernel_1D_fixed_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_decData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);
    
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
void cuSZp_compress_1D_plain_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound, cudaStream_t stream)
{
    // Data blocking.
    int bsize = tblock_size;
    int gsize = (nbEle + bsize * thread_chunk - 1) / (bsize * thread_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    g_scratch_cuSZp_compress_1D_plain_f32.ensure(cmpOffSize);
    unsigned int* d_cmpOffset = g_scratch_cuSZp_compress_1D_plain_f32.d_cmpOffset;
    unsigned int* d_locOffset = g_scratch_cuSZp_compress_1D_plain_f32.d_locOffset;
    int* d_flag = g_scratch_cuSZp_compress_1D_plain_f32.d_flag;
    unsigned int glob_sync;
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuSZp_compress_kernel_1D_plain_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);

    // Obtain compression ratio and move data back to CPU.  
    cudaMemcpy(&glob_sync, d_cmpOffset+cmpOffSize-1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    *cmpSize = (size_t)glob_sync + (nbEle+tblock_size*thread_chunk-1)/(tblock_size*thread_chunk)*(tblock_size*thread_chunk)/32;

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
void cuSZp_decompress_1D_plain_f32(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, float errorBound, cudaStream_t stream)
{
    // Data blocking.
    int bsize = tblock_size;
    int gsize = (nbEle + bsize * thread_chunk - 1) / (bsize * thread_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU decompression.
    g_scratch_cuSZp_decompress_1D_plain_f32.ensure(cmpOffSize);
    unsigned int* d_cmpOffset = g_scratch_cuSZp_decompress_1D_plain_f32.d_cmpOffset;
    unsigned int* d_locOffset = g_scratch_cuSZp_decompress_1D_plain_f32.d_locOffset;
    int* d_flag = g_scratch_cuSZp_decompress_1D_plain_f32.d_flag;
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // cuSZp GPU decompression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuSZp_decompress_kernel_1D_plain_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_decData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);
    
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
void cuSZp_compress_1D_outlier_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound, cudaStream_t stream)
{
    // Data blocking.
    int bsize = tblock_size;
    int gsize = (nbEle + bsize * thread_chunk - 1) / (bsize * thread_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    g_scratch_cuSZp_compress_1D_outlier_f32.ensure(cmpOffSize);
    unsigned int* d_cmpOffset = g_scratch_cuSZp_compress_1D_outlier_f32.d_cmpOffset;
    unsigned int* d_locOffset = g_scratch_cuSZp_compress_1D_outlier_f32.d_locOffset;
    int* d_flag = g_scratch_cuSZp_compress_1D_outlier_f32.d_flag;
    unsigned int glob_sync;
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuSZp_compress_kernel_1D_outlier_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);

    // Obtain compression ratio and move data back to CPU.  
    cudaMemcpy(&glob_sync, d_cmpOffset+cmpOffSize-1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    *cmpSize = (size_t)glob_sync + (nbEle+tblock_size*thread_chunk-1)/(tblock_size*thread_chunk)*(tblock_size*thread_chunk)/32;

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
void cuSZp_decompress_1D_outlier_f32(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, float errorBound, cudaStream_t stream)
{
    // Data blocking.
    int bsize = tblock_size;
    int gsize = (nbEle + bsize * thread_chunk - 1) / (bsize * thread_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU decompression.
    g_scratch_cuSZp_decompress_1D_outlier_f32.ensure(cmpOffSize);
    unsigned int* d_cmpOffset = g_scratch_cuSZp_decompress_1D_outlier_f32.d_cmpOffset;
    unsigned int* d_locOffset = g_scratch_cuSZp_decompress_1D_outlier_f32.d_locOffset;
    int* d_flag = g_scratch_cuSZp_decompress_1D_outlier_f32.d_flag;
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // cuSZp GPU decompression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuSZp_decompress_kernel_1D_outlier_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_decData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);
    
}