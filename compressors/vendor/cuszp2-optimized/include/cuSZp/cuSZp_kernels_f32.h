#ifndef CUSZP_INCLUDE_CUSZP_CUSZP_KERNELS_F32_H
#define CUSZP_INCLUDE_CUSZP_CUSZP_KERNELS_F32_H

// Warps per thread block. Was hard-fixed to 1 (tblock_size=32); H100 profiling
// showed 1-warp blocks cap achieved occupancy at 50% (block-count/SM limit binds
// before the warp/thread limit) and force every warp to walk the decoupled-lookback
// chain individually across the whole grid. Grouping warps into blocks lets a cheap
// intra-block scan shrink that chain by this factor. Tunable.
static const int cmp_warps_per_block = 8;
static const int dec_warps_per_block = 8;
static const int cmp_tblock_size = cmp_warps_per_block * 32;
static const int dec_tblock_size = dec_warps_per_block * 32;
static const int cmp_chunk = 1024;
static const int dec_chunk = 1024;

__global__ void cuSZp_compress_kernel_outlier_f32(const float* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const float eb, const size_t nbEle);
__global__ void cuSZp_decompress_kernel_outlier_f32(float* const __restrict__ decData, const unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const float eb, const size_t nbEle);
__global__ void cuSZp_compress_kernel_plain_f32(const float* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const float eb, const size_t nbEle);
__global__ void cuSZp_decompress_kernel_plain_f32(float* const __restrict__ decData, const unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const float eb, const size_t nbEle);


#endif // CUSZP_INCLUDE_CUSZP_CUSZP_KERNELS_F32_H