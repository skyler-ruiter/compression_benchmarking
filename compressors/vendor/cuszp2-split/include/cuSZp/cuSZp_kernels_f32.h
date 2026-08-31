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

// Flat launch width for the count/pack split-pipeline kernels (compress only).
// Each thread there handles exactly one independent 32-element block -- no warp
// cooperation, no scan, no lookback -- so this is just an occupancy knob, not tied
// to cmp_chunk/cmp_warps_per_block at all.
static const int split_tblock_size = 256;

__global__ void cuSZp_compress_kernel_outlier_f32(const float* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const float eb, const size_t nbEle);
__global__ void cuSZp_decompress_kernel_outlier_f32(float* const __restrict__ decData, const unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const float eb, const size_t nbEle);
__global__ void cuSZp_compress_kernel_plain_f32(const float* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const float eb, const size_t nbEle);
__global__ void cuSZp_decompress_kernel_plain_f32(float* const __restrict__ decData, const unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const float eb, const size_t nbEle);

// Split count->scan->pack compress pipeline (see cuSZp_entry_f32.cu). K1 (count) is
// a fine-grained, dependency-free pass: one thread per 32-element block, writes the
// same 1-byte-per-block rate header the fused kernel writes, over the SAME padded
// header range the multi-warp decompressor expects (rate_ofs), so this remains
// wire-compatible with cuSZp_decompress_kernel_{plain,outlier}_f32 above. Between
// K1 and K2, cub::DeviceScan computes each real block's exclusive payload-byte
// offset directly from the header bytes K1 just wrote (via a transform iterator),
// removing the decoupled-lookback scan from the bandwidth-heavy passes entirely.
// K2 (pack) re-quantizes from oriData (cheap; avoids storing/rereading an
// intermediate) and writes each block's payload at its now-known offset.
__global__ void cuSZp_count_kernel_plain_f32(const float* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, const float eb, const size_t nbEle, const size_t numHeaderBlocks);
__global__ void cuSZp_count_kernel_outlier_f32(const float* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, const float eb, const size_t nbEle, const size_t numHeaderBlocks);
__global__ void cuSZp_pack_kernel_plain_f32(const float* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, const unsigned int* const __restrict__ blockOffsets, const float eb, const size_t nbEle, const size_t numRealBlocks, const unsigned int rate_ofs);
__global__ void cuSZp_pack_kernel_outlier_f32(const float* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, const unsigned int* const __restrict__ blockOffsets, const float eb, const size_t nbEle, const size_t numRealBlocks, const unsigned int rate_ofs);

#endif // CUSZP_INCLUDE_CUSZP_CUSZP_KERNELS_F32_H