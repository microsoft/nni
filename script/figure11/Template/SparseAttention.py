# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
sparse_attention_template = """
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
using namespace std;
// Macro definition for the cuda and cusparse

#include <assert.h>
// CUDA runtime
#include <cuda.h>

#define OFFSET(row, col, ld) ((row) * ld + col)
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&pointer))[0]
#define FETCH_UINT32(pointer) (reinterpret_cast<unsigned int*>(&(pointer))[0])
#define FETCH_UINT32x4(pointer) (reinterpret_cast<uint4*>(&(pointer))[0])
#define FETCH_INT32(pointer) (reinterpret_cast<int*>(&(pointer))[0])
#define FULL_MASK 0xffffffff

#define HEAD_NUM _REPLACE_HEAD_NUM
#define SPARSE_VAL_SIZE _REPLACE_SPARSE_VAL_SIZE
#define GLOBAL_M _REPLACE_GLOBAL_M
#define GLOBAL_N _REPLACE_GLOBAL_N
#define GLOBAL_K _REPLACE_GLOBAL_K
#define SMALL_BLOCK_NUM _REPLACE_SMALL_BLOCK_NUM
#define SOFTMAX_BLOCK_SIZE_M 32
#define SOFTMAX_BLOCK_SIZE_N 32


__device__ __forceinline__ const float* add_ptr_u(const float* src, int offset)      \
{                                                                            \
    const float* dst;                                                            \
    asm("{                       \\n\\t"                                       \
        ".reg .u32 lo,hi,of;     \\n\\t"                                       \
        "mul.lo.u32 of, %2, %3;  \\n\\t"                                       \
        "mov.b64    {lo,hi}, %1; \\n\\t"                                       \
        "add.cc.u32  lo,lo,  of; \\n\\t"                                       \
        "addc.u32    hi,hi,  0;  \\n\\t"                                       \
        "mov.b64 %0, {lo,hi};    \\n\\t"                                       \
        "}" : "=l"(dst) : "l"(src), "r"(offset), "r"((int)sizeof(*src)));    \
    return dst;                                                              \
}

__device__ __forceinline__ float2  _add(float2 x, float2 y) { float2 res; res.x = x.x + y.x; res.y = x.y + y.y; return res; }


__global__ void BLOCK_SPARSE_MATMUL_OUT_32_64_32(
    float* A,
    float* B,
    float* C_val,
    int* m_index,
    int* n_index,
    int* block_index){
    /*
    description:
    tiling k dimension
    smm_dd_s_nn: sparse matmul, dense (MxK, along K) x dense (KxN, along N) -> sparse (MxN, along N)
    the output sparse is block size 32x32, the blocks will be written to bcsr 32x64
    */
    const int BLOCK_SIZE_M = 32;  // 64
    const int BLOCK_SIZE_K = 64;  //8
    const int BLOCK_SIZE_N = 32;  //128
    const int THREAD_SIZE_K = 64;
    const int M = GLOBAL_M;
    const int K = GLOBAL_K;
    const int N = GLOBAL_N;

    A += M*K*blockIdx.y;
    B += K*N*blockIdx.y;
    C_val += SPARSE_VAL_SIZE*blockIdx.y;

    assert(blockDim.x % 32 == 0);
    uint n_warp = 8; // blockDim.x / 32
    assert(THREAD_SIZE_K % n_warp == 0);
    // THREAD_SIZE_K: one loop k
    assert(K % THREAD_SIZE_K == 0);

    assert(BLOCK_SIZE_M == BLOCK_SIZE_N);
    __shared__ float fShare[65 * 32 * 2];
    char* bShare = (char*)fShare;

    uint tid = threadIdx.x;
    uint bx = n_index[blockIdx.x]; // N
    uint by = m_index[blockIdx.x]; // M

    uint tx = tid % 16;
    uint ty = tid / 16;
    assert(THREAD_SIZE_K % 16 == 0);
    uint k = tx * 4;

    uint ori_offsetA00 = (by * 32 + ty) * K + k;
    uint ori_offsetA16 = ori_offsetA00 + K * 16;
    uint ori_offsetB00 = (bx * 32 + ty) * K + k;
    uint ori_offsetB16 = ori_offsetB00 + K * 16;

    uint tid224 = tid & 224;
    uint storAB = (tx * 32 * 4 + ty + tx * 2) * 4;
    uint loadA = (((tid & 16) >> 3) | (tid & 1)) << 4;
    uint loadB = ((tid >> 1) & 7) << 4;
    loadA += (tid224 * 32) + (tid224 / 2);
    loadB += (tid224 * 32) + (tid224 / 2);

    // This keeps all prior logic outside of the loops.
    asm("mov.b32 %0, %0;" : "+r"(storAB) : );
    asm("mov.b32 %0, %0;" : "+r"(loadA)  : );
    asm("mov.b32 %0, %0;" : "+r"(loadB)  : );

    float regC[8][4];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++)
            regC[i][j] = 0.0f;

    for (int k_seq = 0; k_seq < (int)(K/64); k_seq++)
    {
        uint offsetA00 = ori_offsetA00 + 64 * k_seq;
        uint offsetA16 = ori_offsetA16 + 64 * k_seq;
        uint offsetB00 = ori_offsetB00 + 64 * k_seq;
        uint offsetB16 = ori_offsetB16 + 64 * k_seq;

        float4 a00 = {0}, a16 = {0};
        float4 b00 = {0}, b16 = {0};
        a00 = __ldg((const float4*)(add_ptr_u(A, offsetA00)));
        a16 = __ldg((const float4*)(add_ptr_u(A, offsetA16)));
        b00 = __ldg((const float4*)(add_ptr_u(B, offsetB00)));
        b16 = __ldg((const float4*)(add_ptr_u(B, offsetB16)));

        __syncthreads();

        *(float*)&bShare[storAB + (0*32 +  0 + 0*65*32)*4] = a00.x;
        *(float*)&bShare[storAB + (1*32 +  0 + 0*65*32)*4] = a00.y;
        *(float*)&bShare[storAB + (2*32 +  0 + 0*65*32)*4] = a00.z;
        *(float*)&bShare[storAB + (3*32 +  0 + 0*65*32)*4] = a00.w;
        *(float*)&bShare[storAB + (0*32 + 16 + 0*65*32)*4] = a16.x;
        *(float*)&bShare[storAB + (1*32 + 16 + 0*65*32)*4] = a16.y;
        *(float*)&bShare[storAB + (2*32 + 16 + 0*65*32)*4] = a16.z;
        *(float*)&bShare[storAB + (3*32 + 16 + 0*65*32)*4] = a16.w;

        *(float*)&bShare[storAB + (0*32 +  0 + 1*65*32)*4] = b00.x;
        *(float*)&bShare[storAB + (1*32 +  0 + 1*65*32)*4] = b00.y;
        *(float*)&bShare[storAB + (2*32 +  0 + 1*65*32)*4] = b00.z;
        *(float*)&bShare[storAB + (3*32 +  0 + 1*65*32)*4] = b00.w;
        *(float*)&bShare[storAB + (0*32 + 16 + 1*65*32)*4] = b16.x;
        *(float*)&bShare[storAB + (1*32 + 16 + 1*65*32)*4] = b16.y;
        *(float*)&bShare[storAB + (2*32 + 16 + 1*65*32)*4] = b16.z;
        *(float*)&bShare[storAB + (3*32 + 16 + 1*65*32)*4] = b16.w;
        __syncthreads();

        float regA[8], regB[4];
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            // fetch outer product data
            *(float4*)&regA[0] = *(float4*)&bShare[loadA + (32*j +  0)*4];
            *(float4*)&regA[4] = *(float4*)&bShare[loadA + (32*j + 16)*4];
            *(float4*)&regB[0] = *(float4*)&bShare[loadB + (32*j + 65*32)*4];

            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 4; j++)
                    regC[i][j] += regA[i] * regB[j];
        }
        #pragma unroll
        for (int j = 4; j < 8; j++)
        {
            *(float2*)&regA[0] = *(float2*)&bShare[loadA + (32*j +  0 + (j/4)*2)*4];
            *(float2*)&regA[2] = *(float2*)&bShare[loadA + (32*j +  2 + (j/4)*2)*4];
            *(float2*)&regA[4] = *(float2*)&bShare[loadA + (32*j + 16 + (j/4)*2)*4];
            *(float2*)&regA[6] = *(float2*)&bShare[loadA + (32*j + 18 + (j/4)*2)*4];
            *(float2*)&regB[0] = *(float2*)&bShare[loadB + (32*j +  0 + (j/4)*2 + 65*32)*4];
            *(float2*)&regB[2] = *(float2*)&bShare[loadB + (32*j +  2 + (j/4)*2 + 65*32)*4];

            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 4; j++)
                    regC[i][j] += regA[i] * regB[j];
        }
    }

    asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid)   :);
    asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(bx)   :);
    asm volatile ("mov.u32 %0, %ctaid.y;" : "=r"(by) :);

    ty = ((tid & 16) >> 3) + (tid & 1);
    tx = ((tid >> 1) & 7) + ((tid & 224) >> 2) + (ty << 2);

    uint storC = ty*32*8*4 + tx*4;

    tx = tid % 16;
    ty = tid / 16;

    uint readC = ty*32*8 + tx*2 + ((tid & 192)>>2);

    uint blk_index = block_index[blockIdx.x] / 2;
    uint intra_blk_index = block_index[blockIdx.x] % 2;
    C_val += 32 * 64 * blk_index + intra_blk_index * 32;
    C_val += ty * 64 + tx * 2;

    __syncthreads();
    *(float4*)&fShare[storC + 0*32*8] = *(float4*)regC[0];
    *(float4*)&fShare[storC + 1*32*8] = *(float4*)regC[1];
    *(float4*)&fShare[storC + 2*32*8] = *(float4*)regC[2];
    *(float4*)&fShare[storC + 3*32*8] = *(float4*)regC[3];
    __syncthreads();

    float2 c2[8];
    for (int i = 0; i < 8; i++)
        c2[i] = *(float2*)&fShare[readC + i*32];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            c2[i] = _add(c2[i], c2[i+j]);

    //-> store((bhalf2*)C, c2[0]);
    *(float2*)C_val = c2[0];

    __syncthreads();
    *(float4*)&fShare[storC + 0*32*8] = *(float4*)regC[4];
    *(float4*)&fShare[storC + 1*32*8] = *(float4*)regC[5];
    *(float4*)&fShare[storC + 2*32*8] = *(float4*)regC[6];
    *(float4*)&fShare[storC + 3*32*8] = *(float4*)regC[7];
    __syncthreads();

    for (int i = 0; i < 8; i++)
        c2[i] = *(float2*)&fShare[readC + i*32];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            c2[i] = _add(c2[i], c2[i+j]);

    C_val += 16 * 64;
    *(float2*)C_val = c2[0];

}

__global__ void SPARSE_SOFTMAX_32_64_32(
    float* C_val,
    int* C_val_mask,
    int* col_range_index,
    int* block_index){
    /*
    description:
    each row of blocks is dealt with a thread group
    each block is 32x32
    */
    C_val += SPARSE_VAL_SIZE*blockIdx.y;

    uint blk_row_idx = blockIdx.x;
    uint bm = threadIdx.x / 32;
    uint bn = threadIdx.x % 32;
    float regC = 0.0f;
    float regSum = 0.0f;
    int block_seq_start = col_range_index[blk_row_idx];
    int block_seq_end = col_range_index[blk_row_idx+1];

    for (int block_seq = block_seq_start; block_seq < block_seq_end; block_seq++) {
        uint blk_idx = block_index[block_seq];
        uint blk_index = blk_idx / 2;
        uint intra_blk_index = blk_idx % 2;
        uint index = 32*64*blk_index + 32*intra_blk_index + bm*64 + bn;
        regC = 0.0f;
        // regC = (float)C_val_mask[index]*C_val[index];
        if (C_val_mask[index] != 0) {
            regC = expf(C_val[index]);
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            regC += __shfl_down_sync(FULL_MASK, regC, offset);
        }
        regC = __shfl_sync(FULL_MASK, regC, 0);
        regSum += regC;
    }

    for (int block_seq = block_seq_start; block_seq < block_seq_end; block_seq++) {
        uint blk_idx = block_index[block_seq];
        uint blk_index = blk_idx / 2;
        uint intra_blk_index = blk_idx % 2;
        uint index = 32*64*blk_index + 32*intra_blk_index + bm*64 + bn;
        regC = 0.0f;
        if (C_val_mask[index] != 0) {
            regC = expf(C_val[index]);
        }
        if(regSum>0)
            C_val[index] = regC / regSum;

    }
}


__global__ void BLOCK_SPARSE_MATMUL_32_64_32(float* A_val, int* A_row, int* A_col, float* B, float* C){
    /*
    description:
    tiling k dimension
    tile size: 32x64x32
    smm_sd_d_nt: sparse matmul, sparse (MxK, along K, K major bcsr) x dense (KxN, along N, need transpose) -> dense (MxN, along N)
    block sparse matrix (block size: 32x64) X dense matrix -> dense matrix

    */
    const int BLOCK_SIZE_M = 32;  // 64
    const int BLOCK_SIZE_K = 64;  //8
    const int BLOCK_SIZE_N = 32;  //128
    const int THREAD_SIZE_K = 64;
    const int M = GLOBAL_M;
    const int K = GLOBAL_N;
    const int N = GLOBAL_K;

    A_val += SPARSE_VAL_SIZE*blockIdx.z;
    B += K*N*blockIdx.z;
    C += M*N*blockIdx.z;

    assert(blockDim.x % 32 == 0);
    uint n_warp = 8; // blockDim.x / 32
    assert(THREAD_SIZE_K % n_warp == 0);
    // THREAD_SIZE_K: one loop k
    assert(K % THREAD_SIZE_K == 0);

    assert(BLOCK_SIZE_M == BLOCK_SIZE_N);
    __shared__ float fShare[65 * 32 * 2];
    char* bShare = (char*)fShare;

    uint tid = threadIdx.x;
    uint bx = blockIdx.x; // N
    uint by = blockIdx.y; // M

    uint tx = tid % 16;
    uint ty = tid / 16;
    assert(THREAD_SIZE_K % 16 == 0);
    uint k = tx * 4;

    uint ori_offsetB00 = tid / (BLOCK_SIZE_N/4) * N + by * BLOCK_SIZE_N + (tid % (BLOCK_SIZE_N/4)) * 4;
    uint ori_offsetB16 = ori_offsetB00 + N * 32;
    uint storB = (tid * 4 + tid / (BLOCK_SIZE_N/4) / 4 *2) * 4; // (tid *4 + tid / (BLOCK_SIZE_N/4) / 4 * 2)*4

    // B is stored in sparse format, thus, should be dealt with differently
    uint offsetA00 = A_row[bx] * BLOCK_SIZE_M * BLOCK_SIZE_K + ty * BLOCK_SIZE_K + k;
    uint offsetA16 = offsetA00 + BLOCK_SIZE_K * 16;

    uint tid224 = tid & 224;
    uint storAB = (tx * 32 * 4 + ty + tx * 2) * 4;
    uint loadA = (((tid & 16) >> 3) | (tid & 1)) << 4;
    uint loadB = ((tid >> 1) & 7) << 4;
    loadA += (tid224 * 32) + (tid224 / 2);
    loadB += (tid224 * 32) + (tid224 / 2);

    // This keeps all prior logic outside of the loops.
    asm("mov.b32 %0, %0;" : "+r"(storAB) : );
    asm("mov.b32 %0, %0;" : "+r"(loadA)  : );
    asm("mov.b32 %0, %0;" : "+r"(loadB)  : );

    float regC[8][4];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++)
            regC[i][j] = 0.0f;

    // bx means in index of this thread block on N dimonsion
    // index_start and index_end is block index on column
    int index_start = A_row[bx], index_end = A_row[bx+1];
    for(int bcsr_col_idx = index_start; bcsr_col_idx < index_end; bcsr_col_idx += 1)
    {
        uint offsetB00 = ori_offsetB00 + 64 * A_col[bcsr_col_idx] * N;
        uint offsetB16 = ori_offsetB16 + 64 * A_col[bcsr_col_idx] * N;

        float4 a00 = {0}, a16 = {0};
        float4 b00 = {0}, b16 = {0};
        a00 = __ldg((const float4*)(add_ptr_u(A_val, offsetA00)));
        a16 = __ldg((const float4*)(add_ptr_u(A_val, offsetA16)));
        b00 = __ldg((const float4*)(add_ptr_u(B, offsetB00)));
        b16 = __ldg((const float4*)(add_ptr_u(B, offsetB16)));

        offsetA00 += BLOCK_SIZE_M * BLOCK_SIZE_K;
        offsetA16 += BLOCK_SIZE_M * BLOCK_SIZE_K;

        __syncthreads();

        *(float*)&bShare[storAB + (0*32 +  0 + 0*65*32)*4] = a00.x;
        *(float*)&bShare[storAB + (1*32 +  0 + 0*65*32)*4] = a00.y;
        *(float*)&bShare[storAB + (2*32 +  0 + 0*65*32)*4] = a00.z;
        *(float*)&bShare[storAB + (3*32 +  0 + 0*65*32)*4] = a00.w;
        *(float*)&bShare[storAB + (0*32 + 16 + 0*65*32)*4] = a16.x;
        *(float*)&bShare[storAB + (1*32 + 16 + 0*65*32)*4] = a16.y;
        *(float*)&bShare[storAB + (2*32 + 16 + 0*65*32)*4] = a16.z;
        *(float*)&bShare[storAB + (3*32 + 16 + 0*65*32)*4] = a16.w;

        *(float*)&bShare[storB + (1*65*32)*4] = b00.x;
        *(float*)&bShare[storB + (1*65*32 + 1)*4] = b00.y;
        *(float*)&bShare[storB + (1*65*32 + 2)*4] = b00.z;
        *(float*)&bShare[storB + (1*65*32 + 3)*4] = b00.w;
        *(float*)&bShare[storB + (32*32 + 8*2 + 1*65*32)*4] = b16.x;
        *(float*)&bShare[storB + (32*32 + 8*2 + 1*65*32 + 1)*4] = b16.y;
        *(float*)&bShare[storB + (32*32 + 8*2 + 1*65*32 + 2)*4] = b16.z;
        *(float*)&bShare[storB + (32*32 + 8*2 + 1*65*32 + 3)*4] = b16.w;
        __syncthreads();

        float regA[8], regB[4];
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            // fetch outer product data
            *(float4*)&regA[0] = *(float4*)&bShare[loadA + (32*j +  0)*4];
            *(float4*)&regA[4] = *(float4*)&bShare[loadA + (32*j + 16)*4];
            *(float4*)&regB[0] = *(float4*)&bShare[loadB + (32*j + 65*32)*4];

            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 4; j++)
                    regC[i][j] += regA[i] * regB[j];
        }
        #pragma unroll
        for (int j = 4; j < 8; j++)
        {
            *(float2*)&regA[0] = *(float2*)&bShare[loadA + (32*j +  0 + (j/4)*2)*4];
            *(float2*)&regA[2] = *(float2*)&bShare[loadA + (32*j +  2 + (j/4)*2)*4];
            *(float2*)&regA[4] = *(float2*)&bShare[loadA + (32*j + 16 + (j/4)*2)*4];
            *(float2*)&regA[6] = *(float2*)&bShare[loadA + (32*j + 18 + (j/4)*2)*4];
            *(float2*)&regB[0] = *(float2*)&bShare[loadB + (32*j +  0 + (j/4)*2 + 65*32)*4];
            *(float2*)&regB[2] = *(float2*)&bShare[loadB + (32*j +  2 + (j/4)*2 + 65*32)*4];

            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 4; j++)
                    regC[i][j] += regA[i] * regB[j];
        }
    }

    asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid)   :);
    asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(bx)   :);
    asm volatile ("mov.u32 %0, %ctaid.y;" : "=r"(by) :);

    ty = ((tid & 16) >> 3) + (tid & 1);
    tx = ((tid >> 1) & 7) + ((tid & 224) >> 2) + (ty << 2);

    uint storC = ty*32*8*4 + tx*4;

    tx = tid % 16;
    ty = tid / 16;

    uint readC = ty*32*8 + tx*2 + ((tid & 192)>>2);

    // C should be row major
    C += (bx * BLOCK_SIZE_M + ty) * N + (by * BLOCK_SIZE_N + tx * 2);

    __syncthreads();
    *(float4*)&fShare[storC + 0*32*8] = *(float4*)regC[0];
    *(float4*)&fShare[storC + 1*32*8] = *(float4*)regC[1];
    *(float4*)&fShare[storC + 2*32*8] = *(float4*)regC[2];
    *(float4*)&fShare[storC + 3*32*8] = *(float4*)regC[3];
    __syncthreads();

    float2 c2[8];
    for (int i = 0; i < 8; i++)
        c2[i] = *(float2*)&fShare[readC + i*32];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            c2[i] = _add(c2[i], c2[i+j]);

    *(float2*)C = c2[0];

    __syncthreads();
    *(float4*)&fShare[storC + 0*32*8] = *(float4*)regC[4];
    *(float4*)&fShare[storC + 1*32*8] = *(float4*)regC[5];
    *(float4*)&fShare[storC + 2*32*8] = *(float4*)regC[6];
    *(float4*)&fShare[storC + 3*32*8] = *(float4*)regC[7];
    __syncthreads();

    for (int i = 0; i < 8; i++)
        c2[i] = *(float2*)&fShare[readC + i*32];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            c2[i] = _add(c2[i], c2[i+j]);

    C += 16 * N;
    *(float2*)C = c2[0];

}

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

void forward_function(float* Q, float* K, float* V,
                    int* d_m_index, int* d_n_index, int* d_block_index,
                    float* val, int* row_ptr, int* col_idx, int* val_mask, int* col_range_index,
                    int batch_size, float* output)
{
    // already set to zero outside, no need to memset here
    //cudaMemset((void*)val, 0, (SPARSE_VAL_SIZE * HEAD_NUM) * batch_size);
    const dim3 dimBlock(256);
    const dim3 dimGrid(SMALL_BLOCK_NUM, HEAD_NUM*batch_size);
    BLOCK_SPARSE_MATMUL_OUT_32_64_32<<<dimGrid, dimBlock>>>(
        Q,
        K,
        val,
        d_m_index,
        d_n_index,
        d_block_index
    );

    const dim3 softmax_dimBlock(32*32);
    const dim3 softmax_dimGrid(GLOBAL_M/32, HEAD_NUM*batch_size);
    SPARSE_SOFTMAX_32_64_32<<<softmax_dimGrid, softmax_dimBlock>>>(
        val,
        val_mask,
        col_range_index,
        d_block_index
    );

    const dim3 out_dimBlock(256);
    const dim3 out_dimGrid(GLOBAL_M/32, GLOBAL_K/32, HEAD_NUM*batch_size);
    BLOCK_SPARSE_MATMUL_32_64_32<<<out_dimGrid, out_dimBlock>>>(
        val,
        row_ptr,
        col_idx,
        V,
        output
    );
}



__global__ void GRAD_V_KERNEL(int* A_row, int* A_col, int* subblock_index, float* A_val, float* B, float* C){
    /*
    description:
    tiling k dimension
    tile size: 32x64x32
    smm_sd_d_nt: sparse matmul, sparse (MxK, along K, K major bcsr) x dense (KxN, along N, need transpose) -> dense (MxN, along N)
    block sparse matrix (block size: 32x64) X dense matrix -> dense matrix

    */
    const int BLOCK_SIZE_M = 32;  // 64
    const int BLOCK_SIZE_K = 64;  //8
    const int BLOCK_SIZE_N = 32;  //128
    const int THREAD_SIZE_K = 64;
    // FIXME: M=GLOBAL_N
    const int M = GLOBAL_M;
    const int K = GLOBAL_M;
    const int N = GLOBAL_K;
    // TODO: Double check
    A_val += SPARSE_VAL_SIZE*blockIdx.z;
    B += K*N*blockIdx.z;
    C += M*N*blockIdx.z;

    assert(blockDim.x % 32 == 0);
    uint n_warp = 8; // blockDim.x / 32
    assert(THREAD_SIZE_K % n_warp == 0);
    // THREAD_SIZE_K: one loop k
    assert(K % THREAD_SIZE_K == 0);

    assert(BLOCK_SIZE_M == BLOCK_SIZE_N);
    __shared__ float fShare[65 * 32 * 2];
    char* bShare = (char*)fShare;

    uint tid = threadIdx.x;
    uint bx = blockIdx.x; // N
    uint by = blockIdx.y; // M

    uint tx = tid % 16;
    uint ty = tid / 16;
    assert(THREAD_SIZE_K % 16 == 0);
    uint k = tx * 4;

    uint ori_offsetB00 = tid / (BLOCK_SIZE_N/4) * N + by * BLOCK_SIZE_N + (tid % (BLOCK_SIZE_N/4)) * 4;
    uint ori_offsetB16 = ori_offsetB00 + N * 32;
    uint storB = (tid * 4 + tid / (BLOCK_SIZE_N/4) / 4 *2) * 4; // (tid *4 + tid / (BLOCK_SIZE_N/4) / 4 * 2)*4

    // B is stored in sparse format, thus, should be dealt with differently
    // uint offsetA00 = A_row[bx] * BLOCK_SIZE_M * BLOCK_SIZE_K + ty * BLOCK_SIZE_K + k;
    // uint offsetA16 = offsetA00 + BLOCK_SIZE_K * 16;

    uint tid224 = tid & 224;
    // uint storAB = (tx * 32 * 4 + ty + tx * 2) * 4;
    uint storA = ((32*(tx/8)+ty) * 32 + (tx%8) * 4  + 2*((32*(tx/8)+ty)/4)) * 4;
    uint loadA = (((tid & 16) >> 3) | (tid & 1)) << 4;
    uint loadB = ((tid >> 1) & 7) << 4;
    loadA += (tid224 * 32) + (tid224 / 2);
    loadB += (tid224 * 32) + (tid224 / 2);

    // This keeps all prior logic outside of the loops.
    asm("mov.b32 %0, %0;" : "+r"(storA) : );
    asm("mov.b32 %0, %0;" : "+r"(loadA)  : );
    asm("mov.b32 %0, %0;" : "+r"(loadB)  : );

    float regC[8][4];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++)
            regC[i][j] = 0.0f;

    // bx means in index of this thread block on N dimonsion
    // index_start and index_end is block index on column
    int index_start = A_row[bx], index_end = A_row[bx+1];
    for(int bcsr_col_idx = index_start; bcsr_col_idx < index_end; bcsr_col_idx += 1)
    {

        uint offsetA00 = subblock_index[2*bcsr_col_idx+tx/8] + ty * BLOCK_SIZE_K + (tx % 8) * 4;
        uint offsetA16 = offsetA00 + BLOCK_SIZE_K * 16;
        uint offsetB00 = ori_offsetB00 + 64 * A_col[bcsr_col_idx] * N;
        uint offsetB16 = ori_offsetB16 + 64 * A_col[bcsr_col_idx] * N;

        float4 a00 = {0}, a16 = {0};
        float4 b00 = {0}, b16 = {0};
        a00 = __ldg((const float4*)(add_ptr_u(A_val, offsetA00)));
        a16 = __ldg((const float4*)(add_ptr_u(A_val, offsetA16)));
        b00 = __ldg((const float4*)(add_ptr_u(B, offsetB00)));
        b16 = __ldg((const float4*)(add_ptr_u(B, offsetB16)));

        // offsetA00 += BLOCK_SIZE_M * BLOCK_SIZE_K;
        // offsetA16 += BLOCK_SIZE_M * BLOCK_SIZE_K;

        __syncthreads();
        *(float2*)&bShare[storA] = *(float2*)&a00.x;
        *(float2*)&bShare[storA + 8] = *(float2*)&a00.z;
        *(float2*)&bShare[storA + (16*32+4*2)*4] = *(float2*)&a16.x;
        *(float2*)&bShare[storA + (16*32+4*2)*4 + 8] = *(float2*)&a16.z;

        // *(float*)&bShare[storAB + (0*32 +  0 + 0*65*32)*4] = a00.x;
        // *(float*)&bShare[storAB + (1*32 +  0 + 0*65*32)*4] = a00.y;
        // *(float*)&bShare[storAB + (2*32 +  0 + 0*65*32)*4] = a00.z;
        // *(float*)&bShare[storAB + (3*32 +  0 + 0*65*32)*4] = a00.w;
        // *(float*)&bShare[storAB + (0*32 + 16 + 0*65*32)*4] = a16.x;
        // *(float*)&bShare[storAB + (1*32 + 16 + 0*65*32)*4] = a16.y;
        // *(float*)&bShare[storAB + (2*32 + 16 + 0*65*32)*4] = a16.z;
        // *(float*)&bShare[storAB + (3*32 + 16 + 0*65*32)*4] = a16.w;

        *(float*)&bShare[storB + (1*65*32)*4] = b00.x;
        *(float*)&bShare[storB + (1*65*32 + 1)*4] = b00.y;
        *(float*)&bShare[storB + (1*65*32 + 2)*4] = b00.z;
        *(float*)&bShare[storB + (1*65*32 + 3)*4] = b00.w;
        *(float*)&bShare[storB + (32*32 + 8*2 + 1*65*32)*4] = b16.x;
        *(float*)&bShare[storB + (32*32 + 8*2 + 1*65*32 + 1)*4] = b16.y;
        *(float*)&bShare[storB + (32*32 + 8*2 + 1*65*32 + 2)*4] = b16.z;
        *(float*)&bShare[storB + (32*32 + 8*2 + 1*65*32 + 3)*4] = b16.w;
        __syncthreads();

        float regA[8], regB[4];
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            // fetch outer product data
            *(float4*)&regA[0] = *(float4*)&bShare[loadA + (32*j +  0)*4];
            *(float4*)&regA[4] = *(float4*)&bShare[loadA + (32*j + 16)*4];
            *(float4*)&regB[0] = *(float4*)&bShare[loadB + (32*j + 65*32)*4];

            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 4; j++)
                    regC[i][j] += regA[i] * regB[j];
        }
        #pragma unroll
        for (int j = 4; j < 8; j++)
        {
            *(float2*)&regA[0] = *(float2*)&bShare[loadA + (32*j +  0 + (j/4)*2)*4];
            *(float2*)&regA[2] = *(float2*)&bShare[loadA + (32*j +  2 + (j/4)*2)*4];
            *(float2*)&regA[4] = *(float2*)&bShare[loadA + (32*j + 16 + (j/4)*2)*4];
            *(float2*)&regA[6] = *(float2*)&bShare[loadA + (32*j + 18 + (j/4)*2)*4];
            *(float2*)&regB[0] = *(float2*)&bShare[loadB + (32*j +  0 + (j/4)*2 + 65*32)*4];
            *(float2*)&regB[2] = *(float2*)&bShare[loadB + (32*j +  2 + (j/4)*2 + 65*32)*4];

            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 4; j++)
                    regC[i][j] += regA[i] * regB[j];
        }
    }

    asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid)   :);
    asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(bx)   :);
    asm volatile ("mov.u32 %0, %ctaid.y;" : "=r"(by) :);

    ty = ((tid & 16) >> 3) + (tid & 1);
    tx = ((tid >> 1) & 7) + ((tid & 224) >> 2) + (ty << 2);

    uint storC = ty*32*8*4 + tx*4;

    tx = tid % 16;
    ty = tid / 16;

    uint readC = ty*32*8 + tx*2 + ((tid & 192)>>2);

    // C should be row major
    C += (bx * BLOCK_SIZE_M + ty) * N + (by * BLOCK_SIZE_N + tx * 2);

    __syncthreads();
    *(float4*)&fShare[storC + 0*32*8] = *(float4*)regC[0];
    *(float4*)&fShare[storC + 1*32*8] = *(float4*)regC[1];
    *(float4*)&fShare[storC + 2*32*8] = *(float4*)regC[2];
    *(float4*)&fShare[storC + 3*32*8] = *(float4*)regC[3];
    __syncthreads();

    float2 c2[8];
    for (int i = 0; i < 8; i++)
        c2[i] = *(float2*)&fShare[readC + i*32];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            c2[i] = _add(c2[i], c2[i+j]);

    *(float2*)C = c2[0];

    __syncthreads();
    *(float4*)&fShare[storC + 0*32*8] = *(float4*)regC[4];
    *(float4*)&fShare[storC + 1*32*8] = *(float4*)regC[5];
    *(float4*)&fShare[storC + 2*32*8] = *(float4*)regC[6];
    *(float4*)&fShare[storC + 3*32*8] = *(float4*)regC[7];
    __syncthreads();

    for (int i = 0; i < 8; i++)
        c2[i] = *(float2*)&fShare[readC + i*32];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            c2[i] = _add(c2[i], c2[i+j]);

    C += 16 * N;
    *(float2*)C = c2[0];

}


__global__ void GRAD_ATTEN_KERNEL(
    float* A,
    float* B,
    float* C_val,
    int* m_index,
    int* n_index,
    int* block_index){
    /*
    description:
    tiling k dimension
    smm_dd_s_nt: sparse matmul, dense (MxK, along K) x dense (NxK, along k) -> sparse (MxN, along N)
    the output sparse is block size 32x32, the blocks will be written to bcsr 32x64
    */
    const int BLOCK_SIZE_M = 32;  // 64
    const int BLOCK_SIZE_K = 64;  //8
    const int BLOCK_SIZE_N = 32;  //128
    const int THREAD_SIZE_K = 64;
    const int M = GLOBAL_M;
    const int K = GLOBAL_K;
    const int N = GLOBAL_N;

    A += M*K*blockIdx.y;
    B += K*N*blockIdx.y;
    C_val += SPARSE_VAL_SIZE*blockIdx.y;

    assert(blockDim.x % 32 == 0);
    uint n_warp = 8; // blockDim.x / 32
    assert(THREAD_SIZE_K % n_warp == 0);
    // THREAD_SIZE_K: one loop k
    assert(K % THREAD_SIZE_K == 0);

    assert(BLOCK_SIZE_M == BLOCK_SIZE_N);
    __shared__ float fShare[65 * 32 * 2];
    char* bShare = (char*)fShare;

    uint tid = threadIdx.x;
    uint bx = n_index[blockIdx.x]; // N
    uint by = m_index[blockIdx.x]; // M

    uint tx = tid % 16;
    uint ty = tid / 16;
    assert(THREAD_SIZE_K % 16 == 0);
    uint k = tx * 4;

    uint ori_offsetA00 = (by * 32 + ty) * K + k;
    uint ori_offsetA16 = ori_offsetA00 + K * 16;
    uint ori_offsetB00 = (bx * 32 + ty) * K + k;
    uint ori_offsetB16 = ori_offsetB00 + K * 16;

    uint tid224 = tid & 224;
    uint storAB = (tx * 32 * 4 + ty + tx * 2) * 4;
    uint loadA = (((tid & 16) >> 3) | (tid & 1)) << 4;
    uint loadB = ((tid >> 1) & 7) << 4;
    loadA += (tid224 * 32) + (tid224 / 2);
    loadB += (tid224 * 32) + (tid224 / 2);

    // This keeps all prior logic outside of the loops.
    asm("mov.b32 %0, %0;" : "+r"(storAB) : );
    asm("mov.b32 %0, %0;" : "+r"(loadA)  : );
    asm("mov.b32 %0, %0;" : "+r"(loadB)  : );

    float regC[8][4];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++)
            regC[i][j] = 0.0f;

    for (int k_seq = 0; k_seq < (int)(K/64); k_seq++)
    {
        uint offsetA00 = ori_offsetA00 + 64 * k_seq;
        uint offsetA16 = ori_offsetA16 + 64 * k_seq;
        uint offsetB00 = ori_offsetB00 + 64 * k_seq;
        uint offsetB16 = ori_offsetB16 + 64 * k_seq;

        float4 a00 = {0}, a16 = {0};
        float4 b00 = {0}, b16 = {0};
        a00 = __ldg((const float4*)(add_ptr_u(A, offsetA00)));
        a16 = __ldg((const float4*)(add_ptr_u(A, offsetA16)));
        b00 = __ldg((const float4*)(add_ptr_u(B, offsetB00)));
        b16 = __ldg((const float4*)(add_ptr_u(B, offsetB16)));

        __syncthreads();

        *(float*)&bShare[storAB + (0*32 +  0 + 0*65*32)*4] = a00.x;
        *(float*)&bShare[storAB + (1*32 +  0 + 0*65*32)*4] = a00.y;
        *(float*)&bShare[storAB + (2*32 +  0 + 0*65*32)*4] = a00.z;
        *(float*)&bShare[storAB + (3*32 +  0 + 0*65*32)*4] = a00.w;
        *(float*)&bShare[storAB + (0*32 + 16 + 0*65*32)*4] = a16.x;
        *(float*)&bShare[storAB + (1*32 + 16 + 0*65*32)*4] = a16.y;
        *(float*)&bShare[storAB + (2*32 + 16 + 0*65*32)*4] = a16.z;
        *(float*)&bShare[storAB + (3*32 + 16 + 0*65*32)*4] = a16.w;

        *(float*)&bShare[storAB + (0*32 +  0 + 1*65*32)*4] = b00.x;
        *(float*)&bShare[storAB + (1*32 +  0 + 1*65*32)*4] = b00.y;
        *(float*)&bShare[storAB + (2*32 +  0 + 1*65*32)*4] = b00.z;
        *(float*)&bShare[storAB + (3*32 +  0 + 1*65*32)*4] = b00.w;
        *(float*)&bShare[storAB + (0*32 + 16 + 1*65*32)*4] = b16.x;
        *(float*)&bShare[storAB + (1*32 + 16 + 1*65*32)*4] = b16.y;
        *(float*)&bShare[storAB + (2*32 + 16 + 1*65*32)*4] = b16.z;
        *(float*)&bShare[storAB + (3*32 + 16 + 1*65*32)*4] = b16.w;
        __syncthreads();

        float regA[8], regB[4];
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            // fetch outer product data
            *(float4*)&regA[0] = *(float4*)&bShare[loadA + (32*j +  0)*4];
            *(float4*)&regA[4] = *(float4*)&bShare[loadA + (32*j + 16)*4];
            *(float4*)&regB[0] = *(float4*)&bShare[loadB + (32*j + 65*32)*4];

            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 4; j++)
                    regC[i][j] += regA[i] * regB[j];
        }
        #pragma unroll
        for (int j = 4; j < 8; j++)
        {
            *(float2*)&regA[0] = *(float2*)&bShare[loadA + (32*j +  0 + (j/4)*2)*4];
            *(float2*)&regA[2] = *(float2*)&bShare[loadA + (32*j +  2 + (j/4)*2)*4];
            *(float2*)&regA[4] = *(float2*)&bShare[loadA + (32*j + 16 + (j/4)*2)*4];
            *(float2*)&regA[6] = *(float2*)&bShare[loadA + (32*j + 18 + (j/4)*2)*4];
            *(float2*)&regB[0] = *(float2*)&bShare[loadB + (32*j +  0 + (j/4)*2 + 65*32)*4];
            *(float2*)&regB[2] = *(float2*)&bShare[loadB + (32*j +  2 + (j/4)*2 + 65*32)*4];

            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 4; j++)
                    regC[i][j] += regA[i] * regB[j];
        }
    }

    asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid)   :);
    asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(bx)   :);
    asm volatile ("mov.u32 %0, %ctaid.y;" : "=r"(by) :);

    ty = ((tid & 16) >> 3) + (tid & 1);
    tx = ((tid >> 1) & 7) + ((tid & 224) >> 2) + (ty << 2);

    uint storC = ty*32*8*4 + tx*4;

    tx = tid % 16;
    ty = tid / 16;

    uint readC = ty*32*8 + tx*2 + ((tid & 192)>>2);

    uint blk_index = block_index[blockIdx.x] / 2;
    uint intra_blk_index = block_index[blockIdx.x] % 2;
    C_val += 32 * 64 * blk_index + intra_blk_index * 32;
    C_val += ty * 64 + tx * 2;

    __syncthreads();
    *(float4*)&fShare[storC + 0*32*8] = *(float4*)regC[0];
    *(float4*)&fShare[storC + 1*32*8] = *(float4*)regC[1];
    *(float4*)&fShare[storC + 2*32*8] = *(float4*)regC[2];
    *(float4*)&fShare[storC + 3*32*8] = *(float4*)regC[3];
    __syncthreads();

    float2 c2[8];
    for (int i = 0; i < 8; i++)
        c2[i] = *(float2*)&fShare[readC + i*32];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            c2[i] = _add(c2[i], c2[i+j]);

    //-> store((bhalf2*)C, c2[0]);
    *(float2*)C_val = c2[0];

    __syncthreads();
    *(float4*)&fShare[storC + 0*32*8] = *(float4*)regC[4];
    *(float4*)&fShare[storC + 1*32*8] = *(float4*)regC[5];
    *(float4*)&fShare[storC + 2*32*8] = *(float4*)regC[6];
    *(float4*)&fShare[storC + 3*32*8] = *(float4*)regC[7];
    __syncthreads();

    for (int i = 0; i < 8; i++)
        c2[i] = *(float2*)&fShare[readC + i*32];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            c2[i] = _add(c2[i], c2[i+j]);

    C_val += 16 * 64;
    *(float2*)C_val = c2[0];

}


__global__ void GRAD_SOFTMAX_KERNEL(float * C_val, float *val_grad, float* out_grad, int *m_index, int *col_range_index, int *block_index)
{
    C_val += SPARSE_VAL_SIZE * blockIdx.y;
    val_grad += SPARSE_VAL_SIZE * blockIdx.y;
    out_grad += SPARSE_VAL_SIZE * blockIdx.y;
    
    uint shared_load_m = threadIdx.x / 8;
    uint shared_load_n = threadIdx.x % 8;
    uint tm = threadIdx.x / 32;
    uint tn = threadIdx.x % 32;
    uint block_row_id = m_index[blockIdx.x];
    int block_seq_start = col_range_index[block_row_id];
    int block_seq_end = col_range_index[block_row_id+1];
    __shared__ float val_s [32*33];
    __shared__ float grad_s [32*33];
    float regSum[4] = {0};
    float regOut[4] = {0};
    uint out_sub_blk_id = block_index[blockIdx.x];
    // load the grad[i] and out[i] into the register
    // grad[i] = sum(-grad[j]*out[j]*out[i] if i!=j else grad[j]*(1-out[j])*out[j])
    for(int i=tm;i<SOFTMAX_BLOCK_SIZE_M;i+=8){
        uint _tmp_index = 32*64*(out_sub_blk_id/2)+ 32*(out_sub_blk_id%2) + i * 64 + tn;

        regOut[i/8] = C_val[_tmp_index];
        regSum[i/8] = val_grad[_tmp_index]*regOut[i/8];

    }
    // printf("ThreadIdx:%d out_val:%f gradient:%f\\n", threadIdx.x, regOut[0], regSum[0]);
    __syncthreads();

    for(int block_seq=block_seq_start; block_seq < block_seq_end; block_seq++){

        uint sub_block_idx = block_index[block_seq];
        uint global_blk_idx = sub_block_idx/2;
        uint intra_blk_index = sub_block_idx%2;
        uint offset_val = 32*64*global_blk_idx + 32*intra_blk_index + shared_load_m*64 + shared_load_n*4;
        uint offset_grad = offset_val;

        uint store_idx = shared_load_m * 32 + shared_load_m + shared_load_n *4;
        float4 _val4  = {0}, _grad4 = {0};
        _val4 = __ldg((const float4*)(add_ptr_u(C_val, offset_val)));
        _grad4 = __ldg((const float4*)(add_ptr_u(val_grad, offset_grad)));
        __syncthreads();
        *(float*)&val_s[store_idx] = _val4.x;
        *(float*)&val_s[store_idx+1] = _val4.y;
        *(float*)&val_s[store_idx+2] = _val4.z;
        *(float*)&val_s[store_idx+3] = _val4.w;

        *(float*)&grad_s[store_idx+0] = _grad4.x;
        *(float*)&grad_s[store_idx+1] = _grad4.y;
        *(float*)&grad_s[store_idx+2] = _grad4.z;
        *(float*)&grad_s[store_idx+3] = _grad4.w;
        __syncthreads();
        for(int i = tm; i<SOFTMAX_BLOCK_SIZE_M; i+=8){
            for(int j=0; j<SOFTMAX_BLOCK_SIZE_N; j ++){
                uint shared_grad_offset =  i * 32 + i + j;
                regSum[i/8] -= regOut[i/8] * grad_s[shared_grad_offset] * val_s[shared_grad_offset];
            }

        }
        __syncthreads();
    }
    
    // write the gradients back to global memory
    for(int i=tm;i<SOFTMAX_BLOCK_SIZE_M;i+=8){
        uint _tmp_index = 32*64*(out_sub_blk_id/2)+ 32*(out_sub_blk_id%2) + i * 64 + tn;
        out_grad[_tmp_index] = regSum[i/8];
    }
    __syncthreads();

}

__global__ void GRAD_Q_KERNEL(float* A_val, int* A_row, int* A_col, float*B, float*C)
{
    /*
    description:
    tiling k dimension
    tile size: 32x64x32
    smm_sd_d_nt: sparse matmul, sparse (MxK, along K, K major bcsr) x dense (KxN, along N, need transpose) -> dense (MxN, along N)
    block sparse matrix (block size: 32x64) X dense matrix -> dense matrix

    */
    const int BLOCK_SIZE_M = 32;  // 64
    const int BLOCK_SIZE_K = 64;  //8
    const int BLOCK_SIZE_N = 32;  //128
    const int THREAD_SIZE_K = 64;
    const int M = GLOBAL_M;
    const int K = GLOBAL_N;
    const int N = GLOBAL_K;

    A_val += SPARSE_VAL_SIZE*blockIdx.z;
    B += K*N*blockIdx.z;
    C += M*N*blockIdx.z;

    assert(blockDim.x % 32 == 0);
    uint n_warp = 8; // blockDim.x / 32
    assert(THREAD_SIZE_K % n_warp == 0);
    // THREAD_SIZE_K: one loop k
    assert(K % THREAD_SIZE_K == 0);

    assert(BLOCK_SIZE_M == BLOCK_SIZE_N);
    __shared__ float fShare[65 * 32 * 2];
    char* bShare = (char*)fShare;

    uint tid = threadIdx.x;
    uint bx = blockIdx.x; // N
    uint by = blockIdx.y; // M

    uint tx = tid % 16;
    uint ty = tid / 16;
    assert(THREAD_SIZE_K % 16 == 0);
    uint k = tx * 4;

    uint ori_offsetB00 = tid / (BLOCK_SIZE_N/4) * N + by * BLOCK_SIZE_N + (tid % (BLOCK_SIZE_N/4)) * 4;
    uint ori_offsetB16 = ori_offsetB00 + N * 32;
    uint storB = (tid * 4 + tid / (BLOCK_SIZE_N/4) / 4 *2) * 4; // (tid *4 + tid / (BLOCK_SIZE_N/4) / 4 * 2)*4

    // B is stored in sparse format, thus, should be dealt with differently
    uint offsetA00 = A_row[bx] * BLOCK_SIZE_M * BLOCK_SIZE_K + ty * BLOCK_SIZE_K + k;
    uint offsetA16 = offsetA00 + BLOCK_SIZE_K * 16;

    uint tid224 = tid & 224;
    uint storAB = (tx * 32 * 4 + ty + tx * 2) * 4;
    uint loadA = (((tid & 16) >> 3) | (tid & 1)) << 4;
    uint loadB = ((tid >> 1) & 7) << 4;
    loadA += (tid224 * 32) + (tid224 / 2);
    loadB += (tid224 * 32) + (tid224 / 2);

    // This keeps all prior logic outside of the loops.
    asm("mov.b32 %0, %0;" : "+r"(storAB) : );
    asm("mov.b32 %0, %0;" : "+r"(loadA)  : );
    asm("mov.b32 %0, %0;" : "+r"(loadB)  : );

    float regC[8][4];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++)
            regC[i][j] = 0.0f;

    // bx means in index of this thread block on N dimonsion
    // index_start and index_end is block index on column
    int index_start = A_row[bx], index_end = A_row[bx+1];
    for(int bcsr_col_idx = index_start; bcsr_col_idx < index_end; bcsr_col_idx += 1)
    {
        uint offsetB00 = ori_offsetB00 + 64 * A_col[bcsr_col_idx] * N;
        uint offsetB16 = ori_offsetB16 + 64 * A_col[bcsr_col_idx] * N;

        float4 a00 = {0}, a16 = {0};
        float4 b00 = {0}, b16 = {0};
        a00 = __ldg((const float4*)(add_ptr_u(A_val, offsetA00)));
        a16 = __ldg((const float4*)(add_ptr_u(A_val, offsetA16)));
        b00 = __ldg((const float4*)(add_ptr_u(B, offsetB00)));
        b16 = __ldg((const float4*)(add_ptr_u(B, offsetB16)));

        offsetA00 += BLOCK_SIZE_M * BLOCK_SIZE_K;
        offsetA16 += BLOCK_SIZE_M * BLOCK_SIZE_K;

        __syncthreads();

        *(float*)&bShare[storAB + (0*32 +  0 + 0*65*32)*4] = a00.x;
        *(float*)&bShare[storAB + (1*32 +  0 + 0*65*32)*4] = a00.y;
        *(float*)&bShare[storAB + (2*32 +  0 + 0*65*32)*4] = a00.z;
        *(float*)&bShare[storAB + (3*32 +  0 + 0*65*32)*4] = a00.w;
        *(float*)&bShare[storAB + (0*32 + 16 + 0*65*32)*4] = a16.x;
        *(float*)&bShare[storAB + (1*32 + 16 + 0*65*32)*4] = a16.y;
        *(float*)&bShare[storAB + (2*32 + 16 + 0*65*32)*4] = a16.z;
        *(float*)&bShare[storAB + (3*32 + 16 + 0*65*32)*4] = a16.w;

        *(float*)&bShare[storB + (1*65*32)*4] = b00.x;
        *(float*)&bShare[storB + (1*65*32 + 1)*4] = b00.y;
        *(float*)&bShare[storB + (1*65*32 + 2)*4] = b00.z;
        *(float*)&bShare[storB + (1*65*32 + 3)*4] = b00.w;
        *(float*)&bShare[storB + (32*32 + 8*2 + 1*65*32)*4] = b16.x;
        *(float*)&bShare[storB + (32*32 + 8*2 + 1*65*32 + 1)*4] = b16.y;
        *(float*)&bShare[storB + (32*32 + 8*2 + 1*65*32 + 2)*4] = b16.z;
        *(float*)&bShare[storB + (32*32 + 8*2 + 1*65*32 + 3)*4] = b16.w;
        __syncthreads();

        float regA[8], regB[4];
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            // fetch outer product data
            *(float4*)&regA[0] = *(float4*)&bShare[loadA + (32*j +  0)*4];
            *(float4*)&regA[4] = *(float4*)&bShare[loadA + (32*j + 16)*4];
            *(float4*)&regB[0] = *(float4*)&bShare[loadB + (32*j + 65*32)*4];

            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 4; j++)
                    regC[i][j] += regA[i] * regB[j];
        }
        #pragma unroll
        for (int j = 4; j < 8; j++)
        {
            *(float2*)&regA[0] = *(float2*)&bShare[loadA + (32*j +  0 + (j/4)*2)*4];
            *(float2*)&regA[2] = *(float2*)&bShare[loadA + (32*j +  2 + (j/4)*2)*4];
            *(float2*)&regA[4] = *(float2*)&bShare[loadA + (32*j + 16 + (j/4)*2)*4];
            *(float2*)&regA[6] = *(float2*)&bShare[loadA + (32*j + 18 + (j/4)*2)*4];
            *(float2*)&regB[0] = *(float2*)&bShare[loadB + (32*j +  0 + (j/4)*2 + 65*32)*4];
            *(float2*)&regB[2] = *(float2*)&bShare[loadB + (32*j +  2 + (j/4)*2 + 65*32)*4];

            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 4; j++)
                    regC[i][j] += regA[i] * regB[j];
        }
    }

    asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid)   :);
    asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(bx)   :);
    asm volatile ("mov.u32 %0, %ctaid.y;" : "=r"(by) :);

    ty = ((tid & 16) >> 3) + (tid & 1);
    tx = ((tid >> 1) & 7) + ((tid & 224) >> 2) + (ty << 2);

    uint storC = ty*32*8*4 + tx*4;

    tx = tid % 16;
    ty = tid / 16;

    uint readC = ty*32*8 + tx*2 + ((tid & 192)>>2);

    // C should be row major
    C += (bx * BLOCK_SIZE_M + ty) * N + (by * BLOCK_SIZE_N + tx * 2);

    __syncthreads();
    *(float4*)&fShare[storC + 0*32*8] = *(float4*)regC[0];
    *(float4*)&fShare[storC + 1*32*8] = *(float4*)regC[1];
    *(float4*)&fShare[storC + 2*32*8] = *(float4*)regC[2];
    *(float4*)&fShare[storC + 3*32*8] = *(float4*)regC[3];
    __syncthreads();

    float2 c2[8];
    for (int i = 0; i < 8; i++)
        c2[i] = *(float2*)&fShare[readC + i*32];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            c2[i] = _add(c2[i], c2[i+j]);

    *(float2*)C = c2[0];

    __syncthreads();
    *(float4*)&fShare[storC + 0*32*8] = *(float4*)regC[4];
    *(float4*)&fShare[storC + 1*32*8] = *(float4*)regC[5];
    *(float4*)&fShare[storC + 2*32*8] = *(float4*)regC[6];
    *(float4*)&fShare[storC + 3*32*8] = *(float4*)regC[7];
    __syncthreads();

    for (int i = 0; i < 8; i++)
        c2[i] = *(float2*)&fShare[readC + i*32];

    // Tree reduce
    for (int j = 4; j > 0; j >>= 1)
        for (int i = 0; i < j; i++)
            c2[i] = _add(c2[i], c2[i+j]);

    C += 16 * N;
    *(float2*)C = c2[0];

    
}
void backward_function(float* in_grad, float* Q, float* K, float* V,
    int* gradv_row_idx, int* gradv_col_idx, int * gradv_subblock_idx, float* val,
    int* m_index, int * n_index, int* block_index, int * col_range_index, int* row_ptr, int* col_idx, int batch_size,
    float* Q_grad, float * K_grad, float* V_grad, float* val_grad, float* qxk_grad)
{
    /*
        Compute the gradient of the Q, K, V.
        A * B = C
            |  backward()
            V
        Grad_A = Grad_C * B^T
        Grad_B = A^T * Grad_C
    The gradient calculation formulars are as following:
    
    Grad_V = Sparse*dense kernel: (row_ptr, col_idx, val)^T * in_grad 
    Grad_val = output sparse kernel(sparse index row_ptr, col_idx, val): in_grad * V^T 
    Grad_Q = Grad_softmax * K^T
    Grad_K = Q^T * Grad_softmax
    */
    // Grad_V shape is (GLOBAL_MxGLOBAL_K)
    // TODO double check the relationship between the GLOBAL_M and GLOBAL_N
    const dim3 gradv_dimBlock(256);
    const dim3 gradv_dimGrid(GLOBAL_M/32, GLOBAL_K/32, HEAD_NUM*batch_size);
    // printf("M: %d K: %d, SPARSE_VAL_SIZE:%d \\n", GLOBAL_M, GLOBAL_K, SPARSE_VAL_SIZE);
    // the block size is 32x32, the val index stores the output vals of softmax output.
    // grad_row_idx is the transposed csr index of the softmax output tensor. To save
    // memory, we leverage the original vals of the softmax output tensor.
    GRAD_V_KERNEL<<<gradv_dimGrid, gradv_dimBlock>>>(gradv_row_idx,
                                                     gradv_col_idx,
                                                     gradv_subblock_idx,
                                                     val,
                                                     in_grad,
                                                     V_grad);
    // calculate the gradient of the output tensor of softmax
    const dim3 atten_dimBlock(256);
    const dim3 atten_dimGrid(SMALL_BLOCK_NUM, HEAD_NUM*batch_size);
    GRAD_ATTEN_KERNEL<<<atten_dimGrid, atten_dimBlock>>>(in_grad, V, val_grad, m_index, n_index, block_index);

    // calculate the gradient of the input tensor of softmax
    const dim3 softmax_dimBlock(256);
    const dim3 softmax_dimGrid(SMALL_BLOCK_NUM, HEAD_NUM*batch_size);
    GRAD_SOFTMAX_KERNEL<<<softmax_dimGrid, softmax_dimBlock>>>(val, val_grad, qxk_grad, m_index, col_range_index, block_index);
    
    // caculate the gradient of Q
    // grad_Q = qxk_grad * K
    const dim3 gradq_dimBlock(256);
    const dim3 gradq_dimGrid(GLOBAL_M/32, GLOBAL_K/32, HEAD_NUM*batch_size);
    GRAD_Q_KERNEL<<<gradq_dimGrid, gradq_dimBlock>>>(qxk_grad, row_ptr, col_idx, K, Q_grad);

    const dim3 gradk_dimBlock(256);
    // K_grad = qxk_grad^T * Q --> (N,M) x (M,k)
    const dim3 gradk_dimGrid(GLOBAL_N/32, GLOBAL_K/32, HEAD_NUM*batch_size);
    GRAD_V_KERNEL<<<gradk_dimGrid, gradk_dimBlock>>>(gradv_row_idx,
                                                     gradv_col_idx,
                                                     gradv_subblock_idx,
                                                     qxk_grad,
                                                     Q,
                                                     K_grad);

}

at::Tensor our_sparse_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor d_m_index,
    torch::Tensor d_n_index,
    torch::Tensor d_block_index,
    torch::Tensor val,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor val_mask,
    torch::Tensor col_range_index
)
{
    cudaSetDevice(Q.get_device());
    int batch_size = Q.size(0);
    torch::Tensor output = torch::empty({batch_size, HEAD_NUM, GLOBAL_M, GLOBAL_K}, Q.options());
    
    AT_DISPATCH_FLOATING_TYPES(Q.type(), "our_sparse_attention", ([&]
                            { forward_function(
                                    Q.data_ptr<float>(),
                                    K.data_ptr<float>(),
                                    V.data_ptr<float>(),
                                    d_m_index.data_ptr<int>(),
                                    d_n_index.data_ptr<int>(),
                                    d_block_index.data_ptr<int>(),
                                    val.data_ptr<float>(),
                                    row_ptr.data_ptr<int>(),
                                    col_idx.data_ptr<int>(),
                                    val_mask.data_ptr<int>(),
                                    col_range_index.data_ptr<int>(),
                                    batch_size,
                                    
                                    output.data_ptr<float>()
                                ); }));
    return output;
}

std::vector<at::Tensor> our_sparse_attention_backward(
    torch::Tensor grad,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor gradv_row_idx,
    torch::Tensor gradv_col_idx,
    torch::Tensor gradv_subblock_idx,
    torch::Tensor val,
    torch::Tensor m_index,
    torch::Tensor n_index,
    torch::Tensor block_index,
    torch::Tensor col_range_index,
    torch::Tensor row_ptr,
    torch::Tensor col_idx
    )
{
    cudaSetDevice(grad.get_device());
    int batch_size = Q.size(0);
    torch::Tensor Q_grad = torch::empty_like(Q);
    torch::Tensor K_grad = torch::empty_like(K);
    torch::Tensor V_grad = torch::empty_like(V);
    torch::Tensor val_grad = torch::zeros_like(val);
    torch::Tensor qxk_grad = torch::zeros_like(val);
    AT_DISPATCH_FLOATING_TYPES(Q.type(), "our_sparse_attention", ([&]
        { backward_function(
                grad.data_ptr<float>(),
                Q.data_ptr<float>(),
                K.data_ptr<float>(),
                V.data_ptr<float>(),
                gradv_row_idx.data_ptr<int>(),
                gradv_col_idx.data_ptr<int>(),
                gradv_subblock_idx.data_ptr<int>(),
                val.data_ptr<float>(),
                m_index.data_ptr<int>(),
                n_index.data_ptr<int>(),
                block_index.data_ptr<int>(),
                col_range_index.data_ptr<int>(),
                row_ptr.data_ptr<int>(),
                col_idx.data_ptr<int>(),
                batch_size,

                Q_grad.data_ptr<float>(),
                K_grad.data_ptr<float>(),
                V_grad.data_ptr<float>(),
                val_grad.data_ptr<float>(),
                qxk_grad.data_ptr<float>()
            ); }));
    // Note: check if directly return the vector<Tensor> is efficient
     std::vector<at::Tensor> grads({Q_grad, K_grad, V_grad, qxk_grad, val_grad});
     return grads;
}
"""

sparse_attention_interface = """
#include <vector>
#include "torch/extension.h"

at::Tensor our_sparse_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor d_m_index,
    torch::Tensor d_n_index,
    torch::Tensor d_block_index,
    torch::Tensor val,
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor val_mask,
    torch::Tensor col_range_index
);

std::vector<at::Tensor> our_sparse_attention_backward(
    torch::Tensor grad,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor gradv_row_idx,
    torch::Tensor gradv_col_idx,
    torch::Tensor gradv_subblock_idx,
    torch::Tensor val,
    torch::Tensor m_index,
    torch::Tensor n_index,
    torch::Tensor block_index,
    torch::Tensor col_range_index,
    torch::Tensor row_ptr,
    torch::Tensor col_idx
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &our_sparse_attention_forward, "our sparse attention forward");
    m.def("backward", &our_sparse_attention_backward, "our sparse attention backward");

}
"""