#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include <vector>
#include <utility>

#define OP_N 0
#define OP_T 1
#define CEIL_DIV(x, y) (((x) + (y) -   1) / (y))

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

typedef struct __align__(2) bhalf {
    __device__ __forceinline__ bhalf() {}
    __device__ __forceinline__ bhalf(ushort v) : x(v) {}
    ushort x;
} bhalf;

typedef struct __align__(8) bhalf4 {
    __device__ __forceinline__ bhalf4() {}
    __device__ __forceinline__ bhalf4(uint v) : x(v), y(v) {}
    __device__ __forceinline__ bhalf4(uint v0, uint v1) : x(v0), y(v1) {}
    uint x;
    uint y;
} bhalf4;

typedef unsigned long long uint64;

void init(float * ptr, size_t length, float sparsity)
{
    // lock the random seed for
    srand (1);
    for (int i = 0; i < length; i++)
    {
        float pro = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        //printf("pro: %f\n", pro);
        if (pro < sparsity)
        {
            ptr[i] = 0.0;
        }
        else
        {
            ptr[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }
}

void magicu64(uint d, uint &magic, uint &shift)
{
    // common cases
         if (d == 1) { magic = 1; shift = 0; }
    else if (d == 2) { magic = 1; shift = 1; }
    else if (d == 4) { magic = 1; shift = 2; }
    else if (d == 8) { magic = 1; shift = 3; }
    else
    {
        // 3 is a special case that only ends up in the high bits if the nmax is 0xffffffff
        // we can't use 0xffffffff for all cases as some return a 33 bit magic number
        uint   nbits = d == 3 ?   (2*32)+1 :   (2*31)+1;
        uint64 nmax  = d == 3 ? 0xffffffff : 0x7fffffff;
        uint64 d64   = d;
        uint64 nc    = ((nmax + 1ull) / d64) * d64 - 1ull;

        for (uint p = 0; p < nbits; p++)
        {
            if ((1ull << p) > nc * (d64 - 1ull - ((1ull << p) - 1ull) % d64))
            {
                magic = (uint)(((1ull << p) + d64 - 1ull - ((1ull << p) - 1ull) % d64) / d64);
                shift = magic == 1 ? p : p - 32;
                //printf("div:%u magic:%u shift:%u\n", d, magic, shift);
                return;
            }
        }
    }
}


__device__ __forceinline__ float shfl_xor(float var, int laneMask)
{
    float ret;
# if CUDA_VERSION >= 9020
    asm volatile ("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(ret) : "f"(var), "r"(laneMask));
# else
    asm volatile ("shfl.bfly.b32 %0, %1, %2, 0x1f;" : "=f"(ret) : "f"(var), "r"(laneMask));
# endif
    return ret;
}

__device__ __forceinline__ float4 to_float(bhalf4 v)
{
    float4 r;
    asm("{\n\t"
        ".reg .u16 u0, u1;\n\t"
        "mov.b32 {u0, u1}, %4;\n\t" // force XMAD.PSL.CLO instead of SHL
        "mov.b32 %0, {0, u0};\n\t"
        "and.b32 %1, %4, 0xffff0000;\n\t"
        "mov.b32 {u0, u1}, %5;\n\t"
        "mov.b32 %2, {0, u0};\n\t"
        "and.b32 %3, %5, 0xffff0000;\n\t"
        "}" : "=f"(r.x),"=f"(r.y),"=f"(r.z),"=f"(r.w) : "r"(v.x),"r"(v.y));
    return r;
}

__device__ __forceinline__ uint div64(uint value, uint magic, uint shift)
{
    // if the divisor is a power of 2 the magic will be 1 and it's just a simple right shift
    // Otherwise multiply by magic and right shift just the high bits
    uint result;
    asm("{                            \n\t"
        ".reg .pred p;                \n\t"
        ".reg .u64 res64;             \n\t"
        ".reg .u32 lo32, hi32;        \n\t"
        "setp.ne.s32 p, %2, 1;        \n\t"
        "mul.wide.u32 res64, %1, %2;  \n\t"
        "mov.b64 {lo32, hi32}, res64; \n\t"
        "selp.u32 hi32, hi32, %1, p;  \n\t"
        "shr.u32 %0, hi32, %3;        \n\t"
        "}" : "=r"(result) : "r"(value), "r"(magic), "r"(shift));
    return result;
}

__device__ __forceinline__ const bhalf* add_ptr_u(const bhalf* src, int offset)      \
{                                                                            \
    const bhalf* dst;                                                            \
    asm("{                       \n\t"                                       \
        ".reg .u32 lo,hi,of;     \n\t"                                       \
        "mul.lo.u32 of, %2, %3;  \n\t"                                       \
        "mov.b64    {lo,hi}, %1; \n\t"                                       \
        "add.cc.u32  lo,lo,  of; \n\t"                                       \
        "addc.u32    hi,hi,  0;  \n\t"                                       \
        "mov.b64 %0, {lo,hi};    \n\t"                                       \
        "}" : "=l"(dst) : "l"(src), "r"(offset), "r"((int)sizeof(*src)));    \
    return dst;                                                              \
}  


__device__ __forceinline__ bhalf4 __ldg(const bhalf4 *ptr)
{
    bhalf4 ret;
    asm volatile ("ld.global.nc.v2.u32 {%0, %1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ void __stg(float4 *ptr, float4 val)
{
    asm volatile ("st.global.wb.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(ptr), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w)  );
}

__device__ __forceinline__ void store(float4* out, float4 v, int i=0, bool b=true) { if (b) __stg(out + i, v); }

template <uint OP_A, bool N64>
__global__ void __launch_bounds__(128,6) bst_sgemm_32x64x32_xn(
    const uint2* __restrict__ Lut,
    const float* __restrict__ A,
    const float* __restrict__ B,
          float*              C,
    uint szCtxHeadStateB, uint szCtxHeadStateC, uint szHeadState, uint szState,
    uint szHeadBlocksBlk, uint szBlocksBlk, uint szLut,
    uint grid_M, uint grid_N, uint magic_N, uint shift_N)
{
    __shared__ float fShare[(33 + 64)*32];
    uint2* Lut2s = (uint2*)&fShare[(33 + 64)*32];
    char* bShare = (char*)&fShare;

    uint tid    = threadIdx.x;
    uint idx_MN = blockIdx.x; // compound outer product dims
    uint idx_M  = div64(idx_MN, magic_N, shift_N); // idx_M = idx_MN / grid_N;
    uint idx_N  = idx_MN - idx_M*grid_N;           // idx_N = idx_MN % grid_N;
    uint idx_B  = blockIdx.y; // batch dim
    uint idx_H  = blockIdx.z; // head dim

    // assume lower diagonal and schedule large reductions first
    if (OP_A == OP_N)
        idx_M = grid_M - idx_M;

    // each head can optionally have its own lut
    Lut += idx_H*szLut;
    uint2 lut_head   = Lut[idx_M];
    uint  lut_offset = lut_head.x;
    uint  lut_size   = lut_head.y;

    uint txb = tid % 16;
    uint tyb = tid / 16;

    if (lut_size > 0)
    {
        // prefetch the lut data into shared
        Lut += lut_offset;
        #pragma unroll 1
        for (uint i = tid; i < lut_size; i += 128)
        {
            uint2 entry = Lut[i];
            entry.x *= 32*32;  // 1024 entries of A per block
            entry.y *= szHeadState*32;   // 32 lines of B per block
            Lut2s[i] = entry;
        }
        __syncthreads();

        uint txa = tid % 8;
        uint tya = tid / 8;

        uint tid16 = tid & 16;
        uint tid96 = tid & 96;

        uint loadB = ((tid / 2) % 8) * 4*4;
        uint loadA =  (tid % 2)      * 4*4;

        // each warp handles a quarter of the weights
        loadA += tid96;

        // second half of warp starts 16 rows down
        loadB += tid16 * 64*4;
        loadA += tid16 * 32*4;

        uint storB = (tyb*64 + txb*4) * 4;
        uint storA;
        if (OP_A == OP_T)
            storA = tid * 4*4;
        else
        {
            // Transpose weights on store to shared
            // Avoid bank conflicts by shifting writes over by 4 every 4 rows (+txa*4)
            storA = (txa*32*4 + tya + txa*4) * 4;
            loadA += tid16 * 4; // shift over 4 floats every 4 rows, second half of warp starts 16 rows down
        }

        uint b = idx_N*64 + txb*4;
        uint offsetA = idx_B*szHeadBlocksBlk + idx_H*szBlocksBlk + tid*4;
        uint offsetB = idx_B*szCtxHeadStateB + tyb*szHeadState + idx_H*szState + b;

        bool inB = N64 || b < szState;

        // zero accumulation registers
        float regC[4][8];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 8; j++)
                regC[i][j] = 0.0f;

        // Force compiler to fully compute these prior to loop
        asm("mov.b32 %0, %0;" : "+r"(loadA)   : );
        asm("mov.b32 %0, %0;" : "+r"(loadB)   : );
        asm("mov.b32 %0, %0;" : "+r"(storA)   : );
        asm("mov.b32 %0, %0;" : "+r"(storB)   : );
        asm("mov.b32 %0, %0;" : "+r"(offsetA) : );
        asm("mov.b32 %0, %0;" : "+r"(offsetB) : );

        int idx_lut = 0;
        #pragma unroll 1
        do
        {
            //asm volatile (".pragma \"nounroll\";"::); // ptxas, don't get clever

            uint2 entry = Lut2s[idx_lut];

            const float* pA = A + entry.x + offsetA;
            float4 a00 = *((const float4*)(pA +  0*32));
            float4 a16 = *((const float4*)(pA + 16*32));
            float4 b00 = {0.0f}, b08 = {0.0f}, b16 = {0.0f}, b24 = {0.0f};
            entry.y += offsetB;
            if (inB)
            {
                b00 = *((const float4*)(B + (entry.y +  0*szHeadState)));
                b08 = *((const float4*)(B + (entry.y +  8*szHeadState)));
                b16 = *((const float4*)(B + (entry.y + 16*szHeadState)));
                b24 = *((const float4*)(B + (entry.y + 24*szHeadState)));
            }
            __syncthreads();


            if (OP_A == OP_T)
            {
                *(float4*)&bShare[storA + (0*16*32 + 64*32)*4] = a00;
                *(float4*)&bShare[storA + (1*16*32 + 64*32)*4] = a16;
            }
            else
            {
                // transpose the shared store of W
                *(float*)&bShare[storA + (0*32 + 0*16 + 64*32)*4] = a00.x;
                *(float*)&bShare[storA + (1*32 + 0*16 + 64*32)*4] = a00.y;
                *(float*)&bShare[storA + (2*32 + 0*16 + 64*32)*4] = a00.z;
                *(float*)&bShare[storA + (3*32 + 0*16 + 64*32)*4] = a00.w;

                *(float*)&bShare[storA + (0*32 + 1*16 + 64*32)*4] = a16.x;
                *(float*)&bShare[storA + (1*32 + 1*16 + 64*32)*4] = a16.y;
                *(float*)&bShare[storA + (2*32 + 1*16 + 64*32)*4] = a16.z;
                *(float*)&bShare[storA + (3*32 + 1*16 + 64*32)*4] = a16.w;
            }

            *(float4*)&bShare[storB +  0*64*4] = b00;
            *(float4*)&bShare[storB +  8*64*4] = b08;
            *(float4*)&bShare[storB + 16*64*4] = b16;
            *(float4*)&bShare[storB + 24*64*4] = b24;
            __syncthreads();

            // computes a 32x64x32 gemm tile with 4x8 register blocking
            float regA[4];
            float regB[8];
            #pragma unroll
            for (int j = 0; j < 16; j++)
            {
                // fetch outer product data
                *(float4*)&regA[0] = *(float4*)&bShare[loadA + (32*j + 64*32 + (OP_A == OP_T ? 0 : (j/4)*4))*4]; // shift over 4 floats every 4 rows
                *(float4*)&regB[0] = *(float4*)&bShare[loadB + (64*j +  0)*4];
                *(float4*)&regB[4] = *(float4*)&bShare[loadB + (64*j + 32)*4];

                // accumulate outer product
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 8; j++)
                        regC[i][j] += regA[i] * regB[j];
            }


        } while (++idx_lut < lut_size);

        asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid  )  :);
        asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(idx_MN) :);
        asm volatile ("mov.u32 %0, %ctaid.y;" : "=r"(idx_B)  :);
        asm volatile ("mov.u32 %0, %ctaid.z;" : "=r"(idx_H)  :);
        idx_M  = div64(idx_MN, magic_N, shift_N);
        idx_N  = idx_MN - idx_M*grid_N;

        if (OP_A == OP_N)
            idx_M = grid_M - idx_M;

        // printf("%3d %.0f %.0f %.0f %.0f |  %.0f %.0f %.0f %.0f |  %.0f %.0f %.0f %.0f | %.0f %.0f %.0f %.0f\n", tid,
        //     regC[0][0], regC[0][1], regC[0][2], regC[0][3],
        //     regC[1][0], regC[1][1], regC[1][2], regC[1][3],
        //     regC[2][0], regC[2][1], regC[2][2], regC[2][3],
        //     regC[3][0], regC[3][1], regC[3][2], regC[3][3]);

        tid16 = tid & 16;
        tid96 = tid & 96;

        uint tn =  (tid / 2) % 8;
        uint tm = ((tid % 2) + (tid96 / 16))*4 + (tid16 / 16);

        bool t16 = tid16 != 0;

        float outC[2][8];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 8; j++)
            {
                float swap = t16 ? regC[2*i + 0][j] : regC[2*i + 1][j];
                outC[i][j] = t16 ? regC[2*i + 1][j] : regC[2*i + 0][j];
                outC[i][j] += shfl_xor(swap, 16);
            }

        uint n = idx_N*64 + tn*4;
        bool bn00 = N64 || n +  0 < szState;
        bool bn32 = N64 || n + 32 < szState;

        uint offsetC = idx_B*szCtxHeadStateC + (idx_M*32 + tm)*szHeadState + idx_H*szState + n;

        store((float4*)(C + (offsetC + szHeadState*0 +  0)), *(float4*)&outC[0][0], 0, bn00);
        store((float4*)(C + (offsetC + szHeadState*0 + 32)), *(float4*)&outC[0][4], 0, bn32);
        store((float4*)(C + (offsetC + szHeadState*2 +  0)), *(float4*)&outC[1][0], 0, bn00);
        store((float4*)(C + (offsetC + szHeadState*2 + 32)), *(float4*)&outC[1][4], 0, bn32);
    }
    else
    {
        uint c       = idx_N*64 + txb*4;
        uint offsetC = idx_B*szCtxHeadStateC + (idx_M*32 + tyb)*szHeadState + idx_H*szState + c;

        if (N64 || c < szState)
        {
            float4 zero = {0.0f};
            *(float4*)&C[offsetC + szHeadState* 0] = zero;
            *(float4*)&C[offsetC + szHeadState* 8] = zero;
            *(float4*)&C[offsetC + szHeadState*16] = zero;
            *(float4*)&C[offsetC + szHeadState*24] = zero;
        }
    }
}

int main(int argc, char*argv[]) {
    float sparsity_ratio = atof(argv[1]);
    printf("Sparsity Ratio=%f\n", sparsity_ratio);
    int head_dim = 1;
    int batch_dim = 1;
    int time_dimension = 1024;
    int embed_state = 1024;
    int block_size = 32;

    int szState = embed_state;
    int szHeadState = head_dim * szState;
    int ctx_blks_b = time_dimension / block_size;
    int ctx_blks_c = time_dimension / block_size;
    int szCtxHeadStateB = ctx_blks_b * block_size * szHeadState;
    int szCtxHeadStateC = ctx_blks_c * block_size * szHeadState;

    uint magic_, shift_;
    uint div = CEIL_DIV(szState, 64);
    magicu64(div, magic_, shift_);

    size_t A_dense_length = time_dimension * time_dimension;
    size_t A_dense_size = time_dimension * time_dimension * sizeof(float);

    float *A_dense = (float *)malloc(A_dense_size);

    init(A_dense, A_dense_length, sparsity_ratio);

    int layout[ctx_blks_b][ctx_blks_c]; memset(layout, 0, sizeof(layout));

    int blocks = 0;
    std::vector<int> ys, xs;
    for(int i = 0; i < ctx_blks_b; i++){
        for(int j = 0; j < ctx_blks_c; j++){
            int i_start = i * block_size;
            int i_end = (i+1) * block_size;
            int j_start = j * block_size;
            int j_end = (j+1) * block_size;
            bool has_nonzero = false;
            for(int i_in = i_start; i_in < i_end; i_in += 1){
                for(int j_in = j_start; j_in < j_end; j_in += 1){
                    if(A_dense[i_in * time_dimension + j_in] != 0){
                        has_nonzero = true;
                    }
                }
            }
            if(has_nonzero == true){
                ys.push_back(i);
                xs.push_back(j);
                layout[j][i] = 1;
                blocks += 1;
            }
        }
    }

    std::vector<std::vector< std::pair < int, int > > > py_lut(ctx_blks_b);
    for(int i = 0; i < blocks; i++){
        std::pair<int, int> tmp(i, xs[i]);
        py_lut[ys[i]].push_back(tmp);
        // py_lut[ys[i]].push_back(std::pair<int, int>{b, xs[i]});
    }

    int max_lut = 0;
    int offset = ctx_blks_b;
    int size_lut = (blocks+offset)*sizeof(uint2);
    uint2 *np_lut = (uint2*)malloc((blocks+offset)*sizeof(uint2));

    for(int i = 0; i < ctx_blks_b; i++) {
        np_lut[i].x = offset, np_lut[i].y = py_lut[i].size();       // offset, size
        max_lut = max(max_lut, (int)py_lut[i].size());
        for(int j = 0; j < py_lut[i].size(); j++){
            np_lut[offset].x = py_lut[i][j].first;      // block_id
            np_lut[offset].y = py_lut[i][j].second;     // col position
            offset += 1;
        }
    }

    uint szBlocksBlk     = blocks * block_size * block_size;
    uint szHeadBlocksBlk = head_dim * szBlocksBlk;

    // max_lut: maximum length of block line

    // if just one lut head, broadcast block-sparsity to all heads
    uint szLut = 0;

    // compound gridDim.x with m and n coords
    uint gridN  = CEIL_DIV(embed_state, 64);
    uint gridM  = ctx_blks_c - 1;
    uint gridX  = ctx_blks_c * gridN;
    uint shared = ((max_lut+1)/2)*2*8; // round up to nearest even, 8 bytes per entry

    dim3 grid(gridX, batch_dim, head_dim);

    int size_A = batch_dim * head_dim * blocks * block_size * block_size * sizeof(float);
    int size_B = batch_dim * head_dim * time_dimension * embed_state * sizeof(float);
    int size_C = batch_dim * head_dim * time_dimension * embed_state * sizeof(float);

    float *A = (float *)malloc(size_A);
    float *B = (float *)malloc(size_B);
    float *C = (float *)malloc(size_C);

    init(A, time_dimension * time_dimension, sparsity_ratio);
    for(int i = 0; i < size_A / sizeof(float); i++) A[i] = (float)(rand()%5);
    for(int i = 0; i < size_B / sizeof(float); i++) {
        B[i] = rand()%5;
        C[i] = 0;
    }

    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    uint2 *d_lut = NULL;

    int ntimes = 100;

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), size_B));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), size_C));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_lut), size_lut));

    checkCudaErrors(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_C, C, size_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_lut, np_lut, size_lut, cudaMemcpyHostToDevice));

    for(int i = 0; i < ntimes; i++){
        bst_sgemm_32x64x32_xn<OP_N, true><<<grid,128,shared>>>(d_lut, d_A, d_B, d_C, szCtxHeadStateB, szCtxHeadStateC, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, gridM, gridN, magic_, shift_);
    }

    cudaEvent_t start, stop;

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));
    for(int i = 0; i < ntimes; i++){
        bst_sgemm_32x64x32_xn<OP_N, true><<<grid,128,shared>>>(d_lut, d_A, d_B, d_C, szCtxHeadStateB, szCtxHeadStateC, szHeadState, szState, szHeadBlocksBlk, szBlocksBlk, szLut, gridM, gridN, magic_, shift_);
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));

    float milliseconds = 0;

    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
  
    printf("Time= %f ms\n", milliseconds / ntimes);

    free(A);
    free(B);
    free(C);
    free(np_lut);

    checkCudaErrors(cudaFree(reinterpret_cast<void *>(d_A)));
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(d_B)));
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(d_C)));
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(d_lut)));

    return 0;
}