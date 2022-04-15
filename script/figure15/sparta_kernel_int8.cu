/*
 * Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// CUDA sample demonstrating a integer GEMM computation using the Warp Matrix
// Multiply and Accumulate API.

// In this program, the compute_gemm kernel computes the result of a matrix
// multiplication and addition: D = alpha * (A * B + C). The dimensions of
// both C and D matrices are M_GLOBAL x N_GLOBAL. The A matrix is M_GLOBAL x
// K_GLOBAL (row-major), the B matrix is K_GLOBAL x N_GLOBAL (column-major). In
// that kernel, each CTA computes one 128 x 128 tile of the resulting matrix per
// iteration. When the tile is computed, the CTA stores it to the global memory
// and begins a new iteration, selecting a new 128 x 128 tile to compute.
// Each CTA consists of eight warps. For the 128 x 128 tile, each warp computes
// eight 16 x 16 subtiles, organized in a 2 x 4 two-dimensional array. Warps
// compute the 16 x 16 subtiles using nvcuda::wmma::mma_sync operations by
// moving through the K_GLOBAL dimension of the A and B matrices and
// accumulating the intermediate result in the local thread state.

// There are a number of simple optimizations used in the algorithm:
// - The CTA copies the 128 x 128 tile of the C matrix from the global memory to
//   shared memory. After that is done, each warp loads the C matrix fragments
//   from shared memory, thus avoiding a random global memory access.
// - On each internal iteration, the CTA copies a portion of the A and B
// matrices from
//   global memory to shared memory. After that, all warps in the CTA reuse the
//   A and B data from shared memory, thus reducing the number of data copies
//   from global memory.
// - The portions of the A and B matrices are stored in shared memory with an
// additional
//   padding (skew) to reduce the number of shared memory access bank conflicts.
//   (See a detailed explanation near the SKEW_HALF macro definition.)
// - When the CTA finishes computing the tiles of the resulting matrix, each
// warp stores
//   its subtiles to shared memory. The CTA then copies the shared memory
//   contents to global memory, again avoiding redundant random global memory
//   accesses.
// - Note that the CTA tile size is chosen to maximize the GPU register
// utilization,
//   but carefully enough to avoid local memory use.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cstring>
#include <fstream>
#include <iostream>

#include <assert.h>
#include <cuda.h>
#include <mma.h>

// Externally configurable parameters.

size_t load_from_file(char* ptr, size_t buff_size, std::string filepath){
  std::ifstream fin(filepath, std::ios::in | std::ios::binary);
  size_t loaded_size = fin.read(ptr, buff_size).gcount();
  return loaded_size;
}

#ifndef CPU_DEBUG
// Set this to 1 to verify the correctness of the GPU-computed matrix.
#define CPU_DEBUG 1
#endif

#ifndef SHARED_MEMORY_LIMIT_64K
// Set this to 0 to use more than 64 Kb of shared memory to cache data, to
// improve the performance of the computations on GPU.
// Note that you need a GPU that can have more than 64 Kb of shared memory
// per multiprocessor.
#define SHARED_MEMORY_LIMIT_64K 1
#endif

// GPU configuration.

#define WARP_SIZE 32

// MMA matrix tile dimensions.

#define M 16
#define N 16
#define K 16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// GEMM configuration.

#define M_GLOBAL 1024
#define N_GLOBAL 1024
#define K_GLOBAL 1024

#define M_TILES (M_GLOBAL / M)
#define N_TILES (N_GLOBAL / N)
#define K_TILES (K_GLOBAL / K)

#define C_LAYOUT wmma::mem_row_major

// Implementation constants.

// With only 64 Kb shared memory available, we can fit two 8-tile chunks of
// the A and B matrix data, that are 16 * 16 * 8 * 8 * 2 = 32 Kb each
// (i.e. two 8x8 arrays of tiles of 16x16 uint8_t-typed elements per CTA).
// But we cannot account the 8 Kb total skew overhead, without which the
// performance would be severely impacted. So we choose to reduce the chunk size
// in half, i.e. the amount of A and B matrix data we cache in shared memory.
// Accordingly, this doubles the number of outer iterations across the global K
// dimension, which only slightly impacts the performance.

////// CHUNK_K * K <= K_GLOBAL //////

#define CHUNK_K 2

#define CHUNK_LINE_BYTES (CHUNK_K * K * sizeof(uint8_t))                // 128
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))                      // 512
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)  // 4
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)   // 8


#define BLOCK_ROW_WARPS 1
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 2
#define WARP_COL_TILES 2


// may be we can tune number here
#define LANE_ROW_STRIDE (WARP_ROW_TILES * N / 8)
#define LANE_COL_STRIDE (WARP_COL_TILES * M / 4)
#define WARP_STRIDE (WARP_ROW_TILES * N)

#define WARPS_PER_BLOCK (BLOCK_ROW_WARPS * BLOCK_COL_WARPS)
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)


/////////// BLOCK_ROW_TILES <= N_TILES ////////////
#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)

/////////// BLOCK_COL_TILES <= M_TILES ////////////
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

// The macro below is used to shift rows of the A matrix and columns of the B
// matrix in shared memory to minimize possible bank conflicts. Before
// performing the nvcuda::wmma::mma_sync operation, the warp must load the
// matrix data using the nvcuda::wmma::load_matrix_sync operation. Although the
// memory access pattern is not specified for that function, each lane in the
// warp can read one or multiple matrix elements from different matrix rows or
// columns. For shared memory, such access can result in bank conflicts if
// different rows / columns of the matrix map to the same bank. By shifting each
// row and column by a few bytes, we make sure that they map to different banks,
// thus reducing the number of possible bank conflicts. The number of 32
// one-byte "uint8_t" elements is chosen as the minimum possible shift because
// we must keep each row and column 256-bit aligned, as required by
// nvcuda::wmma::load_matrix_sync.
#define SKEW_UINT8 32

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}



using namespace nvcuda;

__global__ void matrixMultiplication(float *input0, float *input1, float *output0);

__global__ void compute_gemm_imma(const uint8_t *A, const uint8_t *B_val, int *B_row, int *B_col,
                                  uint8_t *D, int alpha, int integer) {
  //extern __shared__ uint8_t shmem[][CHUNK_K * K + SKEW_UINT8];
  const int shared_size = MAX(sizeof(uint8_t) * (BLOCK_COL_TILES * M + BLOCK_ROW_TILES * N) *
                       (CHUNK_K * K + SKEW_UINT8),
                   M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N *
                       (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(int));
  __shared__ uint8_t shmem[shared_size/(CHUNK_K * K + SKEW_UINT8)+1][CHUNK_K * K + SKEW_UINT8];

  // Warp and lane identification.
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;

  // Offset in shared memory from which the B matrix is stored.
  const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;       // BLOCK_COL_TILES * M is shared_A row numbers in one block


  // Each CTA slides along the 128 x 128 tiles from the top left corner of the
  // matrix to the right and down, and selects the next tile to compute. Once
  // there's no such tile, all warps in this CTA exit.

    unsigned int block_pos = blockIdx.x;
    const unsigned int block_tile_i =                                   // get the i (row) index of all tiles
        ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
    const unsigned int block_tile_j = (block_pos * BLOCK_ROW_TILES) % N_TILES;

    // Stop when there are no more D matrix tiles to compute in this CTA.


    // This warp's pointer to the C matrix data to copy memory from to shared
    // memory.

    //__syncthreads();
    // These fragments will accumulate the result of A and B matrix fragment
    // multiplications along the K_GLOBAL dimension.
    wmma::fragment<wmma::accumulator, M, N, K, int> c[WARP_COL_TILES]
                                                     [WARP_ROW_TILES];

    // Load the C matrix tiles into fragments from shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
        // within fragment storage are not defined.
        for (int t = 0; t < c[i][j].num_elements; t++) {
            c[i][j].x[t] = 0;
          }
      }
    }

    __syncthreads();

    // Select what warp copies what matrix to shared memory.
    // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.

    int start_tile = B_row[block_tile_j / BLOCK_ROW_TILES];
    int end_tile = B_row[block_tile_j / BLOCK_ROW_TILES + 1];

    // int start_tile = B_row[block_tile_j / WARP_COL_TILES + (warpId % BLOCK_ROW_WARPS)];
    // int end_tile = B_row[block_tile_j / WARP_COL_TILES + (warpId % BLOCK_ROW_WARPS) + 1];

    // Go through the global K dimension by a fixed step at a time.
#pragma unroll
    for(int tile_k_idx = start_tile; tile_k_idx < end_tile; tile_k_idx += 1){

      size_t shmem_idx = 
        warpId < (WARPS_PER_BLOCK / 2)
          ? (warpId % (WARPS_PER_BLOCK / 2)) * CHUNK_COPY_LINES_PER_WARP
          : (warpId % (WARPS_PER_BLOCK / 2)) * CHUNK_COPY_LINES_PER_WARP + shmem_idx_b_off;

      int4 *lane_ptr = NULL;
      if(warpId < (WARPS_PER_BLOCK / 2)){
        const uint8_t *warp_ptr = &A[block_tile_i * M * K_GLOBAL] +
          (warpId % (WARPS_PER_BLOCK / 2)) * CHUNK_COPY_LINES_PER_WARP * K_GLOBAL;
        lane_ptr = (int4 *)(warp_ptr + B_col[ tile_k_idx] * K * CHUNK_K +
                        (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) +
                        (laneId % CHUNK_COPY_LINE_LANES);
      }else{
        const uint8_t *warp_ptr = B_val + tile_k_idx * (N * BLOCK_ROW_TILES) * (K * CHUNK_K) + 
          (warpId % (WARPS_PER_BLOCK / 2)) * CHUNK_COPY_LINES_PER_WARP * (K * CHUNK_K);
        lane_ptr = (int4 *)(warp_ptr + (laneId / CHUNK_COPY_LINE_LANES) * (K * CHUNK_K)) +
                    (laneId % CHUNK_COPY_LINE_LANES);
      }

      // Shift the second half of the warp to the next row / column in the
      // shared memory.
      shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

      int iter_index = warpId < (WARPS_PER_BLOCK / 2)
        ? (BLOCK_COL_TILES * M) / ((WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP)
        : (BLOCK_ROW_TILES * N) / ((WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP);


      #pragma unroll
      for (int i = 0; i < iter_index; i++) {
        // Copy 16 bytes at once in each lane.
        *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
            *lane_ptr;

        
        lane_ptr = warpId < (WARPS_PER_BLOCK / 2) ?
                    (int4 *)((uint8_t *)lane_ptr +
                                        K_GLOBAL * (WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP):
                    (int4 *)((uint8_t *)lane_ptr +
                                        K * CHUNK_K * (WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP);
        shmem_idx += (WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP;
      }

      __syncthreads();

      for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        wmma::fragment<wmma::matrix_a, M, N, K, uint8_t, wmma::row_major>
            a[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_b, M, N, K, uint8_t, wmma::col_major>
            b[WARP_ROW_TILES];

#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
          size_t shmem_idx_a = (warpId / BLOCK_ROW_WARPS) * M * WARP_COL_TILES + (i * M);
          const uint8_t *tile_ptr = &shmem[shmem_idx_a][k_step * K];

          wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_UINT8);

#pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
            if (i == 0) {
              // Load the B matrix fragment once, because it is going to be
              // reused against the other A matrix fragments.
              size_t shmem_idx_b = shmem_idx_b_off +
                                   (WARP_ROW_TILES * N) * (warpId % BLOCK_ROW_WARPS) +
                                   (j * N);
              const uint8_t *tile_ptr = &shmem[shmem_idx_b][k_step * K];

              wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_UINT8);
            }

            wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
          }
        }
      }

      __syncthreads();
    }

    // This pointer is used to access the C and D matrix tiles this warp computes.
    int *shmem_warp_tile_ptr = (int *)&shmem[0][0] +
    (warpId / BLOCK_ROW_WARPS) * M * WARP_COL_TILES * SHMEM_STRIDE  +    // K * 2 is because one warp calculate k * 2 rows.
    (warpId % BLOCK_ROW_WARPS) * SHMEM_OFFSET;

      // Store the D fragments to shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        // Uniform, point-wise transformations of ALL fragment elements by ALL
        // threads in the warp are well-defined even though element indices
        // within fragment storage are not defined.
        for (int t = 0; t < c[i][j].num_elements; t++) {
          c[i][j].x[t] = ((c[i][j].x[t] * alpha) >> integer);
        }

        int *tile_ptr = shmem_warp_tile_ptr + i * M * SHMEM_STRIDE + j * N;

        wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
      }
    }

    __syncthreads();
  

    int *shmem_warp_stream_ptr = (int *)&shmem[0][0] + (warpId / BLOCK_ROW_WARPS) * WARP_COL_TILES * M * SHMEM_STRIDE
                                    + (warpId % BLOCK_ROW_WARPS) * WARP_ROW_TILES * N; 
    const size_t gmem_idx =
        (block_tile_i * M + (warpId / BLOCK_ROW_WARPS) * WARP_COL_TILES * M) * GLOBAL_MEM_STRIDE +
        block_tile_j * N + (warpId % BLOCK_ROW_WARPS) * WARP_ROW_TILES * N;
    uint8_t *dst_gmem_warp_stream_ptr = &D[gmem_idx];

    int *shmem_lane_stream_ptr = 
        shmem_warp_stream_ptr + 
        laneId * LANE_ROW_STRIDE / WARP_STRIDE * LANE_COL_STRIDE * SHMEM_STRIDE +
        laneId * LANE_ROW_STRIDE % WARP_STRIDE;
    uint8_t *dst_gmem_lane_stream_ptr = 
        dst_gmem_warp_stream_ptr + 
        laneId * LANE_ROW_STRIDE / WARP_STRIDE * LANE_COL_STRIDE * GLOBAL_MEM_STRIDE +
        laneId * LANE_ROW_STRIDE % WARP_STRIDE;

#pragma unroll
    for (int i = 0; i < LANE_COL_STRIDE; i++){
      for(int k = 0; k < LANE_ROW_STRIDE; k++){
        *(dst_gmem_lane_stream_ptr + GLOBAL_MEM_STRIDE * i + k) =
          (uint8_t)(*(shmem_lane_stream_ptr + SHMEM_STRIDE * i + k));
      }
    }

    __syncthreads();

}

void HostComputation(uint8_t* A, uint8_t* W, uint8_t* D, int alpha, int integer){
  for(int i = 0; i < M_GLOBAL; i++){
      for(int j = 0; j < N_GLOBAL; j++){
          int cSub = 0;
          for(int k = 0; k < K_GLOBAL; k++){
              cSub += (int)A[i*K_GLOBAL+k] * (int)W[j*K_GLOBAL+k];
          }
          D[i*N_GLOBAL+j] = (uint8_t)((cSub *alpha) >> integer);
      }
  }
}

void HostComputation_sparse(uint8_t* A, int* row, int* col, uint8_t* val, uint8_t* D, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int alpha, int integer){
  size_t mem_size_B = sizeof(uint8_t) * N_GLOBAL * K_GLOBAL;
  uint8_t* B = (uint8_t*)malloc(mem_size_B);
  std::memset(B, 0, mem_size_B);
  int ROW_BLOCK_NUM = N_GLOBAL / BLOCK_SIZE_N;
  for(int i = 0; i < ROW_BLOCK_NUM; i++){
      int index_start = row[i], index_end = row[i+1];
      for(int index = index_start; index < index_end; index += 1){
          int col_index = col[index] * BLOCK_SIZE_K;
          int row_index = i * BLOCK_SIZE_N;
          uint8_t* val_ptr = val + index * BLOCK_SIZE_N * BLOCK_SIZE_K;
          for(int n = row_index; n < row_index + BLOCK_SIZE_N; n += 1){
            for(int k = col_index; k < col_index + BLOCK_SIZE_K; k += 1){
              B[n * K_GLOBAL + k] = *(val_ptr + (n-row_index) * BLOCK_SIZE_K + (k-col_index)); 
            }
          }
      }
  }
  for(int i = 0; i < M_GLOBAL; i += 1){
      for(int j = 0; j < N_GLOBAL; j += 1){
          int cSub = 0;
          for(int k = 0; k < K_GLOBAL; k += 1){
              cSub += A[i * K_GLOBAL + k] * B[j * K_GLOBAL + k];
          }
          D[i * N_GLOBAL + j] = (uint8_t)((cSub *alpha) >> integer);
      }
  }
}

int matrixMultiply(int sparsity){
  int dev = 0;

  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

  int size_A = M_GLOBAL * K_GLOBAL;
  int size_C = M_GLOBAL * N_GLOBAL;

  int mem_size_A = sizeof(uint8_t) * size_A;
  int mem_size_C = sizeof(uint8_t) * size_C;
  int mem_size_D = sizeof(uint8_t) * size_C;
  int mem_size_bias = sizeof(int) * N;

  // memory size of row, col, val
  int mem_size_row = sizeof(int) * M_GLOBAL;
  int mem_size_col = sizeof(int) * M_GLOBAL * N_GLOBAL;
  int mem_size_val = sizeof(uint8_t) * M_GLOBAL * N_GLOBAL;

  uint8_t* h_A = (uint8_t*)malloc(mem_size_A);
  uint8_t* h_C = (uint8_t*)malloc(mem_size_C);
  uint8_t* h_D = (uint8_t*)malloc(mem_size_D);
  uint8_t* h_result = (uint8_t*)malloc(mem_size_C);
  int* h_bias = (int*)malloc(mem_size_bias);

  // memory allocation of row, col, val
  int* h_row = (int*)malloc(mem_size_row);
  int* h_col = (int*)malloc(mem_size_col);
  uint8_t* h_val = (uint8_t*)malloc(mem_size_val);

  // load data
  std::string row_path = "bcsr_row.bin";
  std::string col_path = "bcsr_col.bin";
  std::string val_path = "bcsr_val.bin";

  int alpha = 1;
  int integer = 0;

  float* d_A_32;
  float* d_C_32;
  uint8_t* d_A;
  uint8_t* d_D;
  int* d_row;
  int* d_col;
  uint8_t* d_val;

  const int BLOCK_SIZE_X = N * BLOCK_ROW_TILES;
  const int BLOCK_SIZE_Y = K * CHUNK_K;

  int k_block = K_GLOBAL / BLOCK_SIZE_Y;
  int n_block = N_GLOBAL / BLOCK_SIZE_X;
  int nnz_block = k_block * n_block * ((100-sparsity) / 100.0);
  int stride = k_block * n_block / nnz_block;

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  float msecTotal = 0;
  int nIter = 100;

  for(int i = 0; i < M_GLOBAL; i++){
      for(int j = 0; j < K_GLOBAL; j++){
        h_A[i * K_GLOBAL + j] = rand()%5;
      }
  }

  for(int i = 0; i < N; i++){
    h_bias[i] = 0;
  }

  load_from_file((char*)h_row, mem_size_row, row_path);
  load_from_file((char*)h_col, mem_size_col, col_path);
  load_from_file((char*)h_val, mem_size_val, val_path);

  for(int i = 0; i < M_GLOBAL; i++){
      for(int j = 0; j < N_GLOBAL; j++){
          h_C[i * N_GLOBAL + j] = 0;
      }
  }

  printf("host init successfully!\n");

  checkCudaErrors(cudaMalloc(&d_A, mem_size_A));
  checkCudaErrors(cudaMalloc(&d_D, mem_size_D));

  checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));

  // device csr memory copy
  checkCudaErrors(cudaMalloc(&d_row, mem_size_row));
  checkCudaErrors(cudaMalloc(&d_col, mem_size_col));
  checkCudaErrors(cudaMalloc(&d_val, mem_size_val));

  checkCudaErrors(cudaMalloc(&d_A_32, size_A * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_C_32, size_C * sizeof(float)));

  checkCudaErrors(cudaMemcpy(d_row, h_row, mem_size_row, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_col, h_col, mem_size_col, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_val, h_val, mem_size_val, cudaMemcpyHostToDevice));

  printf("Device init successfully!\n");
  printf("val_csr init successfully\n");

  enum {
    // Compute the right amount of shared memory to request.
    // We need shared memory to hold per-CTA C and D matrix tiles, and to cache
    // per-CTA chunks
    // of the A and B matrices. Therefore, the right amount to request is the
    // maximum of those
    // two numbers.
    SHMEM_SZ = MAX(sizeof(uint8_t) * (BLOCK_COL_TILES * M + BLOCK_ROW_TILES * N) *
                       (CHUNK_K * K + SKEW_UINT8),
                   M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N *
                       (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(int))
  };

  printf("Need shared memory size: %d\n", SHMEM_SZ);

  int block_size_col = BLOCK_COL_TILES * M;
  int block_size_row = BLOCK_ROW_TILES * N;

  int block_num = (M_GLOBAL * N_GLOBAL) / (block_size_col * block_size_row);

    dim3 blocksPerGrid(32, 8);
    dim3 threadsPerBlock(128);

  printf("Begin to run MatrixMulCUDA_8bit() function....\n");
  printf("Computing... using high performance kernel compute_gemm_imma \n");

  for(int run = 0; run < nIter; run++){
    matrixMultiplication<<<blocksPerGrid, threadsPerBlock>>>(d_A_32, d_A_32, d_C_32);
    checkKernelErrors(
        (compute_gemm_imma<<<block_num, THREADS_PER_BLOCK>>>(d_A, d_val, d_row, d_col, d_D, alpha, integer)));
  }

  checkCudaErrors(cudaEventRecord(start));
  // If enough shared memory available on the GPU use high performant kernel

  for(int run = 0; run < nIter; run++){
    matrixMultiplication<<<blocksPerGrid, threadsPerBlock>>>(d_A_32, d_A_32, d_C_32);
    checkKernelErrors(
        (compute_gemm_imma<<<block_num, THREADS_PER_BLOCK>>>(d_A, d_val, d_row, d_col, d_D, alpha, integer)));
  }
  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  float msecPerMatrixMul = msecTotal / nIter;

  printf("8bit quantize sparse kernel gemm Time= %f msec\n", msecPerMatrixMul);

  cudaFree(d_A);
  cudaFree(d_D);
  cudaFree(d_val);
  cudaFree(d_row);
  cudaFree(d_col);

  free(h_A);
  free(h_C);
  free(h_D);
  free(h_result);
  free(h_val);
  free(h_row);
  free(h_col);

  return EXIT_SUCCESS;
}

/**
* Program main
*/
int main(int argc, char **argv) {
  printf("[Block Sparse Matrix Multiply Using TensorCore] - Starting...\n");

  // This will pick the best possible CUDA capable device, otherwise
  // override the device ID based on input provided at the command line

  int sparsity = 60;
  printf("M %d, N %d, K %d\n", M_GLOBAL, N_GLOBAL, K_GLOBAL);
  matrixMultiply(sparsity);
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_0(int blockIdx_y, float *rC, float* input1) {
    rC[9] += 2.0f * input1[(1 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 4.0f * input1[(6 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 12.0f * input1[(11 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[16] += 19.0f * input1[(13 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 11.0f * input1[(13 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 8.0f * input1[(14 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 6.0f * input1[(14 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[16] += 7.0f * input1[(15 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 15.0f * input1[(15 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 5.0f * input1[(16 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 13.0f * input1[(17 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[19] += 1.0f * input1[(17 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 3.0f * input1[(18 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 2.0f * input1[(21 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 17.0f * input1[(21 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 5.0f * input1[(21 * 32 + 26 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[16] += 18.0f * input1[(22 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 17.0f * input1[(24 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[10] += 16.0f * input1[(25 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 13.0f * input1[(26 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 20.0f * input1[(27 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 4.0f * input1[(28 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 15.0f * input1[(28 * 32 + 25 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 17.0f * input1[(28 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 9.0f * input1[(28 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 20.0f * input1[(29 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 17.0f * input1[(29 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 7.0f * input1[(29 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 10.0f * input1[(30 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_1(int blockIdx_y, float *rC, float* input1) {
    rC[4] += 5.0f * input1[(0 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 5.0f * input1[(0 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 16.0f * input1[(0 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 11.0f * input1[(1 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 1.0f * input1[(1 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[10] += 16.0f * input1[(2 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 18.0f * input1[(4 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 6.0f * input1[(7 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 14.0f * input1[(8 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 7.0f * input1[(9 * 32 + 26 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 17.0f * input1[(11 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[19] += 16.0f * input1[(11 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 19.0f * input1[(12 * 32 + 0 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 12.0f * input1[(12 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 1.0f * input1[(12 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 11.0f * input1[(13 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 18.0f * input1[(13 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 5.0f * input1[(14 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 7.0f * input1[(16 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 18.0f * input1[(21 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 6.0f * input1[(21 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[31] += 8.0f * input1[(21 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 9.0f * input1[(25 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[5] += 1.0f * input1[(26 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 15.0f * input1[(26 * 32 + 10 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 6.0f * input1[(27 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 15.0f * input1[(27 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 11.0f * input1[(28 * 32 + 0 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 20.0f * input1[(28 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 8.0f * input1[(28 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 10.0f * input1[(30 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[31] += 18.0f * input1[(30 * 32 + 26 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[31] += 5.0f * input1[(31 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_2(int blockIdx_y, float *rC, float* input1) {
    rC[27] += 14.0f * input1[(0 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 13.0f * input1[(2 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 16.0f * input1[(2 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 10.0f * input1[(3 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 13.0f * input1[(4 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 16.0f * input1[(5 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 16.0f * input1[(6 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 6.0f * input1[(6 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 20.0f * input1[(6 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 15.0f * input1[(7 * 32 + 26 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 4.0f * input1[(7 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 7.0f * input1[(8 * 32 + 19 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 1.0f * input1[(8 * 32 + 19 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 4.0f * input1[(9 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 5.0f * input1[(9 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[31] += 16.0f * input1[(11 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 15.0f * input1[(12 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[5] += 4.0f * input1[(12 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 12.0f * input1[(13 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 13.0f * input1[(13 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[12] += 4.0f * input1[(13 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 3.0f * input1[(14 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 8.0f * input1[(14 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 3.0f * input1[(14 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 14.0f * input1[(17 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 9.0f * input1[(18 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 13.0f * input1[(19 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 9.0f * input1[(19 * 32 + 19 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[5] += 6.0f * input1[(21 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 7.0f * input1[(21 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 2.0f * input1[(22 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 9.0f * input1[(23 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 3.0f * input1[(23 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 6.0f * input1[(25 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 9.0f * input1[(26 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 11.0f * input1[(26 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 15.0f * input1[(27 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 19.0f * input1[(31 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_3(int blockIdx_y, float *rC, float* input1) {
    rC[29] += 19.0f * input1[(0 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 16.0f * input1[(1 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[16] += 20.0f * input1[(2 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 15.0f * input1[(3 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[16] += 20.0f * input1[(5 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 7.0f * input1[(5 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[31] += 10.0f * input1[(5 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 6.0f * input1[(6 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 20.0f * input1[(6 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 19.0f * input1[(7 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 18.0f * input1[(9 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 17.0f * input1[(10 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 12.0f * input1[(13 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 2.0f * input1[(13 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[12] += 10.0f * input1[(14 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[31] += 15.0f * input1[(14 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[12] += 6.0f * input1[(16 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[19] += 1.0f * input1[(16 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 3.0f * input1[(18 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[19] += 8.0f * input1[(21 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 13.0f * input1[(22 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 7.0f * input1[(22 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 5.0f * input1[(23 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 5.0f * input1[(27 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 7.0f * input1[(30 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 1.0f * input1[(30 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 18.0f * input1[(31 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_4(int blockIdx_y, float *rC, float* input1) {
    rC[12] += 10.0f * input1[(0 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[16] += 20.0f * input1[(0 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 13.0f * input1[(2 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[10] += 12.0f * input1[(2 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 3.0f * input1[(4 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[16] += 20.0f * input1[(4 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[5] += 11.0f * input1[(5 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 9.0f * input1[(6 * 32 + 10 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[16] += 10.0f * input1[(6 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 20.0f * input1[(6 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 13.0f * input1[(8 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 8.0f * input1[(8 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 18.0f * input1[(9 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 16.0f * input1[(10 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 14.0f * input1[(10 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 15.0f * input1[(12 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 19.0f * input1[(12 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 2.0f * input1[(15 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 8.0f * input1[(18 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 9.0f * input1[(18 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[10] += 11.0f * input1[(20 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 11.0f * input1[(21 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 19.0f * input1[(21 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 17.0f * input1[(22 * 32 + 26 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 17.0f * input1[(24 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 18.0f * input1[(24 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 9.0f * input1[(24 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 8.0f * input1[(25 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 2.0f * input1[(26 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 15.0f * input1[(27 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 20.0f * input1[(28 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[19] += 12.0f * input1[(28 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 1.0f * input1[(28 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[10] += 2.0f * input1[(29 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[16] += 14.0f * input1[(29 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[5] += 2.0f * input1[(30 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 4.0f * input1[(30 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 14.0f * input1[(30 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_5(int blockIdx_y, float *rC, float* input1) {
    rC[30] += 6.0f * input1[(1 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 15.0f * input1[(3 * 32 + 19 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 9.0f * input1[(3 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 3.0f * input1[(3 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 17.0f * input1[(4 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 9.0f * input1[(4 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 8.0f * input1[(5 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[19] += 9.0f * input1[(7 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 5.0f * input1[(7 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 16.0f * input1[(8 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 10.0f * input1[(9 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 1.0f * input1[(9 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 13.0f * input1[(10 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[10] += 16.0f * input1[(11 * 32 + 19 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 7.0f * input1[(12 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 20.0f * input1[(16 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 20.0f * input1[(17 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 4.0f * input1[(17 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[5] += 16.0f * input1[(18 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 4.0f * input1[(19 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 12.0f * input1[(19 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 8.0f * input1[(21 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 1.0f * input1[(21 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 12.0f * input1[(21 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 17.0f * input1[(22 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[16] += 13.0f * input1[(23 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 14.0f * input1[(24 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 17.0f * input1[(27 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 6.0f * input1[(27 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 10.0f * input1[(28 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 10.0f * input1[(28 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[16] += 10.0f * input1[(29 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 15.0f * input1[(29 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 9.0f * input1[(29 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_6(int blockIdx_y, float *rC, float* input1) {
    rC[25] += 12.0f * input1[(0 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 14.0f * input1[(2 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 7.0f * input1[(2 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 19.0f * input1[(3 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 6.0f * input1[(4 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 15.0f * input1[(4 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 3.0f * input1[(7 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 2.0f * input1[(9 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[12] += 4.0f * input1[(11 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 16.0f * input1[(12 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[5] += 7.0f * input1[(13 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 7.0f * input1[(13 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 20.0f * input1[(15 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 8.0f * input1[(16 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 15.0f * input1[(18 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 11.0f * input1[(18 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 7.0f * input1[(18 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 20.0f * input1[(19 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[14] += 1.0f * input1[(19 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 18.0f * input1[(19 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 13.0f * input1[(20 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 16.0f * input1[(23 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[10] += 16.0f * input1[(23 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 20.0f * input1[(24 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 7.0f * input1[(25 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 5.0f * input1[(26 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 13.0f * input1[(27 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 20.0f * input1[(28 * 32 + 26 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_7(int blockIdx_y, float *rC, float* input1) {
    rC[5] += 16.0f * input1[(1 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 15.0f * input1[(1 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 12.0f * input1[(1 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 4.0f * input1[(1 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 7.0f * input1[(3 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 7.0f * input1[(4 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 13.0f * input1[(4 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 2.0f * input1[(5 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 12.0f * input1[(8 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 5.0f * input1[(9 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 7.0f * input1[(12 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 5.0f * input1[(13 * 32 + 10 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 1.0f * input1[(16 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 15.0f * input1[(16 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 15.0f * input1[(18 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 2.0f * input1[(20 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 14.0f * input1[(23 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 12.0f * input1[(23 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 6.0f * input1[(23 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 2.0f * input1[(23 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[16] += 15.0f * input1[(24 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 3.0f * input1[(25 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 8.0f * input1[(25 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 19.0f * input1[(29 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[19] += 18.0f * input1[(30 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[12] += 11.0f * input1[(31 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[19] += 1.0f * input1[(31 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_8(int blockIdx_y, float *rC, float* input1) {
    rC[12] += 1.0f * input1[(2 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 14.0f * input1[(5 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[16] += 11.0f * input1[(5 * 32 + 19 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 12.0f * input1[(5 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 12.0f * input1[(5 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[31] += 16.0f * input1[(5 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 9.0f * input1[(7 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 4.0f * input1[(7 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 4.0f * input1[(7 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 3.0f * input1[(7 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 2.0f * input1[(9 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 2.0f * input1[(10 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 7.0f * input1[(11 * 32 + 10 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 5.0f * input1[(11 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 15.0f * input1[(12 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 10.0f * input1[(12 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 18.0f * input1[(12 * 32 + 0 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 19.0f * input1[(13 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 3.0f * input1[(14 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 5.0f * input1[(15 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 20.0f * input1[(16 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 8.0f * input1[(16 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 8.0f * input1[(18 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 1.0f * input1[(20 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 4.0f * input1[(21 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 19.0f * input1[(22 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 11.0f * input1[(23 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 6.0f * input1[(23 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 18.0f * input1[(23 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[5] += 20.0f * input1[(24 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 20.0f * input1[(26 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 12.0f * input1[(27 * 32 + 25 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 17.0f * input1[(28 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 10.0f * input1[(28 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 5.0f * input1[(29 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 1.0f * input1[(30 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 3.0f * input1[(30 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_9(int blockIdx_y, float *rC, float* input1) {
    rC[29] += 15.0f * input1[(0 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[10] += 7.0f * input1[(1 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 14.0f * input1[(1 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 13.0f * input1[(3 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 9.0f * input1[(3 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 9.0f * input1[(4 * 32 + 19 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 19.0f * input1[(5 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 3.0f * input1[(6 * 32 + 0 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[12] += 3.0f * input1[(6 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 19.0f * input1[(6 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 18.0f * input1[(7 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 15.0f * input1[(7 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 6.0f * input1[(8 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 13.0f * input1[(8 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[5] += 7.0f * input1[(9 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 6.0f * input1[(9 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 5.0f * input1[(9 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[14] += 19.0f * input1[(10 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 10.0f * input1[(10 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 12.0f * input1[(11 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 12.0f * input1[(11 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 7.0f * input1[(12 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 18.0f * input1[(13 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 9.0f * input1[(13 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 9.0f * input1[(13 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 16.0f * input1[(14 * 32 + 0 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 1.0f * input1[(16 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 11.0f * input1[(17 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 3.0f * input1[(18 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 14.0f * input1[(18 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[14] += 16.0f * input1[(21 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 16.0f * input1[(22 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[10] += 19.0f * input1[(22 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 20.0f * input1[(23 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[12] += 11.0f * input1[(23 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 8.0f * input1[(23 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 16.0f * input1[(23 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 4.0f * input1[(23 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[16] += 9.0f * input1[(24 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[14] += 1.0f * input1[(25 * 32 + 25 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 6.0f * input1[(27 * 32 + 19 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_10(int blockIdx_y, float *rC, float* input1) {
    rC[16] += 6.0f * input1[(0 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 8.0f * input1[(1 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[31] += 16.0f * input1[(1 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 3.0f * input1[(2 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 17.0f * input1[(2 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[31] += 3.0f * input1[(3 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 6.0f * input1[(4 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 9.0f * input1[(5 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 8.0f * input1[(5 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 10.0f * input1[(5 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 8.0f * input1[(5 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[19] += 4.0f * input1[(8 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[31] += 9.0f * input1[(8 * 32 + 25 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 3.0f * input1[(9 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 5.0f * input1[(12 * 32 + 0 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[10] += 3.0f * input1[(12 * 32 + 26 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[16] += 17.0f * input1[(12 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 10.0f * input1[(12 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 2.0f * input1[(12 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 13.0f * input1[(15 * 32 + 26 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 6.0f * input1[(21 * 32 + 10 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 4.0f * input1[(22 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 3.0f * input1[(23 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 8.0f * input1[(23 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 7.0f * input1[(25 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 20.0f * input1[(27 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[16] += 7.0f * input1[(28 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 16.0f * input1[(29 * 32 + 0 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 16.0f * input1[(29 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 15.0f * input1[(30 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 2.0f * input1[(31 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_11(int blockIdx_y, float *rC, float* input1) {
    rC[7] += 11.0f * input1[(0 * 32 + 19 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 14.0f * input1[(0 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[16] += 11.0f * input1[(1 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 15.0f * input1[(2 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 13.0f * input1[(3 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 17.0f * input1[(4 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 16.0f * input1[(4 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[10] += 15.0f * input1[(6 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 4.0f * input1[(7 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 13.0f * input1[(7 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[5] += 3.0f * input1[(9 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 10.0f * input1[(9 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 9.0f * input1[(9 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 4.0f * input1[(10 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[12] += 16.0f * input1[(11 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 18.0f * input1[(11 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 20.0f * input1[(12 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 18.0f * input1[(13 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 19.0f * input1[(14 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 19.0f * input1[(15 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 2.0f * input1[(15 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 4.0f * input1[(16 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 2.0f * input1[(16 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 4.0f * input1[(17 * 32 + 10 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 17.0f * input1[(18 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 3.0f * input1[(18 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 2.0f * input1[(18 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 16.0f * input1[(20 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 11.0f * input1[(20 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 16.0f * input1[(21 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 20.0f * input1[(23 * 32 + 10 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[12] += 7.0f * input1[(23 * 32 + 25 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 20.0f * input1[(26 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 12.0f * input1[(27 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 12.0f * input1[(27 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 10.0f * input1[(31 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 16.0f * input1[(31 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[10] += 4.0f * input1[(31 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 4.0f * input1[(31 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_12(int blockIdx_y, float *rC, float* input1) {
    rC[12] += 3.0f * input1[(0 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 14.0f * input1[(1 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 16.0f * input1[(6 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 2.0f * input1[(6 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 10.0f * input1[(7 * 32 + 10 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 1.0f * input1[(7 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 2.0f * input1[(8 * 32 + 10 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 2.0f * input1[(11 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[12] += 7.0f * input1[(11 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 20.0f * input1[(11 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 14.0f * input1[(13 * 32 + 19 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 9.0f * input1[(14 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 2.0f * input1[(14 * 32 + 26 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 3.0f * input1[(14 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 16.0f * input1[(15 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 14.0f * input1[(15 * 32 + 0 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[5] += 9.0f * input1[(16 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 19.0f * input1[(16 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 8.0f * input1[(17 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[16] += 9.0f * input1[(18 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[19] += 3.0f * input1[(23 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 18.0f * input1[(23 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 16.0f * input1[(26 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 7.0f * input1[(29 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 14.0f * input1[(29 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 17.0f * input1[(30 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_13(int blockIdx_y, float *rC, float* input1) {
    rC[5] += 16.0f * input1[(2 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 4.0f * input1[(2 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 20.0f * input1[(3 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[31] += 12.0f * input1[(4 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[14] += 3.0f * input1[(6 * 32 + 0 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 2.0f * input1[(7 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 17.0f * input1[(8 * 32 + 26 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 14.0f * input1[(8 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 5.0f * input1[(9 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 19.0f * input1[(9 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 12.0f * input1[(10 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 11.0f * input1[(11 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 19.0f * input1[(12 * 32 + 19 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 13.0f * input1[(12 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[5] += 5.0f * input1[(13 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 15.0f * input1[(13 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 17.0f * input1[(14 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 2.0f * input1[(15 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 10.0f * input1[(17 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 7.0f * input1[(17 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 2.0f * input1[(18 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 4.0f * input1[(18 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 9.0f * input1[(18 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[10] += 4.0f * input1[(20 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 11.0f * input1[(20 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 17.0f * input1[(20 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 3.0f * input1[(22 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 9.0f * input1[(23 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 5.0f * input1[(24 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 5.0f * input1[(24 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 13.0f * input1[(24 * 32 + 10 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 2.0f * input1[(26 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 17.0f * input1[(28 * 32 + 26 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 8.0f * input1[(29 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 15.0f * input1[(29 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 15.0f * input1[(31 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 2.0f * input1[(31 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 14.0f * input1[(31 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_14(int blockIdx_y, float *rC, float* input1) {
    rC[1] += 12.0f * input1[(0 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 2.0f * input1[(2 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 13.0f * input1[(2 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 10.0f * input1[(6 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 10.0f * input1[(6 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 10.0f * input1[(6 * 32 + 25 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 13.0f * input1[(7 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 9.0f * input1[(8 * 32 + 25 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 8.0f * input1[(9 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 15.0f * input1[(9 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 16.0f * input1[(11 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 19.0f * input1[(11 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[12] += 16.0f * input1[(13 * 32 + 19 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 18.0f * input1[(14 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 6.0f * input1[(15 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 1.0f * input1[(18 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 11.0f * input1[(20 * 32 + 26 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 19.0f * input1[(20 * 32 + 10 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 17.0f * input1[(21 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 2.0f * input1[(22 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 13.0f * input1[(23 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 12.0f * input1[(24 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[10] += 17.0f * input1[(24 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 15.0f * input1[(25 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 12.0f * input1[(25 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 15.0f * input1[(26 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 18.0f * input1[(26 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 12.0f * input1[(27 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 6.0f * input1[(27 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 15.0f * input1[(28 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[19] += 19.0f * input1[(30 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 7.0f * input1[(31 * 32 + 25 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 20.0f * input1[(31 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_15(int blockIdx_y, float *rC, float* input1) {
    rC[16] += 9.0f * input1[(0 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 11.0f * input1[(0 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 7.0f * input1[(1 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 4.0f * input1[(1 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 16.0f * input1[(2 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 9.0f * input1[(3 * 32 + 25 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[31] += 3.0f * input1[(3 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 9.0f * input1[(5 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 12.0f * input1[(6 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 10.0f * input1[(6 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 5.0f * input1[(7 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 13.0f * input1[(7 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 4.0f * input1[(8 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 17.0f * input1[(12 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 17.0f * input1[(13 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 3.0f * input1[(14 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 2.0f * input1[(16 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 17.0f * input1[(17 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 16.0f * input1[(17 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 13.0f * input1[(18 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 2.0f * input1[(19 * 32 + 0 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 3.0f * input1[(19 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 9.0f * input1[(19 * 32 + 0 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[10] += 14.0f * input1[(22 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 20.0f * input1[(23 * 32 + 26 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 14.0f * input1[(24 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 3.0f * input1[(25 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 7.0f * input1[(26 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 18.0f * input1[(28 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 3.0f * input1[(29 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 8.0f * input1[(29 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[19] += 15.0f * input1[(30 * 32 + 10 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_16(int blockIdx_y, float *rC, float* input1) {
    rC[17] += 9.0f * input1[(0 * 32 + 10 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 14.0f * input1[(2 * 32 + 0 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 17.0f * input1[(3 * 32 + 26 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 2.0f * input1[(4 * 32 + 25 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 14.0f * input1[(7 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[31] += 6.0f * input1[(10 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 4.0f * input1[(11 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 14.0f * input1[(12 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[12] += 13.0f * input1[(13 * 32 + 25 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 19.0f * input1[(15 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 13.0f * input1[(17 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[16] += 5.0f * input1[(18 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 16.0f * input1[(18 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 13.0f * input1[(18 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[10] += 13.0f * input1[(20 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[16] += 20.0f * input1[(20 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 4.0f * input1[(20 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 10.0f * input1[(21 * 32 + 25 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 15.0f * input1[(21 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 14.0f * input1[(24 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 4.0f * input1[(25 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 9.0f * input1[(26 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 1.0f * input1[(27 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 14.0f * input1[(29 * 32 + 10 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[16] += 14.0f * input1[(30 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_17(int blockIdx_y, float *rC, float* input1) {
    rC[25] += 19.0f * input1[(0 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 14.0f * input1[(1 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 8.0f * input1[(2 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 4.0f * input1[(3 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 13.0f * input1[(4 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 10.0f * input1[(5 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[31] += 8.0f * input1[(6 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 16.0f * input1[(8 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 14.0f * input1[(9 * 32 + 19 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 17.0f * input1[(13 * 32 + 0 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 15.0f * input1[(16 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 12.0f * input1[(18 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 18.0f * input1[(18 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[14] += 2.0f * input1[(18 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 16.0f * input1[(21 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[19] += 1.0f * input1[(21 * 32 + 10 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 11.0f * input1[(21 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 12.0f * input1[(24 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 15.0f * input1[(24 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 16.0f * input1[(24 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 6.0f * input1[(25 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[19] += 15.0f * input1[(25 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 4.0f * input1[(26 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 16.0f * input1[(26 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 16.0f * input1[(28 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 15.0f * input1[(29 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 4.0f * input1[(30 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 2.0f * input1[(30 * 32 + 25 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 18.0f * input1[(30 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 1.0f * input1[(30 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_18(int blockIdx_y, float *rC, float* input1) {
    rC[5] += 5.0f * input1[(1 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 7.0f * input1[(1 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 5.0f * input1[(3 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 5.0f * input1[(5 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 11.0f * input1[(6 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 18.0f * input1[(7 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 14.0f * input1[(8 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 5.0f * input1[(11 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 13.0f * input1[(12 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 20.0f * input1[(13 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 13.0f * input1[(14 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 14.0f * input1[(15 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 13.0f * input1[(15 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 1.0f * input1[(16 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 9.0f * input1[(19 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 15.0f * input1[(19 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 19.0f * input1[(23 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 8.0f * input1[(24 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 3.0f * input1[(25 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 18.0f * input1[(25 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 12.0f * input1[(26 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[31] += 6.0f * input1[(30 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_19(int blockIdx_y, float *rC, float* input1) {
    rC[21] += 15.0f * input1[(2 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 7.0f * input1[(2 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 8.0f * input1[(3 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[31] += 16.0f * input1[(4 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 4.0f * input1[(6 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 8.0f * input1[(8 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 15.0f * input1[(9 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 2.0f * input1[(11 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 13.0f * input1[(11 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 11.0f * input1[(11 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 19.0f * input1[(15 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 10.0f * input1[(15 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[31] += 16.0f * input1[(15 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 15.0f * input1[(16 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 20.0f * input1[(18 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 5.0f * input1[(20 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 13.0f * input1[(20 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 12.0f * input1[(21 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 19.0f * input1[(21 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 15.0f * input1[(21 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[12] += 4.0f * input1[(23 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 9.0f * input1[(23 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 17.0f * input1[(27 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[12] += 10.0f * input1[(27 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 4.0f * input1[(27 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 4.0f * input1[(29 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 7.0f * input1[(31 * 32 + 26 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[14] += 18.0f * input1[(31 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 2.0f * input1[(31 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_20(int blockIdx_y, float *rC, float* input1) {
    rC[2] += 3.0f * input1[(2 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 4.0f * input1[(2 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 8.0f * input1[(2 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[19] += 6.0f * input1[(3 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[12] += 1.0f * input1[(4 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 8.0f * input1[(4 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 1.0f * input1[(5 * 32 + 10 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 3.0f * input1[(5 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[5] += 14.0f * input1[(6 * 32 + 10 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[14] += 13.0f * input1[(7 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 11.0f * input1[(8 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[14] += 16.0f * input1[(9 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 18.0f * input1[(10 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 14.0f * input1[(10 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 16.0f * input1[(10 * 32 + 19 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 5.0f * input1[(11 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 9.0f * input1[(12 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 13.0f * input1[(12 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[12] += 12.0f * input1[(13 * 32 + 19 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[14] += 10.0f * input1[(13 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 15.0f * input1[(14 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 12.0f * input1[(16 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 11.0f * input1[(17 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 2.0f * input1[(18 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[16] += 14.0f * input1[(19 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 10.0f * input1[(20 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 4.0f * input1[(20 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 10.0f * input1[(22 * 32 + 26 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 13.0f * input1[(22 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 2.0f * input1[(22 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[14] += 3.0f * input1[(23 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 19.0f * input1[(27 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 14.0f * input1[(28 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 8.0f * input1[(29 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 15.0f * input1[(31 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 10.0f * input1[(31 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_21(int blockIdx_y, float *rC, float* input1) {
    rC[31] += 5.0f * input1[(0 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 1.0f * input1[(1 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 17.0f * input1[(2 * 32 + 19 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 20.0f * input1[(3 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 10.0f * input1[(4 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 16.0f * input1[(4 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 1.0f * input1[(4 * 32 + 25 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 19.0f * input1[(5 * 32 + 0 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[14] += 18.0f * input1[(8 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 10.0f * input1[(8 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[12] += 10.0f * input1[(10 * 32 + 19 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 5.0f * input1[(12 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 9.0f * input1[(12 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 3.0f * input1[(12 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 15.0f * input1[(12 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 15.0f * input1[(13 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 15.0f * input1[(13 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 16.0f * input1[(13 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[16] += 10.0f * input1[(14 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 20.0f * input1[(14 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[10] += 16.0f * input1[(15 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 7.0f * input1[(16 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 9.0f * input1[(19 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 5.0f * input1[(20 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 16.0f * input1[(20 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 8.0f * input1[(22 * 32 + 19 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 9.0f * input1[(22 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 6.0f * input1[(25 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 18.0f * input1[(27 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[31] += 10.0f * input1[(27 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 1.0f * input1[(31 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 8.0f * input1[(31 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_22(int blockIdx_y, float *rC, float* input1) {
    rC[14] += 17.0f * input1[(0 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 11.0f * input1[(2 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 8.0f * input1[(3 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 10.0f * input1[(3 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 7.0f * input1[(4 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 11.0f * input1[(4 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 1.0f * input1[(5 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 7.0f * input1[(5 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 17.0f * input1[(7 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 3.0f * input1[(7 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 3.0f * input1[(7 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 20.0f * input1[(7 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 15.0f * input1[(8 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 16.0f * input1[(8 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 20.0f * input1[(10 * 32 + 10 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 3.0f * input1[(10 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 12.0f * input1[(11 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 9.0f * input1[(12 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 6.0f * input1[(14 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 17.0f * input1[(17 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 5.0f * input1[(18 * 32 + 25 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 8.0f * input1[(19 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 11.0f * input1[(20 * 32 + 10 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 12.0f * input1[(23 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 14.0f * input1[(24 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[19] += 20.0f * input1[(26 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 6.0f * input1[(27 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 9.0f * input1[(30 * 32 + 19 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 2.0f * input1[(30 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 9.0f * input1[(31 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_23(int blockIdx_y, float *rC, float* input1) {
    rC[28] += 2.0f * input1[(0 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 1.0f * input1[(1 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 4.0f * input1[(2 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 16.0f * input1[(3 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 7.0f * input1[(4 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 15.0f * input1[(4 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 18.0f * input1[(5 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 3.0f * input1[(5 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 9.0f * input1[(6 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 11.0f * input1[(6 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 6.0f * input1[(7 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 14.0f * input1[(7 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 14.0f * input1[(7 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 3.0f * input1[(9 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[14] += 2.0f * input1[(10 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 17.0f * input1[(11 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 12.0f * input1[(11 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 10.0f * input1[(12 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 5.0f * input1[(12 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 2.0f * input1[(13 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[12] += 1.0f * input1[(13 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 4.0f * input1[(14 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 4.0f * input1[(14 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 19.0f * input1[(17 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 5.0f * input1[(17 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 20.0f * input1[(18 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 6.0f * input1[(19 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 20.0f * input1[(20 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 1.0f * input1[(22 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 9.0f * input1[(26 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 11.0f * input1[(30 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_24(int blockIdx_y, float *rC, float* input1) {
    rC[23] += 12.0f * input1[(0 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 14.0f * input1[(0 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 7.0f * input1[(0 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[12] += 8.0f * input1[(1 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 6.0f * input1[(3 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 9.0f * input1[(4 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 2.0f * input1[(4 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 6.0f * input1[(7 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 13.0f * input1[(8 * 32 + 25 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 16.0f * input1[(9 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 11.0f * input1[(11 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 4.0f * input1[(15 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 12.0f * input1[(15 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 17.0f * input1[(15 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 10.0f * input1[(16 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 2.0f * input1[(17 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[12] += 7.0f * input1[(19 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 18.0f * input1[(19 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 7.0f * input1[(19 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[5] += 7.0f * input1[(20 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[12] += 2.0f * input1[(20 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[5] += 20.0f * input1[(21 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 5.0f * input1[(21 * 32 + 0 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 3.0f * input1[(22 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 15.0f * input1[(22 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 8.0f * input1[(23 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[5] += 18.0f * input1[(23 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 9.0f * input1[(23 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 6.0f * input1[(23 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[31] += 5.0f * input1[(23 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 18.0f * input1[(24 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 13.0f * input1[(24 * 32 + 25 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 3.0f * input1[(29 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 7.0f * input1[(30 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 11.0f * input1[(30 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 7.0f * input1[(31 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_25(int blockIdx_y, float *rC, float* input1) {
    rC[29] += 9.0f * input1[(0 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 13.0f * input1[(2 * 32 + 19 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 18.0f * input1[(2 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 18.0f * input1[(3 * 32 + 26 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 20.0f * input1[(4 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[19] += 7.0f * input1[(5 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[14] += 10.0f * input1[(6 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 15.0f * input1[(6 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 3.0f * input1[(6 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 13.0f * input1[(7 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 11.0f * input1[(7 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 15.0f * input1[(8 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 1.0f * input1[(8 * 32 + 0 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 5.0f * input1[(9 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 14.0f * input1[(10 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 11.0f * input1[(11 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 11.0f * input1[(11 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[31] += 11.0f * input1[(12 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 5.0f * input1[(13 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 12.0f * input1[(17 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[12] += 7.0f * input1[(18 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 16.0f * input1[(18 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 6.0f * input1[(19 * 32 + 26 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 11.0f * input1[(19 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 14.0f * input1[(19 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[5] += 5.0f * input1[(21 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[12] += 14.0f * input1[(21 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 9.0f * input1[(23 * 32 + 25 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 10.0f * input1[(24 * 32 + 10 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[19] += 5.0f * input1[(24 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[14] += 7.0f * input1[(27 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 3.0f * input1[(27 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 18.0f * input1[(28 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[10] += 3.0f * input1[(29 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 13.0f * input1[(30 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 10.0f * input1[(30 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 17.0f * input1[(31 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_26(int blockIdx_y, float *rC, float* input1) {
    rC[17] += 12.0f * input1[(0 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 14.0f * input1[(0 * 32 + 10 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 17.0f * input1[(1 * 32 + 10 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 10.0f * input1[(2 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 16.0f * input1[(5 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 10.0f * input1[(6 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 12.0f * input1[(6 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 1.0f * input1[(8 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[10] += 3.0f * input1[(10 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 13.0f * input1[(10 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 18.0f * input1[(10 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[31] += 8.0f * input1[(10 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 9.0f * input1[(11 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 18.0f * input1[(13 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 15.0f * input1[(13 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 13.0f * input1[(14 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 14.0f * input1[(18 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 10.0f * input1[(18 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 10.0f * input1[(19 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 15.0f * input1[(19 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 2.0f * input1[(21 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 13.0f * input1[(22 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 14.0f * input1[(23 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 7.0f * input1[(25 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 10.0f * input1[(25 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 5.0f * input1[(25 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 4.0f * input1[(25 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 14.0f * input1[(27 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[19] += 12.0f * input1[(27 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 20.0f * input1[(29 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 3.0f * input1[(29 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 18.0f * input1[(30 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 2.0f * input1[(30 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_27(int blockIdx_y, float *rC, float* input1) {
    rC[0] += 5.0f * input1[(1 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 6.0f * input1[(2 * 32 + 25 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 9.0f * input1[(2 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 10.0f * input1[(3 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 20.0f * input1[(4 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 15.0f * input1[(5 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 6.0f * input1[(5 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 18.0f * input1[(5 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 16.0f * input1[(6 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[12] += 8.0f * input1[(6 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 17.0f * input1[(6 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 5.0f * input1[(7 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 8.0f * input1[(7 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[5] += 10.0f * input1[(8 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 9.0f * input1[(9 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[16] += 19.0f * input1[(9 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 14.0f * input1[(12 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 20.0f * input1[(12 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 13.0f * input1[(13 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 1.0f * input1[(14 * 32 + 19 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 13.0f * input1[(15 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[19] += 5.0f * input1[(15 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 6.0f * input1[(15 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 20.0f * input1[(18 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[14] += 9.0f * input1[(18 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 16.0f * input1[(18 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 18.0f * input1[(20 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 12.0f * input1[(22 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 2.0f * input1[(24 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 2.0f * input1[(25 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 17.0f * input1[(26 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 9.0f * input1[(26 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 16.0f * input1[(26 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 8.0f * input1[(26 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 4.0f * input1[(29 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 6.0f * input1[(30 * 32 + 16 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_28(int blockIdx_y, float *rC, float* input1) {
    rC[16] += 18.0f * input1[(0 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 18.0f * input1[(1 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 12.0f * input1[(2 * 32 + 10 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 10.0f * input1[(4 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 1.0f * input1[(5 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[31] += 4.0f * input1[(6 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 12.0f * input1[(7 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 12.0f * input1[(7 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[5] += 4.0f * input1[(10 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[14] += 19.0f * input1[(10 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 6.0f * input1[(10 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[31] += 19.0f * input1[(10 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[12] += 18.0f * input1[(11 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 10.0f * input1[(11 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 15.0f * input1[(14 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 8.0f * input1[(15 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 1.0f * input1[(15 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 6.0f * input1[(16 * 32 + 0 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 20.0f * input1[(17 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 1.0f * input1[(20 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 10.0f * input1[(21 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 8.0f * input1[(23 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[14] += 18.0f * input1[(24 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 10.0f * input1[(24 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 16.0f * input1[(26 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[10] += 12.0f * input1[(26 * 32 + 10 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[14] += 4.0f * input1[(28 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 18.0f * input1[(28 * 32 + 25 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 2.0f * input1[(28 * 32 + 22 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 20.0f * input1[(29 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 17.0f * input1[(30 * 32 + 4 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[14] += 7.0f * input1[(30 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 16.0f * input1[(31 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[16] += 19.0f * input1[(31 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_29(int blockIdx_y, float *rC, float* input1) {
    rC[8] += 12.0f * input1[(1 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 16.0f * input1[(1 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[31] += 7.0f * input1[(4 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 1.0f * input1[(5 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 15.0f * input1[(5 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 10.0f * input1[(5 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 12.0f * input1[(6 * 32 + 26 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 20.0f * input1[(6 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 14.0f * input1[(6 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 14.0f * input1[(7 * 32 + 26 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[16] += 4.0f * input1[(7 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 4.0f * input1[(12 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 9.0f * input1[(12 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[19] += 2.0f * input1[(12 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[6] += 15.0f * input1[(14 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[18] += 9.0f * input1[(14 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[31] += 14.0f * input1[(15 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 17.0f * input1[(16 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[31] += 17.0f * input1[(21 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 4.0f * input1[(23 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 15.0f * input1[(24 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 7.0f * input1[(28 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 1.0f * input1[(28 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 1.0f * input1[(28 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 10.0f * input1[(29 * 32 + 30 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_30(int blockIdx_y, float *rC, float* input1) {
    rC[29] += 6.0f * input1[(1 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 5.0f * input1[(2 * 32 + 26 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[9] += 6.0f * input1[(3 * 32 + 25 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[10] += 2.0f * input1[(3 * 32 + 3 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 6.0f * input1[(4 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[10] += 6.0f * input1[(5 * 32 + 21 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 14.0f * input1[(7 * 32 + 0 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 16.0f * input1[(8 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[22] += 7.0f * input1[(8 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 12.0f * input1[(12 * 32 + 12 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 1.0f * input1[(13 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[10] += 8.0f * input1[(14 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[30] += 5.0f * input1[(14 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[3] += 12.0f * input1[(15 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 4.0f * input1[(17 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 17.0f * input1[(17 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 11.0f * input1[(18 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[4] += 5.0f * input1[(19 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[12] += 8.0f * input1[(19 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 4.0f * input1[(20 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 15.0f * input1[(20 * 32 + 10 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[8] += 11.0f * input1[(22 * 32 + 10 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[27] += 6.0f * input1[(23 * 32 + 11 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 4.0f * input1[(25 * 32 + 23 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 4.0f * input1[(27 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 18.0f * input1[(29 * 32 + 15 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[13] += 4.0f * input1[(30 * 32 + 19 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 13.0f * input1[(30 * 32 + 8 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}

__forceinline__ __device__ void sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_31(int blockIdx_y, float *rC, float* input1) {
    rC[27] += 18.0f * input1[(2 * 32 + 5 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 20.0f * input1[(2 * 32 + 19 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 17.0f * input1[(3 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[2] += 14.0f * input1[(3 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 18.0f * input1[(7 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[19] += 9.0f * input1[(7 * 32 + 7 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[14] += 14.0f * input1[(9 * 32 + 9 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[21] += 12.0f * input1[(9 * 32 + 28 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[14] += 9.0f * input1[(10 * 32 + 13 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 18.0f * input1[(13 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 6.0f * input1[(14 * 32 + 27 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[19] += 5.0f * input1[(15 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[7] += 16.0f * input1[(18 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[26] += 7.0f * input1[(18 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 14.0f * input1[(18 * 32 + 1 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[0] += 18.0f * input1[(19 * 32 + 14 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[20] += 1.0f * input1[(20 * 32 + 6 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 2.0f * input1[(22 * 32 + 24 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[11] += 8.0f * input1[(23 * 32 + 31 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[17] += 15.0f * input1[(24 * 32 + 18 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[23] += 18.0f * input1[(27 * 32 + 2 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[29] += 3.0f * input1[(27 * 32 + 0 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[15] += 20.0f * input1[(29 * 32 + 20 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[25] += 17.0f * input1[(30 * 32 + 26 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[28] += 3.0f * input1[(30 * 32 + 25 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[1] += 19.0f * input1[(31 * 32 + 29 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
    rC[24] += 18.0f * input1[(31 * 32 + 17 * 1)*1024 + blockIdx_y * 128 + threadIdx.x * 1];
}


__global__ void matrixMultiplication(float *input0, float *input1, float *output0)
{
float rC[32];
memset(rC, 0, 32*sizeof(float));
if (blockIdx.x == 0) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_0(blockIdx.y, rC, input1);
}
if (blockIdx.x == 1) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_1(blockIdx.y, rC, input1);
}
if (blockIdx.x == 2) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_2(blockIdx.y, rC, input1);
}
if (blockIdx.x == 3) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_3(blockIdx.y, rC, input1);
}
if (blockIdx.x == 4) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_4(blockIdx.y, rC, input1);
}
if (blockIdx.x == 5) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_5(blockIdx.y, rC, input1);
}
if (blockIdx.x == 6) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_6(blockIdx.y, rC, input1);
}
if (blockIdx.x == 7) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_7(blockIdx.y, rC, input1);
}
if (blockIdx.x == 8) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_8(blockIdx.y, rC, input1);
}
if (blockIdx.x == 9) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_9(blockIdx.y, rC, input1);
}
if (blockIdx.x == 10) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_10(blockIdx.y, rC, input1);
}
if (blockIdx.x == 11) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_11(blockIdx.y, rC, input1);
}
if (blockIdx.x == 12) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_12(blockIdx.y, rC, input1);
}
if (blockIdx.x == 13) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_13(blockIdx.y, rC, input1);
}
if (blockIdx.x == 14) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_14(blockIdx.y, rC, input1);
}
if (blockIdx.x == 15) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_15(blockIdx.y, rC, input1);
}
if (blockIdx.x == 16) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_16(blockIdx.y, rC, input1);
}
if (blockIdx.x == 17) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_17(blockIdx.y, rC, input1);
}
if (blockIdx.x == 18) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_18(blockIdx.y, rC, input1);
}
if (blockIdx.x == 19) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_19(blockIdx.y, rC, input1);
}
if (blockIdx.x == 20) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_20(blockIdx.y, rC, input1);
}
if (blockIdx.x == 21) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_21(blockIdx.y, rC, input1);
}
if (blockIdx.x == 22) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_22(blockIdx.y, rC, input1);
}
if (blockIdx.x == 23) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_23(blockIdx.y, rC, input1);
}
if (blockIdx.x == 24) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_24(blockIdx.y, rC, input1);
}
if (blockIdx.x == 25) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_25(blockIdx.y, rC, input1);
}
if (blockIdx.x == 26) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_26(blockIdx.y, rC, input1);
}
if (blockIdx.x == 27) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_27(blockIdx.y, rC, input1);
}
if (blockIdx.x == 28) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_28(blockIdx.y, rC, input1);
}
if (blockIdx.x == 29) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_29(blockIdx.y, rC, input1);
}
if (blockIdx.x == 30) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_30(blockIdx.y, rC, input1);
}
if (blockIdx.x == 31) {
sparse_matrix_1024_1024_1024_999_bert_osdi_final_device_func_blockIdx_x_31(blockIdx.y, rC, input1);
}
for (int i = 0; i < 32; i++) {
output0[(blockIdx.x*32+i)*1024 + blockIdx.y*128+threadIdx.x] = rC[i];
}

}
