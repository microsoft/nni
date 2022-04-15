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

#define M_GLOBAL M_VALUE
#define N_GLOBAL N_VALUE
#define K_GLOBAL K_VALUE

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

#define CHUNK_K CHUNK_K_VALUE

#define CHUNK_LINE_BYTES (CHUNK_K * K * sizeof(uint8_t))                // 128
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))                      // 512
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)  // 4
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)   // 8


#define BLOCK_ROW_WARPS BLOCK_ROW_WARPS_VALUE
#define BLOCK_COL_WARPS BLOCK_COL_WARPS_VALUE

#define WARP_ROW_TILES WARP_ROW_TILES_VALUE
#define WARP_COL_TILES WARP_COL_TILES_VALUE


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


__global__ void compute_gemm_imma(const uint8_t *A, const uint8_t *B_val, int *B_row, int *B_col,
                                  uint8_t *D, int *bias, int alpha, int integer) {
  const unsigned int OUTPUT_LINE_BYTES = (BLOCK_ROW_TILES * N);
  const unsigned int OUTPUT_LANE_PER_LINE = OUTPUT_LINE_BYTES / 4;
  const unsigned int OUTPUT_LINES_PER_BLOCK = THREADS_PER_BLOCK / OUTPUT_LANE_PER_LINE;
  const unsigned int OUTPUT_ITERS = (BLOCK_COL_WARPS * WARP_COL_TILES * M) / OUTPUT_LINES_PER_BLOCK;

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


    /////////////////// bias ///////////////////
    int *shmem_load_output = ((int *)&shmem[0][0] + threadIdx.x / OUTPUT_LANE_PER_LINE * (BLOCK_ROW_TILES * N));
    const size_t gmem_load_output = block_tile_j * N;
    const int *src_gmem_output = (int *)(&bias[gmem_load_output]);

    // Stream multiple C tiles to shared memory.
#pragma unroll
    for (int i = 0; i < OUTPUT_ITERS; i++) {
      *((int4 *)(shmem_load_output) + threadIdx.x % OUTPUT_LANE_PER_LINE) =
        *((int4 *)(src_gmem_output) + threadIdx.x % OUTPUT_LANE_PER_LINE);
      shmem_load_output += OUTPUT_LINES_PER_BLOCK * (BLOCK_ROW_TILES * N);   
    }

    __syncthreads();
    
    // This pointer is used to access the C and D matrix tiles this warp computes.
    int *shmem_warp_tile_ptr = (int *)&shmem[0][0] +
    (warpId / BLOCK_ROW_WARPS) * M * WARP_COL_TILES * SHMEM_STRIDE  +    // K * 2 is because one warp calculate k * 2 rows.
    (warpId % BLOCK_ROW_WARPS) * SHMEM_OFFSET;
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
        const int *tile_ptr =
            shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

        wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
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

void HostComputation_sparse(uint8_t* A, int* row, int* col, uint8_t* val, uint8_t* D, int *bias, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int alpha, int integer){
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
          int cSub = bias[j];
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
  int mem_size_bias = sizeof(int) * N_GLOBAL;

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
  std::string row_path = ROW_PATH_VALUE;
  std::string col_path = COL_PATH_VALUE;
  std::string val_path = VAL_PATH_VALUE;

  int alpha = 1;
  int integer = 0;

  uint8_t* d_A;
  uint8_t* d_D;
  int* d_row;
  int* d_col;
  uint8_t* d_val;
  int *d_bias;

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

  for(int i = 0; i < N_GLOBAL; i++){
    h_bias[i] = rand()%5;
  }

  load_from_file((char*)h_row, mem_size_row, row_path);
  load_from_file((char*)h_col, mem_size_col, col_path);
  load_from_file((char*)h_val, mem_size_val, val_path);

  for(int i = 0; i < M_GLOBAL * N_GLOBAL; i += 1){
    h_val[i] = rand()%5;
  }

  for(int i = 0; i < M_GLOBAL; i++){
      for(int j = 0; j < N_GLOBAL; j++){
          h_C[i * N_GLOBAL + j] = 0;
      }
  }

  printf("host init successfully!\n");

  checkCudaErrors(cudaMalloc(&d_A, mem_size_A));
  checkCudaErrors(cudaMalloc(&d_D, mem_size_D));
  checkCudaErrors(cudaMalloc(&d_bias, mem_size_bias));

  checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_bias, h_bias, mem_size_bias, cudaMemcpyHostToDevice));

  // device csr memory copy
  checkCudaErrors(cudaMalloc(&d_row, mem_size_row));
  checkCudaErrors(cudaMalloc(&d_col, mem_size_col));
  checkCudaErrors(cudaMalloc(&d_val, mem_size_val));

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

  printf("Begin to run MatrixMulCUDA_8bit() function....\n");
  printf("Computing... using high performance kernel compute_gemm_imma \n");

  for(int run = 0; run < nIter; run++){
    checkKernelErrors(
        (compute_gemm_imma<<<block_num, THREADS_PER_BLOCK>>>(d_A, d_val, d_row, d_col, d_D, d_bias, alpha, integer)));
  }

  checkCudaErrors(cudaEventRecord(start));
  // If enough shared memory available on the GPU use high performant kernel

  for(int run = 0; run < nIter; run++){
    checkKernelErrors(
        (compute_gemm_imma<<<block_num, THREADS_PER_BLOCK>>>(d_A, d_val, d_row, d_col, d_D, d_bias, alpha, integer)));
  }
  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  float msecPerMatrixMul = msecTotal / nIter;

  printf("8bit quantize sparse kernel gemm Time= %f msec\n", msecPerMatrixMul);

#if CPU_DEBUG
  printf("Verify the correctness on host\n");

  int BLOCK_SIZE_M = BLOCK_COL_TILES * M;
  int BLOCK_SIZE_N = BLOCK_ROW_TILES * N;
  int BLOCK_SIZE_K = CHUNK_K * K;

  HostComputation_sparse(h_A, h_row, h_col, h_val, h_C, h_bias, BLOCK_SIZE_N, BLOCK_SIZE_K, alpha, integer);

  // HostComputation(h_A, h_B, h_C, h_D, alpha, integer);

  checkCudaErrors(cudaMemcpy( h_result, d_D, mem_size_D, cudaMemcpyDeviceToHost));

  bool correct = true;
  double eps = 1.e-6;

  for(int i = 0; i < M_GLOBAL * N_GLOBAL; i++){
      double abs_err = fabs(h_C[i] - h_result[i]);
      double dot_length = M;
      double abs_val = fabs(h_C[i]);
      double rel_err = abs_err / abs_val / dot_length;
      if (rel_err > eps) {
          printf("Error! Matrix[%05d]=%d, ref=%d error term is > %E\n",
                  i, h_result[i], h_C[i], eps);
          correct = false;
          break;
      }
  }

  if(correct) printf("Result = Pass\n");
  else printf("Result = Fail\n");

  printf("val in pos 206: %d, %d\n", h_result[206], h_D[206]);

#endif

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
