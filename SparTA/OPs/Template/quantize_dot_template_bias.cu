extern "C" __global__ void MatrixMulCUDA_8bit_bias(float *input0, float *input1, float *input2, float *input3, float *input4, float *input5, float * input6,float  *input7, float *output0) 
{
    const unsigned int M_GLOBAL=M_GLOBAL_VALUE;
    const unsigned int K_GLOBAL=K_GLOBAL_VALUE;
    const unsigned int N_GLOBAL=N_GLOBAL_VALUE;
    // const parameters
    const unsigned int  WARP_SIZE=32;
    const unsigned int  M=16;
    const unsigned int  N=16;
    const unsigned int  K=16;
    const unsigned int  WMMA_M=16;
    const unsigned int  WMMA_N=16;
    const unsigned int  WMMA_K=16;

    const unsigned int  M_TILES=(M_GLOBAL/M);
    const unsigned int  K_TILES=(K_GLOBAL/K);
    const unsigned int  N_TILES=(N_GLOBAL/N);



    // typedef C_LAYOUT wmma::mem_row_major;

    const unsigned int CHUNK_K=CHUNK_K_VALUE;
    const unsigned int BLOCK_ROW_WARPS=BLOCK_ROW_WARPS_VALUE;
    const unsigned int BLOCK_COL_WARPS=BLOCK_COL_WARPS_VALUE;

    const unsigned int WARP_ROW_TILES =WARP_ROW_TILES_VALUE;
    const unsigned int WARP_COL_TILES =WARP_COL_TILES_VALUE;

    const unsigned int WARPS_PER_BLOCK=(BLOCK_ROW_WARPS * BLOCK_COL_WARPS);
    const unsigned int THREADS_PER_BLOCK= (WARP_SIZE * WARPS_PER_BLOCK);
    const unsigned int CHUNK_LINE_BYTES=(CHUNK_K * K * sizeof(uint8_t));
    const unsigned int WARP_COPY_BYTES=(WARP_SIZE * sizeof(int4));
    const unsigned int CHUNK_COPY_LINES_PER_WARP=(WARP_COPY_BYTES / CHUNK_LINE_BYTES);
    const unsigned int CHUNK_COPY_LINE_LANES=(WARP_SIZE / CHUNK_COPY_LINES_PER_WARP);

    const unsigned int LANE_ROW_STRIDE = (WARP_ROW_TILES * N / 8);
    const unsigned int LANE_COL_STRIDE = (WARP_COL_TILES * M / 4);
    const unsigned int WARP_STRIDE = (WARP_ROW_TILES * N);

    const unsigned int BLOCK_ROW_TILES =(WARP_ROW_TILES * BLOCK_ROW_WARPS);
    const unsigned int BLOCK_COL_TILES =(WARP_COL_TILES * BLOCK_COL_WARPS);

    const unsigned int GLOBAL_MEM_STRIDE =N_GLOBAL;

    const unsigned int SHMEM_STRIDE=(N * BLOCK_ROW_TILES);
    const unsigned int SHMEM_OFFSET=(N * WARP_ROW_TILES);

    const unsigned int SKEW_UINT8=32;


    // Convert the input pointers
    const uint8_t * A = reinterpret_cast<uint8_t*>(input0); // activation
    const uint8_t * B =  reinterpret_cast<uint8_t*>(input1); // weight
    const int * C = reinterpret_cast< int *>(input7);
    uint8_t * D = reinterpret_cast<uint8_t*>(output0);
    const int alpha = (int)(*input5);
    const int integer = (int)(*input6);


  const int shared_size = MAX(sizeof(uint8_t) * (BLOCK_COL_TILES * M + BLOCK_ROW_TILES * N) *
                       (CHUNK_K * K + SKEW_UINT8),
                   M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N *
                       (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(int));
  __shared__ uint8_t shmem[shared_size/(CHUNK_K * K + SKEW_UINT8)+1][CHUNK_K * K + SKEW_UINT8];

  // Warp and lane identification.
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;

  // Offset in shared memory from which the B matrix is stored.
  const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;       // BLOCK_COL_TILES is shared_A row numbers in one block

  // This pointer is used to access the C and D matrix tiles this warp computes.
  int *shmem_warp_tile_ptr = (int *)&shmem[0][0] +
                             (warpId / BLOCK_ROW_WARPS) * SHMEM_STRIDE * M * WARP_COL_TILES +    // original K * 2 is because one warp calculate k * 2 rows.
                             (warpId % BLOCK_ROW_WARPS) * SHMEM_OFFSET;


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

    // Stream multiple C tiles to shared memory.

    __syncthreads();

    // These fragments will accumulate the result of A and B matrix fragment
    // multiplications along the K_GLOBAL dimension.
    wmma::fragment<wmma::accumulator, M, N, K, int> c[WARP_COL_TILES]
                                                     [WARP_ROW_TILES];

#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        // Uniform, point-wise transformations of ALL fragment elements by ALL
        // threads in the warp are well-defined even though element indices
        // within fragment storage are not defined.
        for (int t = 0; t < c[i][j].num_elements; t++) {
          c[i][j].x[t] = 0;
        }
      }
    }

    __syncthreads();

    // Select what warp copies what matrix to shared memory.
    // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.

    // need to modify, per warp correspond to block of matrix A and B
    // especially need to change the value 2

    // threads number of loading per row to shared memory
    const unsigned int THREADS_PER_ROW = (CHUNK_K * K * sizeof(uint8_t)) / (sizeof(int));

    // number of lines loading data to shared memory per cycle.
    const unsigned int TILE_LINE_STRIDE = THREADS_PER_BLOCK / THREADS_PER_ROW;

    const uint8_t *warp_ptr = (warpId < (WARPS_PER_BLOCK / 2)) ? (&A[block_tile_i * M * K_GLOBAL] +
                                              M * K_GLOBAL * (warpId % (WARPS_PER_BLOCK / 2)) * (BLOCK_COL_TILES / (WARPS_PER_BLOCK / 2)))
                                           : (&B[block_tile_j * N * K_GLOBAL] +
                                              N * K_GLOBAL * (warpId % (WARPS_PER_BLOCK / 2)) * (BLOCK_ROW_TILES / (WARPS_PER_BLOCK / 2)));

    // Go through the global K dimension by a fixed step at a time.
#pragma unroll
    for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
      size_t shmem_idx =
          warpId < (WARPS_PER_BLOCK / 2)
              ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * (BLOCK_COL_TILES / (WARPS_PER_BLOCK / 2)))
              : (N * (warpId % (WARPS_PER_BLOCK / 2)) * (BLOCK_ROW_TILES / (WARPS_PER_BLOCK / 2)) + shmem_idx_b_off);

      // First half of the warp copies the first row / column of the matrix,
      // the second half of the warp copies the next.
      int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * K +
                                (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) +
                       (laneId % CHUNK_COPY_LINE_LANES);

      // Shift the second half of the warp to the next row / column in the
      // shared memory.
      shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

      int iter_index = warpId < (WARPS_PER_BLOCK / 2) 
        ? (BLOCK_COL_TILES / (WARPS_PER_BLOCK / 2)) * M / CHUNK_COPY_LINES_PER_WARP 
        : (BLOCK_ROW_TILES / (WARPS_PER_BLOCK / 2)) * N / CHUNK_COPY_LINES_PER_WARP;
#pragma unroll
      //for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2;
      for (int i = 0; i < iter_index;
           i++) {
        // Copy 16 bytes at once in each lane.
        *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
            *lane_ptr;

        // Advance the global memory pointer and the shared memory index.
        lane_ptr = (int4 *)((uint8_t *)lane_ptr +
                            K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
        shmem_idx += CHUNK_COPY_LINES_PER_WARP;
      }

      __syncthreads();

      /*
    #pragma unroll
      // load matrix A from global memory to shared memory.
      for(int i = 0; i < BLOCK_COL_TILES * M; i += TILE_LINE_STRIDE){
        *((int*)&shmem[i + threadIdx.x / THREADS_PER_ROW][0] + (threadIdx.x % THREADS_PER_ROW)) = 
          *((int*)&A[block_tile_i * M * K_GLOBAL + tile_k * K + (i + threadIdx.x) / THREADS_PER_ROW * K_GLOBAL] + (threadIdx.x % THREADS_PER_ROW));
      }
    
    #pragma unroll
      for(int i = 0; i < BLOCK_ROW_TILES * N; i += TILE_LINE_STRIDE){
        *((int*)&shmem[shmem_idx_b_off + i + threadIdx.x / THREADS_PER_ROW][0] + (threadIdx.x % THREADS_PER_ROW)) = 
          *((int*)&B[block_tile_j * N * K_GLOBAL + tile_k * K + (i + threadIdx.x) / THREADS_PER_ROW * K_GLOBAL] + (threadIdx.x % THREADS_PER_ROW));
      }

      __syncthreads();
      */

      // Compute a grid of C matrix tiles in each warp.
#pragma unroll
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

        int *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

        wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, wmma::mem_row_major);
      }
    }

    __syncthreads();

    // Now that shared memory contains all the D tiles, stream them to global
    // memory.
    // This pointer is used to stream the C and D matrices block-wide tile to and
    // from shared memory.

    // ( (warpId / BLOCK_ROW_WARPS) * WARP_COL_TILES * M, (warpId % BLOCK_ROW_WARPS) * WARP_ROW_TILES * N )
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
          (uint8_t)*((int *)(shmem_lane_stream_ptr + SHMEM_STRIDE * i + k));
      }
    }

    
    /*
    int *shmem_warp_stream_ptr = (int *)&shmem[0][0] + warpId * SHMEM_STRIDE * K;   // confuse, may be used to read from global memory?
    const size_t gmem_idx =
        (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
    uint8_t *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
    for (int i = 0; i < K; i++) {
      for(int k = 0; k < 4; k++){
        *(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i + laneId * 4 + k) =
          (uint8_t)*((int *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i + laneId * 4 + k));
      }
    }
    */
    
    __syncthreads();
}
