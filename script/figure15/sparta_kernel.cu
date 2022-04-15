
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>

using namespace std;

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

template <
    const int BLOCK_SIZE_M, // 64
    const int BLOCK_SIZE_K, // 8
    const int BLOCK_SIZE_N, // 128
    const int THREAD_SIZE_M, // 8
    const int THREAD_SIZE_K, // 4
    const int THREAD_SIZE_N  // 8
>
__global__ void BLOCK_SPARSE_MATMUL(float* A, float* W_val, int* W_row, int* W_col, float* C, float *bias, int M, int K, int N){
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    __shared__ float As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    __shared__ float Bs[BLOCK_SIZE_N * BLOCK_SIZE_K];

    float accum[THREAD_SIZE_N][THREAD_SIZE_M] = {0};
    float a_frag[THREAD_SIZE_M][THREAD_SIZE_K];
    float b_frag[THREAD_SIZE_N][THREAD_SIZE_K];

    int A_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    int B_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    int bszy = BLOCK_SIZE_M / THREAD_SIZE_M;
    int bszx = BLOCK_SIZE_N / THREAD_SIZE_N;

    int THREADS_PER_BLOCK = bszy * bszx;

    int A_TILE_ROW_STRIDE = THREADS_PER_BLOCK / A_THREAD_PER_ROW;
    int B_TILE_ROW_STRIDE = THREADS_PER_BLOCK / B_THREAD_PER_ROW;

    int tid = ty * bszx + tx;

    int A_BLOCK_ROW_START = tid / A_THREAD_PER_ROW;
    int B_BLOCK_ROW_START = tid / B_THREAD_PER_ROW;

    int A_BLOCK_COL_START = tid % A_THREAD_PER_ROW * 4;
    int B_BLOCK_COL_START = tid % B_THREAD_PER_ROW * 4;

    int index_start = W_row[bx], index_end = W_row[bx+1];

    const int vBLOCK_SIZE_M = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int vBLOCK_SIZE_N = BLOCK_SIZE_N / THREAD_SIZE_N;
    for(int tile_block_idx = index_start; tile_block_idx < index_end; tile_block_idx += 1){
        int tile_idx = W_col[tile_block_idx] * BLOCK_SIZE_K;
        #pragma unroll
        for(int k = 0; k < BLOCK_SIZE_M; k += A_TILE_ROW_STRIDE){
            *((float4 *)(&As[(k+A_BLOCK_ROW_START) * BLOCK_SIZE_K + A_BLOCK_COL_START])) =
                *((float4 *)(&A[(by*BLOCK_SIZE_M+k+A_BLOCK_ROW_START) * K + tile_idx+A_BLOCK_COL_START]));
            //FETCH_FLOAT4(As[OFFSET(k+A_BLOCK_ROW_START, A_BLOCK_COL_START, BLOCK_SIZE_K)]) =
                //FETCH_FLOAT4(A[OFFSET(by*BLOCK_SIZE_M+k+A_BLOCK_ROW_START, tile_idx+A_BLOCK_COL_START, K)]);
        }

        #pragma unroll
        for(int k = 0; k < BLOCK_SIZE_K; k += B_TILE_ROW_STRIDE){
            *((float4 *)(&Bs[(k+B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START])) =
                *((float4 *)(&W_val[tile_block_idx * BLOCK_SIZE_N * BLOCK_SIZE_K + (k+B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START]));
            //FETCH_FLOAT4(Bs[OFFSET(k+B_BLOCK_ROW_START, B_BLOCK_COL_START, BLOCK_SIZE_N)]) = 
                //FETCH_FLOAT4(W_val[tile_block_idx * BLOCK_SIZE_N * BLOCK_SIZE_K + (k+B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START]);
        }

        __syncthreads();

        #pragma unroll
        for(int k = 0; k < BLOCK_SIZE_K; k += THREAD_SIZE_K){
            #pragma unroll
            for(int i = 0; i < THREAD_SIZE_K; i++){
                #pragma unroll
                for(int j = 0; j < THREAD_SIZE_M; j += 1){
                    a_frag[j][i] = As[(ty + vBLOCK_SIZE_M * j) * BLOCK_SIZE_K + k + i];
                    //a_frag[j][i] = As[OFFSET(ty + vBLOCK_SIZE_M * j, k+i, BLOCK_SIZE_K)];
                }
            }

            #pragma unroll
            for(int i = 0; i < THREAD_SIZE_K; i++){
                #pragma unroll
                for(int j = 0; j < THREAD_SIZE_N; j += 1){
                    b_frag[j][i] = Bs[(k+i) * BLOCK_SIZE_N + tx + vBLOCK_SIZE_N * j];
                    //b_frag[j][i] = Bs[OFFSET(k+i, tx + vBLOCK_SIZE_N * j, BLOCK_SIZE_N)];
                }
            }

            #pragma unroll
            for(int i = 0; i < THREAD_SIZE_N; i++){
                #pragma unroll
                for(int j = 0; j < THREAD_SIZE_M; j++){
                    #pragma unroll
                    for(int k_in = 0; k_in < THREAD_SIZE_K; k_in++){
                        // accum[i][j] = fma(a_frag[j][k_in], b_frag[i][k_in], accum[i][j]);
                        accum[i][j] += a_frag[j][k_in] * b_frag[i][k_in];
                    }
                }
            }
        }

        __syncthreads();
    }

    float bias_local[THREAD_SIZE_N];
    for(int thread_x = 0; thread_x < THREAD_SIZE_N; thread_x++){
        bias_local[thread_x] = bias[BLOCK_SIZE_N * bx + tx + thread_x * vBLOCK_SIZE_N];
    }

    #pragma unroll
    for(int thread_x = 0; thread_x < THREAD_SIZE_N; thread_x++){
        #pragma unroll
        for(int thread_y = 0; thread_y < THREAD_SIZE_M; thread_y+=1){
            C[(BLOCK_SIZE_M * by + ty + thread_y * vBLOCK_SIZE_M) * N + BLOCK_SIZE_N * bx + tx + thread_x * vBLOCK_SIZE_N] = (accum[thread_x][thread_y]) + bias_local[thread_x];
            /*
            C[OFFSET(
                BLOCK_SIZE_M * by + ty + thread_y * vBLOCK_SIZE_M,
                BLOCK_SIZE_N * bx + tx + thread_x * vBLOCK_SIZE_N,
                N
            )] = (accum[thread_x][thread_y]) + bias_local[thread_x];
            */
        }
    }
}


__global__ void matrixMultiplication(float *input0, float *input1, float *output0);

void normal_matmul(vector<float> A, vector<float> B, vector<float> &C_right, int M, int N, int K)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                C_right[m*N+n] += A[m*K+k] * B[k*N+n];
            }
        }
    }
}

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
	cudaError_t e=cudaGetLastError();                                 \
	if(e!=cudaSuccess) {                                              \
	  printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
	  exit(0); \
	}                                                                 \
   }

size_t load_from_file(char* ptr, size_t buff_size, string filepath){
    std::ifstream fin(filepath, ios::in | ios::binary);
    size_t loaded_size = fin.read(ptr, buff_size).gcount();
    return loaded_size;
}

int matrixMultiply(int M, int N, int K){
    int size_A = M * K;
    int size_C = M * N;

    /*
    const int BLOCK_SIZE_M = 32; // 64
    const int BLOCK_SIZE_K = 32;  //8
    const int BLOCK_SIZE_N = 32;  //128
    const int THREAD_SIZE_M = 8;  //8
    const int THREAD_SIZE_K = 4;  //4
    const int THREAD_SIZE_N = 8;  //8
    */

    const int BLOCK_SIZE_M = 64; // 64
    const int BLOCK_SIZE_K = 16;  //8
    const int BLOCK_SIZE_N = 32;  //128
    const int THREAD_SIZE_M = 8;  //8
    const int THREAD_SIZE_K = 1;  //4
    const int THREAD_SIZE_N = 8;  //8

    int mem_size_A = sizeof(float) * size_A;
    int mem_size_C = sizeof(float) * size_C;
    int mem_size_bias = sizeof(float) * N;

    // memory size of row, col, val
    int mem_size_row = sizeof(int) * M;
    int mem_size_col = sizeof(int) * M * N;
    int mem_size_val = sizeof(float) * M * N;

    float* h_A = (float*)malloc(mem_size_A);
    float* h_C = (float*)malloc(mem_size_C);
    float* h_bias = (float*)malloc(mem_size_bias);
    float* h_result = (float*)malloc(mem_size_C);

    // memory allocation of row, col, val
    int* h_row = (int*)malloc(mem_size_row);
    int* h_col = (int*)malloc(mem_size_col);
    float* h_val = (float*)malloc(mem_size_val);

    // load data
    std::string row_path = "bcsr_row.bin";
    std::string col_path = "bcsr_col.bin";
    std::string val_path = "bcsr_val.bin";

    load_from_file((char*)h_row, mem_size_row, row_path);
    load_from_file((char*)h_col, mem_size_col, col_path);
    load_from_file((char*)h_val, mem_size_val, val_path);

    float* d_A;
    float* d_C;
    float* d_bias;

    // device memory allocation
    int* d_row;
    int* d_col;
    float* d_val;

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 10;

    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
            h_A[i * K + j] = rand()%5;
        }
    }

    for(int i = 0; i < N; i++){
        h_bias[i] = rand()%5;
    }

    printf("host init successfully!\n");
    printf("number of iteration: %d\n", nIter);

    checkCudaErrors(cudaMalloc(&d_A, mem_size_A));
    checkCudaErrors(cudaMalloc(&d_C, mem_size_C));
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

    printf("Begin to run MatrixMulCUDA_8bit() function....\n");
    dim3 dimBlock(float(BLOCK_SIZE_N / THREAD_SIZE_N), BLOCK_SIZE_M / THREAD_SIZE_M);
    dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);

    dim3 blocksPerGrid(32, 8);
    dim3 threadsPerBlock(128);

    // warm-up
    for(int run = 0; run < nIter; run++){
        matrixMultiplication<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_A, d_C);
        BLOCK_SPARSE_MATMUL<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_K, THREAD_SIZE_N><<<dimGrid, dimBlock>>>(d_A, d_val, d_row, d_col, d_C, d_bias, M, K, N);
    }

    checkCudaErrors(cudaEventRecord(start));
    for(int run = 0; run < nIter; run++) {
        matrixMultiplication<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_A, d_C);
        BLOCK_SPARSE_MATMUL<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_K, THREAD_SIZE_N><<<dimGrid, dimBlock>>>(d_A, d_val, d_row, d_col, d_C, d_bias, M, K, N);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_result, d_C, mem_size_C, cudaMemcpyDeviceToHost));

    float msecPerMatrixMul = msecTotal / nIter;

    printf("float32 block sparse kernel gemm Time= %f msec\n", msecPerMatrixMul);

    cudaFree(d_A);
    cudaFree(d_C);
    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_val);

    free(h_A);
    free(h_C);
    free(h_row);
    free(h_col);
    free(h_val);

    return EXIT_SUCCESS;
}

int main()
{
    printf("[Matrix Multiply Using CUDA] - Starting...\n");

    // This will pick the best possible CUDA capable device, otherwise
    // override the device ID based on input provided at the command line

    int M = 1024, N = 1024, K = 1024;

    printf("MatrixA(%d, %d), MatrixB(%d, %d)\n", M, K, K, N);

    matrixMultiply(M, N, K);
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