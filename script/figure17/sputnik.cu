#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cstdio>             // printf
#include <cstdlib>            // std::rand
#include "sputnik/cuda_utils.h"
#include "sputnik/matrix_utils.h"
#include "sputnik/spmm/cuda_spmm.h"
#include "time.h"
#include <vector>
using namespace std;
using namespace sputnik;
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

constexpr int EXIT_UNSUPPORTED = 2;
// for finegrained kernels
int32_t * row_idx, *col_idx, *d_row_idx, *d_col_idx, *row_swizzle, *d_row_swizzle;
int32_t row_idx_size, col_idx_size, values_size;
float * values, *d_values;
constexpr int m     = 1024; // bigger sizes may require dynamic allocations
constexpr int n     = 1024; // bigger sizes may require dynamic allocations
constexpr int k     = 1024; // bigger sizes may require dynamic allocations
int A_size = m*k*sizeof(float);
int B_size = k*n*sizeof(float);
int C_size = m*n*sizeof(float);
float hA[m * k];
float hB[k * n];
float hC[m * n];
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
void SortedRowSwizzle(int rows, int *row_offsets, int *row_indices) {
  // Create our unsorted row indices.
  std::vector<int> swizzle_staging(rows);
  std::iota(swizzle_staging.begin(), swizzle_staging.end(), 0);

  // Argsort the row indices based on their length.
  std::sort(swizzle_staging.begin(), swizzle_staging.end(),
            [&row_offsets](int idx_a, int idx_b) {
              int length_a = row_offsets[idx_a + 1] - row_offsets[idx_a];
              int length_b = row_offsets[idx_b + 1] - row_offsets[idx_b];
              return length_a > length_b;
            });

  // Copy the ordered row indices to the output.
  std::memcpy(row_indices, swizzle_staging.data(), sizeof(int) * rows);
}

int convert_csr(float * ptr, int32_t row, int32_t col, int32_t * &row_idx, int32_t * &col_idx, float * &values)
{
    auto v_row_idx = std::make_shared<vector<int32_t>>();
    auto v_col_idx = std::make_shared<vector<int32_t>>();
    auto v_values = std::make_shared<vector<float>>();

    for (int i = 0; i < row; i++)
    {
        v_row_idx->push_back(v_values->size());
        for (int j = 0; j < col; j++)
        {
            size_t pos = i * col + j;
            if (ptr[pos] < 1e-8)
            {
                // sparsity
                continue;
            }
            else
            {
                v_values->push_back(ptr[pos]);
                v_col_idx->push_back(j);
            }
        }
    }
    v_row_idx->push_back(v_values->size());
    row_idx_size = sizeof(int32_t)*v_row_idx->size();
    col_idx_size = sizeof(int32_t)*v_col_idx->size();
    values_size = sizeof(float)*v_values->size();
    printf("values_size: %d\n", values_size);
    row_idx = (int32_t*) malloc(row_idx_size);
    col_idx = (int32_t*) malloc(col_idx_size);
    values = (float*) malloc(values_size);
    memcpy(row_idx, v_row_idx->data(), row_idx_size);
    memcpy(col_idx, v_col_idx->data(), col_idx_size);
    memcpy(values, v_values->data(), values_size);
    return v_values->size();
}

void transform(float* A, float*A1, float*A2, int length){
    // split the matrix A into A1 and A2
    // A1 is for the saprse tensor core, A2 is for the finegrained sparse kernel
    memset(A1, 0, sizeof(float)*length);
    memset(A2, 0, sizeof(float)*length);
    assert(length%2==0);
    int nnz=0;
    for(int i=0; i<length/2;i++){
        int start = i*2;
        int end= start+2;
        nnz=0;
        for(int j=start; j<end; j++){
            if(A[j]!=0){
                if(nnz<1){
                    A1[j]=A[j];
                }else{
                    A2[j]=A[j];
                }
                nnz++;
            }
        }
    }
}

int main(int argc, char*argv[]) {
    float sparsity_ratio = atof(argv[1]);
    printf("Sparsity Ratio=%f\n", sparsity_ratio);
    int major_cc, minor_cc;
    // Host problem definition, row-major order


    init(hA, m*k, sparsity_ratio);
    init(hB, k*n, 0);
    
    // build the index for the finegrained kernel
    convert_csr(hA, m,k, row_idx, col_idx, values);
    CHECK_CUDA(cudaMalloc(&d_row_idx, row_idx_size));
    CHECK_CUDA(cudaMalloc(&d_col_idx, col_idx_size));
    CHECK_CUDA(cudaMalloc(&d_values, values_size));
    CHECK_CUDA(cudaMemcpy(d_row_idx, row_idx, row_idx_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_idx, col_idx, col_idx_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, values, values_size, cudaMemcpyHostToDevice));
    row_swizzle = (int *) malloc(sizeof(int) * m);
    CHECK_CUDA(cudaMalloc(&d_row_swizzle, sizeof(int)*m));
    SortedRowSwizzle(m, row_idx, row_swizzle);
    CHECK_CUDA(cudaMemcpy(d_row_swizzle, row_swizzle, sizeof(int)*m, cudaMemcpyHostToDevice));
    int fine_nnz = values_size / sizeof(float);
    printf("fine_nnz: %d\n", fine_nnz);
    float alpha = 1.0f;
    float beta  = 1.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    float *dA, *dB, *dC, *dD;
    CHECK_CUDA( cudaMalloc((void**) &dA, A_size) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size) )
    
    CHECK_CUDA( cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice) )
    
    //--------------------------------------------------------------------------
    
    float ms_total;
    int n_iter = 1000;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i=0;i<n_iter;i++){
        CHECK_CUDA(CudaSpmm(m ,k, n, fine_nnz, d_row_swizzle, d_values, d_row_idx, d_col_idx, dB, dC, 0));

    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_total, start, stop);
    // printf("Timecost: %f ms\n",ms_total/n_iter);
    printf("Time= %f ms\n", ms_total/n_iter);
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //CHECK_CUDA( cudaMemcpy(hA, dA, A_size, cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size, cudaMemcpyDeviceToHost) )

  
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    return 0;
}
