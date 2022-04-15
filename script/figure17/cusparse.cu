#include "cusparse.h"
#include "iostream"
#include "sstream"
#include "cuda.h"
#include "time.h"
#include "memory"
#include "cublas_v2.h"
#include "vector"
using namespace std;

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

#define CUBLAS_SAFE_CALL(func)                                                                     \
do                                                                                             \
{                                                                                              \
    cublasStatus_t e = (func);                                                                 \
    if (e != CUBLAS_STATUS_SUCCESS)                                                            \
    {                                                                                          \
        std::stringstream safe_call_ss;                                                        \
        safe_call_ss << "\nerror: " #func " failed with error"                                 \
                        << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << e;    \
        throw std::runtime_error(safe_call_ss.str());                                          \
    }                                                                                          \
} while (0)
#define CUSPARSE_SAFE_CALL(func)                                                                     \
do                                                                                             \
{                                                                                              \
    cusparseStatus_t e = (func);                                                                 \
    if (e != CUSPARSE_STATUS_SUCCESS)                                                            \
    {                                                                                          \
        std::stringstream safe_call_ss;                                                        \
        safe_call_ss << "\nerror: " #func " failed with error"                                 \
                        << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << e;    \
        throw std::runtime_error(safe_call_ss.str());                                          \
    }                                                                                          \
} while (0)

#define CUDA_SAFE_CALL(x)                                                                          \
do                                                                                             \
{                                                                                              \
    cudaError_t result = (x);                                                                  \
    if (result != cudaSuccess)                                                                 \
    {                                                                                          \
        const char* msg = cudaGetErrorString(result);                                          \
        std::stringstream safe_call_ss;                                                        \
        safe_call_ss << "\nerror: " #x " failed with error"                                    \
                        << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  \
        throw std::runtime_error(safe_call_ss.str());                                          \
    }                                                                                          \
} while (0)


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
    int row_idx_size = sizeof(int32_t)*v_row_idx->size();
    int col_idx_size = sizeof(int32_t)*v_col_idx->size();
    int values_size = sizeof(float)*v_values->size();
    row_idx = (int32_t*) malloc(row_idx_size);
    col_idx = (int32_t*) malloc(col_idx_size);
    values = (float*) malloc(values_size);
    memcpy(row_idx, v_row_idx->data(), row_idx_size);
    memcpy(col_idx, v_col_idx->data(), col_idx_size);
    memcpy(values, v_values->data(), values_size);
    return v_values->size();
}
void init(float * ptr, size_t length, float sparsity)
{
    for (int i = 0; i < length; i++)
    {
        float pro = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
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
void calculate_reference(int m, int k, int n, float * A, float *B, float * C) 
{
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            float sum = 0.0;
            for(int tmp=0; tmp<k; tmp++){
                sum += A[i * k + tmp] * B[tmp * n + j];
            }
            C[i*n+j] = sum;
        }
    }
}
int main(int argc, char *argv[]){
    float sparsity_ratio = atof(argv[1]);
    printf("Sparsity Ratio=%f\n", sparsity_ratio);
    // Calculate the matA(Activation: Shape=mxk) * matB(Weight:Shape=k*n)
    // Specify the random seed here
    srand(1);
    int32_t * row_idx, *col_idx, *d_row_idx, *d_col_idx;
    int nnz;
    float * values, *d_values;
    float * matA, *matB, *matC, *matC_ref,*d_matA, *d_matB, *d_matC, *dBuffer;
    int m=1024, k=1024, n=1024;
    float alpha=1.0, beta=0.0;
    float sparsity = sparsity_ratio;

    matA = (float*) malloc(sizeof(float)*m*k);
    matB = (float*) malloc(sizeof(float)*k*n);
    matC = (float*) malloc(sizeof(float)*m*n);
    matC_ref = (float*) malloc(sizeof(float)*m*n);
    init(matA, m*k, 0);
    init(matB, k*n, sparsity);
    calculate_reference(m , k , n , matA, matB, matC_ref);
    nnz = convert_csr(matB, k, n, row_idx, col_idx, values);
    int values_size = nnz * sizeof(float);
    int col_idx_size = nnz * sizeof(int);
    int row_idx_size = (k+1) * sizeof(int);
    cusparseHandle_t cusparse_handle;
    CUSPARSE_SAFE_CALL(cusparseCreate(&cusparse_handle));
    CUDA_SAFE_CALL(cudaMalloc(&d_matA, sizeof(float)*m*k));
    CUDA_SAFE_CALL(cudaMalloc(&d_matB, sizeof(float)*n*k));
    CUDA_SAFE_CALL(cudaMalloc(&d_row_idx, row_idx_size));
    CUDA_SAFE_CALL(cudaMalloc(&d_col_idx, col_idx_size));
    CUDA_SAFE_CALL(cudaMalloc(&d_values, values_size));
    CUDA_SAFE_CALL(cudaMalloc(&d_matC, sizeof(float)*m*n));
    CUDA_SAFE_CALL(cudaMemcpy(d_matA, matA, sizeof(float)*m*k, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_matB, matB, sizeof(float)*n*k, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_row_idx, row_idx, row_idx_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_col_idx, col_idx, col_idx_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_values, values, values_size, cudaMemcpyHostToDevice));

    cusparseSpMatDescr_t sp_weight;
    cusparseDnMatDescr_t in_activation, output_m;
    size_t bufferSize;
    CUSPARSE_SAFE_CALL(cusparseCreateCsr(&sp_weight,
        k,
        n,
        nnz,
        (void*) d_row_idx,
        (void*) d_col_idx,
        (void*) d_values,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F));
    CUSPARSE_SAFE_CALL( cusparseCreateDnMat(&in_activation, k, m, k, d_matA,
            CUDA_R_32F, CUSPARSE_ORDER_COL));
    // Create dense matrix C
    CUSPARSE_SAFE_CALL( cusparseCreateDnMat(&output_m, n, m, n, d_matC,
            CUDA_R_32F, CUSPARSE_ORDER_COL));
    // allocate an external buffer if needed
    CUSPARSE_SAFE_CALL(cusparseSpMM_bufferSize(
        cusparse_handle,
        CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, sp_weight, in_activation, &beta, output_m, CUDA_R_32F,
        CUSPARSE_SPMM_CSR_ALG2, &bufferSize));
    CUDA_SAFE_CALL(cudaMalloc(&dBuffer, bufferSize));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 30;
    
    for(int i = 0; i < nIter; i += 1){
        CUSPARSE_SAFE_CALL( cusparseSpMM(cusparse_handle,
            CUSPARSE_OPERATION_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, sp_weight, in_activation, &beta, output_m, CUDA_R_32F,
            CUSPARSE_SPMM_CSR_ALG2, dBuffer));
    }

    checkCudaErrors(cudaEventRecord(start));

    for(int i = 0; i < nIter; i += 1){
        CUSPARSE_SAFE_CALL( cusparseSpMM(cusparse_handle,
            CUSPARSE_OPERATION_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, sp_weight, in_activation, &beta, output_m, CUDA_R_32F,
            CUSPARSE_SPMM_CSR_ALG2, dBuffer));
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    float msecPerMatrixMul = msecTotal / nIter;
    printf("Time= %f msec\n", msecPerMatrixMul);



    CUDA_SAFE_CALL(cudaMemcpy(matC, d_matC, sizeof(float)*m*n, cudaMemcpyDeviceToHost));
    float s1, s2=0;
    for(int i=0;i<m*n;i++){
        s1+=matC_ref[i];
        s2+=matC[i];
        // printf("dense: %f cusparse: %f\n", matC_ref[i], matC[i]);
    }
    printf("dense sum: %f, cusparse sum: %f\n", s1, s2);
    return 0;
}