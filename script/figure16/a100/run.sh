source ~/anaconda/etc/profile.d/conda.sh
conda activate artifact

SPUTNIK_ROOT=/root/sputnik
CUSPARSELT_ROOT=/root/libcusparse_lt
nvcc -forward-unknown-to-host-compiler  -I/usr/local/cuda/include -I${SPUTNIK_ROOT} -I${CUSPARSELT_ROOT}/include -I${SPUTNIK_ROOT}/third_party/abseil-cpp -L/usr/local/cuda/lib64  -L${SPUTNIK_ROOT}/build/sputnik -L${CUSPARSELT_ROOT}/lib64  -lcusparse -lcudart -lcusparseLt -lspmm  --generate-code=arch=compute_80,code=sm_80 -std=c++14  sparta.cu -o sparta
nvcc -forward-unknown-to-host-compiler  -I/usr/local/cuda/include -I${SPUTNIK_ROOT} -I${CUSPARSELT_ROOT}/include -I${SPUTNIK_ROOT}/third_party/abseil-cpp -L/usr/local/cuda/lib64  -L${SPUTNIK_ROOT}/build/sputnik -L${CUSPARSELT_ROOT}/lib64  -lcusparse -lcudart -lcusparseLt -lspmm  --generate-code=arch=compute_80,code=sm_80 -std=c++14  sputnik.cu -o sputnik
nvcc openai_blocksparse.cu -o openai_blocksparse
nvcc -lcublas -o cublas cublas.cu
sparsity_ratio=(0.5 0.6 0.7 0.8 0.9)
mkdir -p log
for sparsity in ${sparsity_ratio[@]}
do
    echo $sparsity
    ./cublas $sparsity > log/cublas_${sparsity}.log
    ./openai_blocksparse $sparsity > log/openai_${sparsity}.log
    ./sputnik $sparsity > log/sputnik_${sparsity}.log
    ./sparta $sparsity > log/sparta_${sparsity}.log
done
python draw.py