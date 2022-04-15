source ~/anaconda/etc/profile.d/conda.sh
conda activate artifact
nvcc -gencode arch=compute_75,code=sm_75 -lcublas -o cublas cublas.cu
nvcc -gencode arch=compute_75,code=sm_75 -lcusparse -o cusparse cusparse.cu
nvcc -forward-unknown-to-host-compiler -I/usr/local/cuda/include -I/root/sputnik  -L/usr/local/cuda/lib64  -L/usr/local/lib -lcudart -lspmm  --generate-code=arch=compute_75,code=sm_75 -std=c++14  sputnik.cu -o sputnik

mkdir log
for sparsity in 0.5 0.7 0.8 0.9 0.95 0.99
do
    echo $sparsity
    ./cusparse ${sparsity} > log/cusparse_${sparsity}.log
    ./sputnik ${sparsity} > log/sputnik_${sparsity}.log
    ./cublas ${sparsity} > log/cublas_${sparsity}.log
done


# taco
rm taco_latency.txt
taco-test --gtest_filter="scheduling_eval.spmmDCSRGPU50"
taco-test --gtest_filter="scheduling_eval.spmmDCSRGPU70"
taco-test --gtest_filter="scheduling_eval.spmmDCSRGPU80"
taco-test --gtest_filter="scheduling_eval.spmmDCSRGPU90"
taco-test --gtest_filter="scheduling_eval.spmmDCSRGPU95"
taco-test --gtest_filter="scheduling_eval.spmmDCSRGPU99"
mv taco_latency.txt log/

pushd sparta
python spmm_finegrained_codegen.py --sparsity 50 > ../log/sparta_0.5.log
python spmm_finegrained_codegen.py --sparsity 70 > ../log/sparta_0.7.log
python spmm_finegrained_codegen.py --sparsity 80 > ../log/sparta_0.8.log
python spmm_finegrained_codegen.py --sparsity 90 > ../log/sparta_0.9.log
python spmm_finegrained_codegen.py --sparsity 95 > ../log/sparta_0.95.log
python spmm_finegrained_codegen.py --sparsity 99 > ../log/sparta_0.99.log
popd

python draw.py