pushd ../../checkpoints/hubert
python hubert_jit_time.py \
        --model_name_or_path superb/hubert-base-superb-ks \
        --dataset_name superb \
        --dataset_config_name ks \
        --output_dir sdas \
        --overwrite_output_dir \
        --remove_unused_columns False \
        --do_eval --eval_split_name test --learning_rate 1e-3 \
        --max_length_seconds 1 \
        --seed 0 --iterations 300
popd
