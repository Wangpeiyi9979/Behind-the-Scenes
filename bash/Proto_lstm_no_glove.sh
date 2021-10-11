#!/usr/bin/env bash
N=(5 10)
K=(5 10)
sample_methods=(normal trigger_uniform blurry_uniform)
seeds=(1 2 3)
DATASET=maven

# shellcheck disable=SC2068
for n in ${N[@]};do
  for k in ${K[@]};do
      for sample_method in ${sample_methods[@]};do
        for seed in ${seeds[@]}; do
            python3 train.py run \
            --model=Proto \
            --encoder=lstm \
            --use_glove=False \
            --avg=trigger \
            --dataset=${DATASET} \
            --gpu_id=3 \
            --N=${n} \
            --K=${k} \
            --test_sample_method=${sample_method} \
            --seed=${seed} \
            --save_opt=Proto_lstm_noGlove_${DATASET}_${n}_${k}_${sample_method}_${seed}  \
            --log_dir=Proto_${DATASET}_lstm_noGlove.txt
        done
      done
    done
done