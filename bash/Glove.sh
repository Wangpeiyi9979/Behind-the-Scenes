#!/usr/bin/env bash
N=(5)
K=(5 10)
sample_methods=(normal trigger_uniform blurry_uniform)
seeds=(1)
DATASET=maven
# shellcheck disable=SC2068
for n in ${N[@]};do
  for k in ${K[@]};do
      for sample_method in ${sample_methods[@]};do
        for seed in ${seeds[@]}; do
            python3 train.py run \
            --model=Glove \
            --avg=trigger \
            --dataset=${DATASET} \
            --gpu_id=0 \
            --val_step=2000 \
            --lr=1e-2 \
            --N=${n} \
            --K=${k} \
            --test_sample_method=${sample_method} \
            --seed=${seed} \
            --save_opt=Glove_${DATASET}_${n}_${k}_${sample_method}_${seed}  \
            --log_dir=Glove_${DATASET}_4tune.txt \
            --early_stop=3
        done
      done
    done
done
