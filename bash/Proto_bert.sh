#!/usr/bin/env bash
N=(5 10)
K=(5 10)
sample_methods=(normal trigger_uniform blurry_uniform)
seeds=(4 5 6 7 8 9 10 11 12)
DATASET=wsdm
# shellcheck disable=SC2068
for n in ${N[@]};do
  for k in ${K[@]};do
      for sample_method in ${sample_methods[@]};do
        for seed in ${seeds[@]}; do
            python3 train.py run \
            --model=Proto \
            --encoder=bert \
            --avg=trigger \
            --dataset=${DATASET} \
            --gpu_id=2 \
            --N_train=5 \
            --N=${n} \
            --K=${k} \
            --test_sample_method=${sample_method} \
            --seed=${seed} \
            --save_opt=Proto_bert_${DATASET}_${n}_${k}_${sample_method}_${seed}  \
            --log_dir=Proto_${DATASET}_bert4tune.txt
        done
      done
    done
done
