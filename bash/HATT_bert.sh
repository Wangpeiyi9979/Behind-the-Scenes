#!/usr/bin/env bash
N=(5 10)
K=(5 10)
sample_methods=(normal trigger_uniform blurry_uniform)
seeds=(1 2 3)
DATASET=wsdm
# shellcheck disable=SC2068
for n in ${N[@]};do
  for k in ${K[@]};do
      for sample_method in ${sample_methods[@]};do
        for seed in ${seeds[@]}; do
            python3 train.py run \
            --model=ProtoHATT \
            --encoder=bert \
            --avg=trigger \
            --dataset=${DATASET} \
            --gpu_id=0 \
            --N=${n} \
            --K=${k} \
            --test_sample_method=${sample_method} \
            --seed=${seed} \
            --save_opt=ProtoHATT_bert_${DATASET}_${n}_${k}_${sample_method}_${seed}  \
            --log_dir=ProtoHATT_${DATASET}_bert.txt
        done
      done
    done
done

