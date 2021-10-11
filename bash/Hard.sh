#!/usr/bin/env bash
N=(5 10)
K=(5 10)
seeds=(1 2 3)
sample_methods=(normal trigger_uniform blurry_uniform)
DATASET=maven
for n in ${N[@]};do
  for k in ${K[@]};do
      for sample_method in ${sample_methods[@]};do
        for seed in ${seeds[@]}; do
          python3 train.py run \
          --model=Hard  \
          --dataset=${DATASET} \
          --gpu_id=3 \
          --N=${n} \
          --K=${k} \
          --test_sample_method=${sample_method} \
          --save_opt=Hard_${DATASET}_${n}_${k}_${sample_method}_${seed} \
          --seed=${seed} \
          --early_stop=1 \
          --log_dir=${DATASET}_Hard.txt
          done
      done
  done
done

#- model: Proto/Glove
#- encoder: cnn/lstm/transformer/bert
#- dataset: maven/ace
#- avg: max/trigger/head_marker
#- sample_method: normal/trigger_uniform

