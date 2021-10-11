#!/usr/bin/env bash
N=(5 10)
K=(5 10)
sample_methods=(normal)
seeds=(1 2 3)
DATASET=wsdm
# shellcheck disable=SC2068
for n in ${N[@]};do
  for k in ${K[@]};do
      for sample_method in ${sample_methods[@]};do
        for seed in ${seeds[@]}; do
            python3 train.py run \
            --model=Proto \
            --encoder=cnn \
            --use_glove=False \
            --avg=max \
            --dataset=${DATASET} \
            --gpu_id=3 \
            --N=${n} \
            --K=${k} \
            --test_sample_method=${sample_method} \
            --seed=${seed} \
            --save_opt=Proto_cnn_noGlove_${DATASET}_${n}_${k}_${sample_method}_${seed}  \
            --log_dir=Proto_${DATASET}_cnn_noGlove.txt
        done
      done
    done
done

