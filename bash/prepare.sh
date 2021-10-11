#!/bin/bash

# 先把doc-level数据放在了dataset/maven_origin_data
# 还有Glove文件也放在了tool_data

echo 'maven: doc -> sent'
cd dataset/maven_sent_level
export PYTHONPATH=`pwd`
python docu2sent.py

echo 'filter those classes that contain less than 100 sentences'
python filter_class.py

cd ../..
export PYTHONPATH=`pwd`
echo 'generate word2id & id2word & word_embedding.npy in tool_data'
python datamodels/FewECDataLoader.py process_pretrain_word_emb --vocab_file tool_data/vocab.txt --pretrain_file tool_data/glove.6B.300d.txt --output_dir tool_data

echo 'generate all data in maven_processed_data'
python datamodels/FewECDataLoader.py process_data

echo 'fewevent: fewevent -> sent'
cd dataset/fewevent_sent_level
python fewevent2sent.py

echo 'split fewevent data the same as pakdd paper'
python filter_class.py

cd ../..
export PYTHONPATH=`pwd`
echo 'generate all data in fewevent_processed_data'
python datamodels/FewECDataLoader.py process_data --dataset=fewevent
