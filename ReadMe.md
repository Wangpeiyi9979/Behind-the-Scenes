# Dataset
- MAVEN
    - [download](https://drive.google.com/drive/folders/19Q0lqJE6A98OLnRqQVhbX3e6rG4BVGn8)
    - put the files `train.jsonl, val.jsonl, test.jsonl` under the file `dataset/maven_origin_data`.
- FewEvent
    - already in `dataset/fewevent_origin_data`
- wordvector
    - [download](https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/gluon/embeddings/glove/glove.6B.zip) `glove.6B.300d.txt`, and put it under the filt `tool_data`

# Environment
```
pip install -r requirements.txt
```
# process data
```
bash bash/prepare.sh
```

# run
```bash
bash bash/Glove.sh gpu_id
```
you will get the results:
- IUS(normal): 5-way-5-shot test   Test accuracy: 84.97
- TUS(trigger_unifirom): 5-way-5-shot test   Test accuracy: 60.74
- COS(blurry_uniform): 5-way-5-shot test   Test accuracy: 45.91

# Citation
If you use our code and data in your work, please cite our paper:
```
@article{wang2021behind,
  title={Behind the Scenes: An Exploration of Trigger Biases Problem in Few-Shot Event Classification},
  author={Wang, Peiyi and Xu, Runxin and Liu, Tianyu and Dai, Damai and Chang, Baobao and Sui, Zhifang},
  journal={arXiv preprint arXiv:2108.12844},
  year={2021}
}
```

# Acknowledgements
Thanks 
```
@inproceedings{wang2020MAVEN,
  title={{MAVEN}: A Massive General Domain Event Detection Dataset},
  author={Wang, Xiaozhi and Wang, Ziqi and Han, Xu and Jiang, Wangyi and Han, Rong and Liu, Zhiyuan and Li, Juanzi and Li, Peng and Lin, Yankai and Zhou, Jie},
  booktitle={Proceedings of EMNLP 2020},
  year={2020}
}
```
for providing the `Maven` dataset.
and
```
@inproceedings{deng2020meta,
  title={Meta-learning with dynamic-memory-based prototypical network for few-shot event detection},
  author={Deng, Shumin and Zhang, Ningyu and Kang, Jiaojian and Zhang, Yichi and Zhang, Wei and Chen, Huajun},
  booktitle={Proceedings of the 13th International Conference on Web Search and Data Mining},
  pages={151--159},
  year={2020}
}
```
for providing the `FewEvent` dataset.


