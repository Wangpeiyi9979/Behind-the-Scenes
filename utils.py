#encoding:utf-8
"""
@Time: 2020/12/3 14:52
@Author: Wang Peiyi
@Site : 
@File : utils.py
"""
import json
class RunningAverage():
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def clear(self):
        self.steps = 0
        self.total = 0

    def __call__(self):
        return self.total / float(self.steps)

def read_jsonl_data(data_path):
    all_data = []
    f = open(data_path, encoding='utf-8')
    datas = f.readlines()
    for data in datas:
        all_data.append(json.loads(data))
    return all_data

def create_id2label_dict(path, reverse=False):
    id2label = {}
    with open(path, 'r') as f:
        labels = f.readlines()
        for idx, label in enumerate(labels):
            id2label[idx] = label.strip()
    if reverse:
        return {v : k for k, v in id2label.items()}
    return id2label