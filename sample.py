import configs
import json
import os
import utils
import torch
import random
import numpy as np
class Sampler():
    """
    FewEC Dataset
    """

    def __init__(self, opt, case='train'):

        self.type2sentence_index = json.load(open(os.path.join(opt.data_dir, case + 'type2sentence_index.json'), 'r'))
        self.type2trigger2sentence_index = json.load(open(os.path.join(opt.data_dir, case + 'type2trigger2sentence_index.json'), 'r'))
        self.target_classes = list(self.type2sentence_index.keys()) #
        self.label2id = utils.create_id2label_dict('tool_data/{}/{}_classes.txt'.format(opt.dataset, case), reverse=True)
        self.id2label = utils.create_id2label_dict('tool_data/{}/{}_classes.txt'.format(opt.dataset, case), reverse=False)
        self.word2id = json.load(open('tool_data/word2id.json'))
        self.id2word = json.load(open('tool_data/id2word.json'))
        all_data = torch.load(os.path.join(opt.data_dir, "{}_alldata.pth".format(case)))
        try:
            self.synonym_dict = torch.load(os.path.join(opt.data_dir, "{}_synonym_dict.pth".format(case)))  #  self.synonym_dict[event1][event2]: confusion triggers accorss event1 and event2
        except FileNotFoundError:
            print('no synonym dict file')
        self.all_data = all_data
        if case == 'train':
            self.N = opt.N_train
        else:
            self.N = opt.N
        self.K = opt.K
        self.opt = opt

    def next_one(self, method='normal', blurry_p=1, out_sample=False, train=False):
        target_classes = random.sample(self.target_classes, self.N)
        # import ipdb; ipdb.set_trace()
        support_data = [] # N * K
        query_data = [] # N
        support_label_id2rel_label = {i : self.id2label[int(v)] for i, v in enumerate(target_classes)}

        # sampling
        if method == 'normal': 
            for i, class_name in enumerate(target_classes):
                sentence_indexes = self.type2sentence_index[class_name]
                indices = np.random.choice(sentence_indexes, self.K + 1, False) # use the last one as query
                for idx in indices[:-1]:
                    support_data.append(self.all_data[idx])
                query_data.append(self.all_data[indices[-1]])

        elif method == 'trigger_uniform': 
            for i, class_name in enumerate(target_classes):
                all_trigger = list(self.type2trigger2sentence_index[class_name].keys())
                query_trigger =  np.random.choice(all_trigger, 1, False)
                all_trigger_left = list(set(all_trigger) - set(query_trigger))
                if len(all_trigger_left) >= self.K:
                    trigger_list = np.random.choice(all_trigger_left, self.K, False)
                else:
                    try:
                        trigger_list = np.random.choice(all_trigger_left, self.K, True)
                    except:
                        trigger_list = np.random.choice(all_trigger, self.K, True) # this class has only 1 instance
                all_select_idxs = set()
                for trigger in trigger_list:
                    try:
                        left_idxs = list(set(self.type2trigger2sentence_index[class_name][trigger]) - all_select_idxs)
                        idx = np.random.choice(left_idxs, 1, False)[0]
                    except:
                        idx = np.random.choice(self.type2trigger2sentence_index[class_name][trigger], 1, False)[0]
                    all_select_idxs.add(idx)
                    support_data.append(self.all_data[idx])
                idx = np.random.choice(self.type2trigger2sentence_index[class_name][query_trigger[0]], 1, False)[0]
                query_data.append(self.all_data[idx])
        elif method == 'confusion_uniform':
            for i, class_name in enumerate(target_classes):
                all_trigger = set(self.type2trigger2sentence_index[class_name].keys())
                blurry_trigger = set()
                for j, other_class_name in enumerate(target_classes):
                    if i != j:
                        blurry_trigger = blurry_trigger | set(self.synonym_dict[int(class_name)].get(int(other_class_name), []))
                remain_trigger = all_trigger - blurry_trigger

                query_trigger =  np.random.choice(list(all_trigger), 1, False)
                all_remain_trigger_left = remain_trigger - set(query_trigger)
                all_blurry_trigger_left = blurry_trigger - set(query_trigger)
                all_select_idxs = set()
                for _ in range(self.K):
                    p = np.random.uniform()
                    if len(all_blurry_trigger_left) == 0 and len(all_remain_trigger_left) == 0:
                        trigger = np.random.choice(list(all_trigger-set(query_trigger)), 1, False)[0]
                    elif 0 <= p <= blurry_p and len(all_blurry_trigger_left) !=0 or len(all_remain_trigger_left) == 0:
                        trigger = np.random.choice(list(all_blurry_trigger_left), 1, False)[0]
                        all_blurry_trigger_left = all_blurry_trigger_left - {trigger}
                    else:
                        trigger = np.random.choice(list(all_remain_trigger_left), 1, False)[0]
                        all_remain_trigger_left = all_remain_trigger_left - {trigger}
                    try:
                        left_idxs = list(set(self.type2trigger2sentence_index[class_name][trigger]) - all_select_idxs)
                        idx = np.random.choice(left_idxs, 1, False)[0]
                    except ValueError:
                        idx = np.random.choice(list(set(self.type2trigger2sentence_index[class_name][trigger])), 1, False)[0]
                    support_data.append(self.all_data[idx])
                    all_select_idxs.add(idx)
                idx = np.random.choice(self.type2trigger2sentence_index[class_name][query_trigger[0]], 1, False)[0]
                query_data.append(self.all_data[idx])
        else:
            assert 1 == 2

        # save_data_to_file
        support_tokens = [d['token'] for d in support_data]
        support_trigger_indices = [d['trigger_index'] for d in support_data]

        support_labels = []
        for k, v in support_label_id2rel_label.items():
            support_labels.extend([v] * self.K)
        query_tokens = [d['token'] for d in query_data]
        query_trigger_indices = [d['trigger_index'] for d in query_data]

        query_labels = []
        for k, v in support_label_id2rel_label.items():
            query_labels.append(v)
            
        return {
            'support_tokens': support_tokens,
            'query_tokens': query_tokens,
            'support_trigger_indices': support_trigger_indices,
            'query_trigger_indices': query_trigger_indices,
            'support_labels': support_labels,
            'query_labels': query_labels,
        }


from tqdm import tqdm
def sample(**keward):
    opt = getattr(configs, 'ProtoConfig')()
    opt.parse(keward)

    train_sample = 20000
    eval_sample = 2000
    test_sample = 10000

    # normal
    Ns=[5, 10]
    Ks=[5, 10]
    test_methods=['normal', 'trigger_uniform', 'confusion_uniform']
    for test_method in test_methods:
        for N in Ns:
            for K in Ks:
                opt.K = K
                opt.N = N
                print(f'method :{test_method} N: {N} K:{K}')
                train_sampler = Sampler(opt, case='train')
                dev_sampler = Sampler(opt, case='val')
                test_sampler = Sampler(opt, case='test')
                all_train_tasks = []
                all_eval_tasks = []
                all_test_tasks = []
                for _ in tqdm(range(train_sample)):
                    task = train_sampler.next_one(method='normal')
                    all_train_tasks.append(task)
                for _ in tqdm(range(eval_sample)):
                    task = dev_sampler.next_one(method=test_method)
                    all_eval_tasks.append(task)
                for _ in tqdm(range(test_sample)):
                    task = test_sampler.next_one(method=test_method)
                    all_test_tasks.append(task)
                out_dir = f'./sampled_tasks/{N}way-{K}shot-{test_method}'
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                with open(out_dir+'/train.jsonl', 'w') as f:
                    for task in all_train_tasks:
                        f.write(json.dumps(task) + '\n')
                with open(out_dir+'/dev.jsonl', 'w') as f:
                    for task in all_eval_tasks:
                        f.write(json.dumps(task) + '\n')
                with open(out_dir+'/test.jsonl', 'w') as f:
                    for task in all_test_tasks:
                        f.write(json.dumps(task) + '\n')



if __name__ == '__main__':
    import fire
    fire.Fire()
  