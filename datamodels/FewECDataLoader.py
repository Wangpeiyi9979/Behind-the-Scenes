import os
import torch
import numpy as np
import random
import json
from tqdm import tqdm
import fire
import utils
from transformers import AutoTokenizer
import torch.nn as nn
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import copy
lemmatizer = WordNetLemmatizer()

class FewECDataLoader():
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
        elif method == 'blurry_uniform':
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
        if train:
            query_data = query_data[:self.opt.query_num]

     
        # batch data
        support_token_ids = [torch.tensor(d['token_id']).long() for d in support_data]
        support_bert_token_ids = [torch.tensor(d['bert_token_id']).long() for d in support_data]
        query_token_ids = [torch.tensor(d['token_id']).long() for d in query_data]
        query_bert_token_ids = [torch.tensor(d['bert_token_id']).long() for d in query_data]
        support_length = [d['length'] for d in support_data]
        query_length = [d['length'] for d in query_data]
        support_trigger_indices = [d['trigger_index'] for d in support_data]
        query_trigger_indices = [d['trigger_index'] for d in query_data]
        query_trigger_tokens = [d['trigger'] for d in query_data]
        support_trigger_tokens = [d['trigger'] for d in support_data]

        support_bert_trigger_indices = [d['bert_trigger_index'] for d in support_data]
        query_bert_trigger_indices = [d['bert_trigger_index'] for d in query_data]
        # padding & to tensor

        # save_data_to_file
        
        support_token_ids = nn.utils.rnn.pad_sequence(support_token_ids, batch_first=True, padding_value=0) # bsz * seq_len
        support_bert_token_ids = nn.utils.rnn.pad_sequence(support_bert_token_ids, batch_first=True, padding_value=0) # bsz * seq_len
        query_token_ids = nn.utils.rnn.pad_sequence(query_token_ids, batch_first=True, padding_value=0) # bsz * seq_len
        query_bert_token_ids = nn.utils.rnn.pad_sequence(query_bert_token_ids, batch_first=True, padding_value=0) # bsz * seq_len
        support_length = torch.tensor(support_length).long()
        query_length = torch.tensor(query_length).long()
        support_trigger_indices = torch.tensor(support_trigger_indices).long()
        query_trigger_indices = torch.tensor(query_trigger_indices).long()
        support_bert_trigger_indices = torch.tensor(support_bert_trigger_indices).long()
        query_bert_trigger_indices = torch.tensor(query_bert_trigger_indices).long()

        return {
            'support_token_ids': support_token_ids,
            'support_bert_token_ids': support_bert_token_ids,
            'query_token_ids': query_token_ids,
            'query_bert_token_ids': query_bert_token_ids,
            'support_length': support_length,
            'query_length': query_length,
            'support_trigger_indices': support_trigger_indices,
            'query_trigger_indices': query_trigger_indices,
            'support_bert_trigger_indices': support_bert_trigger_indices,
            'query_bert_trigger_indices': query_bert_trigger_indices,
            'query_trigger_tokens': query_trigger_tokens,
            'support_trigger_tokens': support_trigger_tokens
            
        }, support_label_id2rel_label

class FewECProcessor(object):
    def __init__(self, dataset, max_length):
        self.dataset = dataset
        self.data_dir = './dataset/{}_processed_data'.format(dataset)
        self.word2id = json.load(open('tool_data/word2id.json'))
        self.id2word = json.load(open('tool_data/id2word.json'))
        self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert_tokenizer.do_basic_tokenize = False # only tokenize single token to subwords
        self.word_emb = np.load('tool_data/word_embedding.npy')
        self.max_length = max_length
        self.drop_num = 0
        self.over_length_num = 0

    def create_examples(self, case='train'):
        self.label2id = utils.create_id2label_dict('tool_data/{}/{}_classes.txt'.format(self.dataset, case), reverse=True)
        self.id2label = utils.create_id2label_dict('tool_data/{}/{}_classes.txt'.format(self.dataset, case), reverse=False)
        json_datas = utils.read_jsonl_data('dataset/{}_sent_level/few_shot_{}_data.jsonl'.format(self.dataset, case))
        all_data = []
        type2trigger2sentence_index = {}
        type2sentence_index = {}

        for idx, data in enumerate(tqdm(json_datas)):
            return_data = self._create_single_example(data)
            if len(return_data) == 0:
                continue
            L = len(all_data)
            for i in range(len(return_data)):
                cur_label = return_data[i]['label']
                cur_trigger = return_data[i]['trigger']

                # 变成词根
                cur_trigger = lemmatizer.lemmatize(cur_trigger, pos='v')

                # new entry
                if cur_label not in type2trigger2sentence_index:
                    type2trigger2sentence_index[cur_label] = {}
                if cur_trigger not in type2trigger2sentence_index[cur_label]:
                    type2trigger2sentence_index[cur_label][cur_trigger] = []
                if cur_label not in type2sentence_index:
                    type2sentence_index[cur_label] = []

                type2trigger2sentence_index[cur_label][cur_trigger].append(L+i)
                type2sentence_index[cur_label].append(L+i)
            all_data.extend(return_data)

        print('drop num:{}; over length num:{}'.format(self.drop_num, self.over_length_num))
        print('total num: {}'.format(len(all_data)))
        self.drop_num = 0
        self.over_length_num = 0
        json.dump(obj=type2sentence_index, fp=open(os.path.join(self.data_dir, case + 'type2sentence_index.json'), 'w'))
        json.dump(obj=type2trigger2sentence_index, fp=open(os.path.join(self.data_dir, case + 'type2trigger2sentence_index.json'), 'w'))
        torch.save(all_data, os.path.join(self.data_dir, '{}_alldata.pth'.format(case)))

        synonym_dict = self.synonym(type2trigger2sentence_index)
        torch.save(synonym_dict, os.path.join(self.data_dir, '{}_synonym_dict.pth'.format(case)))

    def _create_single_example(self, data):
        return_data = []
        token = data['token']
        if len(token) > self.max_length:
            token = token[:self.max_length]
            self.over_length_num += 1
        token = [x.lower() for x in token]
        token_id = []
        for t in token:
            if t in self.word2id:
                token_id.append(self.word2id[t])
            else:
                token_id.append(self.word2id['[UNK]'])

        for event in data['events']:
            offset = event['offset']
            if offset[-1] - offset[0] > 1 or offset[-1] >= self.max_length:
                self.drop_num += 1
                continue
            event_type = event['type']
            label = self.label2id[event_type]

            # for BERT
            L = [0]
            bert_token = ['CLS']
            bert_token_id = self.bert_tokenizer.convert_tokens_to_ids(['[CLS]'])
            L.append(len(bert_token))
            for tok in token:
                subwords = self.bert_tokenizer.tokenize(tok)
                bert_token.extend(subwords)
                bert_token_id.extend(self.bert_tokenizer.convert_tokens_to_ids(subwords))
                L.append(len(bert_token))
            bert_token.append('[SEP]')
            bert_token_id.extend(self.bert_tokenizer.convert_tokens_to_ids(['[SEP]']))
            L.append(len(bert_token))
            bert_trigger = bert_token[L[offset[0]+1]]
            bert_trigger_index = L[offset[0]+1]

            # For example
            # ['I', 'have', 'an', 'apple'] -> ['I', 'ha', 've', 'an', 'app', 'le']
            # L = [0, 1, 3, 4, 6]
            return_data.append({
                'token': token,
                'token_id': token_id,
                'trigger': token[offset[0]],
                'trigger_index': offset[0],
                'bert_token': bert_token,
                'bert_token_id': bert_token_id,
                'bert_trigger': bert_trigger,
                'bert_trigger_index': bert_trigger_index,
                'L': L,
                'length': len(token),
                'label': label,
            })
        return return_data

    def synonym(self, type2trigger2sentence_index):
        synonym_dict = {}
        type_lists = list(type2trigger2sentence_index.keys())
        for t1 in type_lists:
            synonym_dict[t1] = {}
            for t2 in type_lists:
                if t1 != t2:
                    synonym_dict[t1][t2] = set()
        print('calculate synonym_dict')
        blurry_num = 6
        k = 20
        for i, type1 in enumerate(tqdm(type_lists)):
            triggers1 = list(type2trigger2sentence_index[type1].keys())
            triggers1 = list(filter(lambda x: x in self.word2id, triggers1))
            t1_embs = torch.tensor([self.word_emb[self.word2id[t]] for t in triggers1]) # k1 x dim'

            # 类内距离
            inner_dist = torch.norm(t1_embs.unsqueeze(1) - t1_embs.unsqueeze(0).expand(t1_embs.size(0), -1, -1), dim=-1, p=2)  # k1 x k1
            k1, _ = inner_dist.size()
            topk_inner_dist = torch.sort(inner_dist, dim=-1)[0][:, :min(k1, k)]
            inner_dist_mean = torch.mean(topk_inner_dist, dim=-1)  # k1   # 每一个trigger和类内top K小的trigger的平均距离

            for j, type2 in enumerate(type_lists):
                if type1 == type2:
                    continue
                triggers2 = list(type2trigger2sentence_index[type2].keys())
                triggers2 = list(filter(lambda x: x in self.word2id, triggers2))
                t2_embs = torch.tensor([self.word_emb[self.word2id[t]] for t in triggers2])  # k2 x dim

                # 类间距离
                inter_dist = torch.norm(t1_embs.unsqueeze(1) - t2_embs.unsqueeze(0).expand(t1_embs.size(0), -1, -1), dim=-1, p=2) #k1 x k2
                k2, _ = inter_dist.size()
                topk_inter_dist = torch.sort(inter_dist, dim=-1)[0][:,:min(k2, k)]
                inter_dist_mean = torch.mean(topk_inter_dist, dim=-1) # k1  # 每一个trigger和type2的trigger top K小的trigger的平均距离

                dist = -inner_dist_mean + inter_dist_mean  # k  # 内间距离参与到blurry trigger候选中非常重要
                # dist = inter_dist_mean  # k

                chose_index = torch.topk(dist, k=min(len(dist), blurry_num), largest=False)[1]
                for index in chose_index:
                    synonym_dict[type1][type2].add(triggers1[index.item()])
        return synonym_dict

# 通过vocab.txt和glove.**.txt处理得到可以直接load进nn.Embedding的npy文件
def process_pretrain_word_emb(vocab_file, pretrain_file, output_dir):
    # PADDING_TOKEN [pad]: 0,
    # UNKNOW_TOKEN [UNK]: 1

    # word & word_id
    word2id = {}
    with open(vocab_file, 'r') as f:
        words = f.readlines()
        for idx, word in enumerate(words):
            word2id[word.strip()] = idx
    id2word = {k: v for v, k in word2id.items()}
    vocab_size = len(word2id)

    # word & word embedding
    word2vec = {}
    if pretrain_file.endswith('txt'):
        with open(pretrain_file, 'r') as f:
            for line in f:
                line = line.strip()
                word, vec = line.split(' ', 1)
                vec = vec.strip().split(' ')
                vec = np.array(list(map(lambda x: float(x), vec)))
                word2vec[word] = vec
    else:
        word2vec_json_datas = json.load(open(pretrain_file))
        for data in word2vec_json_datas:
            word = data['word']
            vec = data['vec']
            vec = np.array(list(map(lambda x: float(x), vec)))
            word2vec[word] = vec

    word_dim = len(list(word2vec.values())[0])
    look_up_table = []
    for id in range(vocab_size):
        word = id2word[id]
        vec = word2vec.get(word, np.random.randn(word_dim))
        look_up_table.append(vec)
    look_up_table = np.array(look_up_table)

    # save
    json.dump(obj=word2id, fp=open(os.path.join(output_dir, 'word2id.json'), 'w'))
    json.dump(obj=id2word, fp=open(os.path.join(output_dir, 'id2word.json'), 'w'))
    np.save(os.path.join(output_dir, 'word_embedding.npy'), look_up_table)

def process_data(**kwargs):
    max_length = kwargs.get('max_length', 128)
    dataset = kwargs.get('dataset', 'maven')
    data_processor = FewECProcessor(dataset=dataset, max_length=max_length)
    data_processor.create_examples('train')
    data_processor.create_examples('val')
    data_processor.create_examples('test')


if __name__ == '__main__':
    fire.Fire()
