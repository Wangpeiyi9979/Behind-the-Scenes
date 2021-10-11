# encoding:utf-8
"""
@Time: 2020/10/13 15:43
@Author: Wang Peiyi
@Site :
@File : Proto
"""
import torch
import torch.nn as nn
from .BasicModule import BasicModule
import numpy as np
from nltk.stem import WordNetLemmatizer


class Hard(BasicModule):
    def __init__(self, opt):
        super(Hard, self).__init__(opt)
        self.opt = opt
        self.max_sen_length = opt.max_sen_length
        self.model_name = opt.model
        self.lemmatizer = WordNetLemmatizer()
        
    def forward(self, batch, N, K, train=True, alpha=None):
        support_trigger_tokens = batch['support_trigger_tokens'] # N*K
        query_trigger_tokens = batch['query_trigger_tokens'] # query
        
        support_trigger_tokens_lemma = [self.lemmatizer.lemmatize(x, pos='v') for x in support_trigger_tokens]
        query_trigger_tokens_lemma = [self.lemmatizer.lemmatize(x, pos='v') for x in query_trigger_tokens]
        
        logits = batch['support_token_ids'].new_zeros((len(query_trigger_tokens_lemma), N), dtype=torch.float)
        for i, q in enumerate(query_trigger_tokens_lemma):
            for j, s in enumerate(support_trigger_tokens_lemma):
                if q == s:
                    logits[i][j//K] = 1
                    break
            else:
                # randomly pick the first event type
                logits[i][0] = 1
        pred = torch.max(logits, -1)[1]
        
        return_data = {}
        return_data['logits'] = logits # query * N
        return_data['pred'] = pred # query
        return return_data


