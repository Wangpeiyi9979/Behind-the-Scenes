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


class Glove(BasicModule):
    def __init__(self, opt):
        super(Glove, self).__init__(opt)
        self.opt = opt
        self.max_sen_length = opt.max_sen_length
        self.model_name = opt.model
        self.word_emb = nn.Embedding.from_pretrained(torch.from_numpy(np.load(opt.word2vec_npy_path)).float(),
                                                     freeze=opt.freeze, padding_idx=0)
        self.dropout = nn.Dropout(opt.dropout)
      
    def forward(self, batch, N, K, train=True, alpha=None):
        support_feature = self.word_emb(batch['support_token_ids'])
        query_feature = self.word_emb(batch['query_token_ids'])  
        support_rep = self.select_single_token_rep(support_feature, batch['support_trigger_indices'])
        query_rep = self.select_single_token_rep(query_feature, batch['query_trigger_indices'])

        support_rep = torch.mean(support_rep.view(N, K, -1), 1)  # N x dim
        query_num = query_rep.size(0)
        support_rep = support_rep.unsqueeze(0).expand(query_num, -1, -1)  # query_num x NK x dim
        query_rep = query_rep.unsqueeze(1).expand(-1, N, -1)  # query_num x *N x dim

        logits = -self.__dist__(support_rep, query_rep, dim=2)  # query_num x N
        pred = torch.max(logits, -1)[1]
        return_data = {}
        return_data['logits'] = logits
        return_data['pred'] = pred
        return return_data


