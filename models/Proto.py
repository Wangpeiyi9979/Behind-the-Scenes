#encoding:utf-8
"""
@Time: 2020/10/13 15:43
@Author: Wang Peiyi
@Site : 
@File : Proto
"""
import torch
import numpy as np
import torch.nn as nn
from .Encoder import CNNEncoder, LSTMEncoder, TransformerEncoder, make_model
from transformers import *
from .BasicModule import BasicModule

class Proto(BasicModule):
    def __init__(self, opt):
        opt.encoder = opt.encoder.lower()
        self.opt = opt
        self.max_sen_length = opt.max_sen_length
        self.model_name = opt.model
        super(Proto, self).__init__(opt)

        if opt.encoder == 'bert':
            self.sen_encoder  = AutoModel.from_pretrained(opt.bert_model_path)
            vocab_size = self.sen_encoder.embeddings.word_embeddings.weight.size(0)

        else:
            self.pos_emb = nn.Embedding(self.max_sen_length * 2, opt.pos_dim)
            if opt.use_glove:
                self.word_emb = nn.Embedding.from_pretrained(torch.from_numpy(np.load(opt.word2vec_npy_path)).float(),freeze=opt.freeze, padding_idx=0)
            else:
                self.word_emb = nn.Embedding(opt.glove_vocab_size, opt.word_dim)
            if opt.encoder == 'cnn':
                self.sen_encoder = CNNEncoder(
                    filters_num=opt.filter_num,
                    filters=opt.filters,
                    din=opt.word_dim + opt.pos_dim
                )
            elif opt.encoder == 'lstm':
                self.sen_encoder = LSTMEncoder(
                    din=opt.word_dim + opt.pos_dim,
                    dout=opt.lstm_dout // 2,
                    num_layers=opt.lstm_num_layers)
            elif opt.encoder == 'transformer':
                self.linear_berfore = nn.Linear(opt.word_dim+opt.pos_dim, opt.d_model)
                self.sen_encoder = make_model(
                    d_model=opt.d_model,
                    nhead=opt.nhead,
                    num_layers=opt.transformer_num_layers)
            else:
                raise RuntimeError('no encoder')
            vocab_size = self.word_emb.weight.size(0)

        self.dropout = nn.Dropout(opt.dropout)
        if opt.dist == 'linear':
            self.multilayer = nn.Sequential(nn.Linear(opt.hidden_size * 2, opt.hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(opt.hidden_size, 1))
        if self.opt.rec:
            self.rec_linear = nn.Sequential(
                nn.Linear(opt.hidden_size, vocab_size)
            )
            for m in self.rec_linear:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
    def forward(self, batch, N, K, support_disturb=None, query_disturb=None, train=True, alpha=None):
        if self.opt.encoder == 'bert':
            support_feature = self.get_bert_feature(batch['support_bert_token_ids'],
                                                    batch['support_bert_trigger_indices'],
                                                    support_disturb)
            query_feature = self.get_bert_feature(batch['query_bert_token_ids'],
                                                  batch['query_bert_trigger_indices'],
                                                  query_disturb)
        else:
            support_feature = self.get_feature(batch['support_token_ids'],
                                               batch['support_trigger_indices'],
                                               batch['support_length'],
                                               support_disturb)
            query_feature = self.get_feature(batch['query_token_ids'],
                                             batch['query_trigger_indices'],
                                             batch['query_length'],
                                             query_disturb)
        if self.opt.avg == 'max':
                support_rep = torch.max(support_feature, 1)[0]
                query_rep = torch.max(query_feature, 1)[0]
        elif self.opt.avg == 'trigger':
            if self.opt.encoder == 'bert':
                support_rep = self.select_single_token_rep(support_feature, batch['support_bert_trigger_indices'])
                query_rep = self.select_single_token_rep(query_feature, batch['query_bert_trigger_indices'])
            else:
                support_rep = self.select_single_token_rep(support_feature, batch['support_trigger_indices'])
                query_rep = self.select_single_token_rep(query_feature, batch['query_trigger_indices'])
        else:
            raise RuntimeError('no avg method for pooling')
        # support_rep: NK x dim
        # query_rep: Q  x dim
        support_rep = torch.mean(support_rep.view(N, K, -1), 1)  # N x dim
        query_num = query_rep.size(0)
        support_rep = support_rep.unsqueeze(0).expand(query_num, -1, -1) # query_num x N x dim
        query_rep = query_rep.unsqueeze(1).expand(-1, N, -1) # query_num x N x dim
        logits = -self.__dist__(support_rep, query_rep, dim=2)  # query_num x N
        pred = torch.max(logits, -1)[1]
        return_data = {}
        return_data['logits'] = logits
        return_data['pred'] = pred
        if self.opt.rec and train:
            if self.opt.encoder == 'bert':
                rec_loss = self.get_rec_loss(batch['support_bert_token_ids'],
                                             batch['support_bert_trigger_indices'],
                                             batch['query_bert_token_ids'],
                                             batch['query_bert_trigger_indices']
                                             )
            else:
                rec_loss = self.get_rec_loss(batch['support_token_ids'],
                                             batch['support_trigger_indices'],
                                             batch['query_token_ids'],
                                             batch['query_trigger_indices'],
                                             batch['support_length'],
                                             batch['query_length']                                             )
            return_data['J_rec'] = rec_loss
        # return_data['trigger_index'] =  batch['support_trigger_indices']
        # return_data['max_index'] = torch.max(support_feature, 1)[1]
        return return_data
