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
import torch.nn.functional as F

class ProtoHATT(BasicModule):
    def __init__(self, opt):
        super(ProtoHATT, self).__init__(opt)
        opt.encoder = opt.encoder.lower()
        self.opt = opt
        self.max_sen_length = opt.max_sen_length
        self.model_name = opt.model

        if opt.encoder == 'bert':
            self.sen_encoder  = AutoModel.from_pretrained(opt.bert_model_path)
            self.hidden_size = opt.bert_hidden_size
            vocab_size = self.sen_encoder.embeddings.word_embeddings.weight.size(0)
        else:
            self.pos_emb = nn.Embedding(self.max_sen_length * 2, opt.pos_dim)
            if opt.use_glove:
                self.word_emb = nn.Embedding.from_pretrained(torch.from_numpy(np.load(opt.word2vec_npy_path)).float(),freeze=False, padding_idx=0)
            else:
                self.word_emb = nn.Embedding(opt.glove_vocab_size, opt.word_dim)
            if opt.encoder == 'cnn':
                self.sen_encoder = CNNEncoder(
                    filters_num=opt.filter_num,
                    filters=opt.filters,
                    din=opt.word_dim + opt.pos_dim
                )
                self.hidden_size = opt.filter_num * len(opt.filters)
            elif opt.encoder == 'lstm':
                self.sen_encoder = LSTMEncoder(
                    din=opt.word_dim + opt.pos_dim,
                    dout=opt.lstm_dout // 2,
                    num_layers=opt.lstm_num_layers)
                self.hidden_size = opt.lstm_dout
            elif opt.encoder == 'transformer':
                self.linear_berfore = nn.Linear(opt.word_dim+opt.pos_dim, opt.d_model)
                self.sen_encoder = make_model(
                    d_model=opt.d_model,
                    nhead=opt.nhead,
                    num_layers=opt.transformer_num_layers)
                self.hidden_size = opt.d_model
            else:
                raise RuntimeError('no encoder')
            vocab_size = self.word_emb.weight.size(0)
        self.dropout = nn.Dropout(opt.dropout)
        if self.opt.rec:
            self.rec_linear = nn.Linear(opt.hidden_size, vocab_size)
            nn.init.xavier_normal_(self.rec_linear.weight)
        # for instance-level attention
        self.fc = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # for feature-level attention
        self.conv1 = nn.Conv2d(1, 32, (opt.K, 1), padding=(opt.K // 2, 0))
        self.conv2 = nn.Conv2d(32, 64, (opt.K, 1), padding=(opt.K // 2, 0))
        self.conv_final = nn.Conv2d(64, 1, (opt.K, 1), stride=(opt.K, 1))
        
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
        elif self.opt.avg in ['trigger', 'head_marker']:
            if self.opt.encoder == 'bert':
                support_rep = self.select_single_token_rep(support_feature, batch['support_bert_trigger_indices'])
                query_rep = self.select_single_token_rep(query_feature, batch['query_bert_trigger_indices'])
            else:
                support_rep = self.select_single_token_rep(support_feature, batch['support_trigger_indices'])
                query_rep = self.select_single_token_rep(query_feature, batch['query_trigger_indices'])
        else:
            raise RuntimeError('no avg method for pooling')
        # support_rep: NK x dim
        # query_rep: Q x dim
        Q = query_rep.size(0)
        support_rep = support_rep.view(N, K, self.hidden_size)
        query_rep = query_rep.view(Q, self.hidden_size)
        
        # feature-level attention
        fea_att_score = support_rep.view(N, 1, K, self.hidden_size) # (N, 1, K, D)
        fea_att_score = F.relu(self.conv1(fea_att_score)) # (N, 32, K, D) 
        fea_att_score = F.relu(self.conv2(fea_att_score)) # (N, 64, K, D)
        fea_att_score = self.dropout(fea_att_score)
        fea_att_score = self.conv_final(fea_att_score) # (N, 1, 1, D)
        
        fea_att_score = fea_att_score.tanh()
        fea_att_score = fea_att_score.view(N, self.hidden_size).unsqueeze(0) # (1, N, D)
        
        # instance-level attention 
        support_rep = support_rep.unsqueeze(0).expand(Q, -1, -1, -1) # (Q, N, K, D)
        support_for_att = self.fc(support_rep) 
        query_for_att = self.fc(query_rep.unsqueeze(1).unsqueeze(2).expand(-1, N, K, -1)) # (Q, N, K, D)
        ins_att_score = F.softmax(torch.tanh(support_for_att * query_for_att).sum(-1), dim=-1) # (Q, N, K)
        support_proto = (support_rep * ins_att_score.unsqueeze(3).expand(-1, -1, -1, self.hidden_size)).sum(2) # (Q, N, D)
        
        # Prototypical Networks 
        query_rep = query_rep.unsqueeze(1).expand(-1, N, -1) # (Q, N, D)
        logits = -self.__dist__(support_proto, query_rep, 2, fea_att_score)
        pred = torch.max(logits, -1)[1]
        return_data = {}
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
                                             batch['query_length'])
            return_data['J_rec'] = rec_loss
        return_data['logits'] = logits
        return_data['pred'] = pred
        return return_data
    