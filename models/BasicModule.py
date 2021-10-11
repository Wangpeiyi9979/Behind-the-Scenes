# -*- coding: utf-8 -*-

import torch
import time
import torch.nn as nn
from transformers import AutoTokenizer
import numpy as np
import copy
class BasicModule(torch.nn.Module):
    '''
    '''
    def __init__(self, opt):
        super(BasicModule, self).__init__()
        self.cost = nn.CrossEntropyLoss()
        if opt.debias:
            self.bias_glove = nn.Embedding.from_pretrained(torch.from_numpy(np.load(opt.word2vec_npy_path)).float(),
                                                     freeze=True, padding_idx=0)


    def get_rec_loss(self, support_token_ids, support_trigger_index, query_token_ids, query_trigger_index, support_length=None, query_length=None):

        query_token_ids, query_trigger_ids = self.mask_trigger(copy.deepcopy(query_token_ids),
                                              copy.deepcopy(query_trigger_index)
                                            )
        if self.opt.encoder in ['bert', 'lstm']:
            query_feature = self.get_bert_feature(query_token_ids, query_trigger_index)
            query_rep = self.select_single_token_rep(query_feature, query_trigger_index)
        else:
            query_feature = self.get_feature(query_token_ids, query_trigger_index, query_length)
            query_rep = torch.max(query_feature, 1)[0]
        query_logits = self.rec_linear(query_rep)
        rec_loss = self.cross_entopy_loss(query_logits, query_trigger_ids)
        return rec_loss

    def mask_trigger(self, token_ids, trigger_index):
        """
        Args:
            token_ids: B x L
            trigger_index: B
        Returns:
            B x L
        """
        B, L= token_ids.size()
        shift = torch.arange(B) * L
        if self.opt.use_gpu:
            shift = shift.cuda()
        trigger_index = trigger_index + shift
        trigger_ids = token_ids.contiguous().view(-1)[trigger_index]
        token_ids.contiguous().view(-1)[trigger_index] = 103
        return token_ids, trigger_ids
    def get_bias_probs(self, support_token_ids, query_token_ids, support_trigger_index, query_trigger_index, alpha, N, K):
        support_feature = self.bias_glove(support_token_ids)
        query_feature = self.bias_glove(query_token_ids)
        support_rep = self.select_single_token_rep(support_feature, support_trigger_index)
        query_rep = self.select_single_token_rep(query_feature, query_trigger_index)

        support_rep = torch.mean(support_rep.view(N, K, -1), 1)  # N x dim
        query_num = query_rep.size(0)
        support_rep = support_rep.unsqueeze(0).expand(query_num, -1, -1)  # query_num x NK x dim
        query_rep = query_rep.unsqueeze(1).expand(-1, N, -1)  # query_num x *N x dim

        if self.opt.dist == 'l2':
            logits = -self.__dist__(support_rep, query_rep, dim=2)  # query_num x N
        elif self.opt.dist == 'linear':
            logits = self.multilayer(torch.cat([support_rep, query_rep], dim=-1)).squeeze()
        elif self.opt.dist == 'cos':
            logits = self.__cos_dist__(support_rep, query_rep)  # query_num x N
        else:
            raise RuntimeError('xxx')
        probs = (alpha*logits).softmax(-1)
        # probs = logits.softmax(-1)
        return probs

    def get_bert_feature(self, bert_token_ids, trigger_index, disturb=None):
        """
        bert encoder, the type id of trigger is 1, othere is 0
        Args:
            bert_token_ids: N*K x L
            trigger_index: N*K

        Returns:
        """
        encoder_padding_mask = bert_token_ids.eq(0).logical_not_()
        type_token_ids = torch.zeros(bert_token_ids.size())
        if trigger_index.is_cuda:
            type_token_ids = type_token_ids.cuda()
        type_token_ids = type_token_ids.scatter_(dim=1, index=trigger_index.unsqueeze(1), value=1).long()
        word_embedding = self.sen_encoder.embeddings(bert_token_ids, token_type_ids=type_token_ids)

        if disturb is not None:
            NK, L = bert_token_ids.size()
            if self.opt.avg == 'head_marker':
                onehot = torch.zeros(NK, L).to(disturb.device).scatter_(1, (trigger_index + 1).unsqueeze(1),1)  # NK x length
            else:
                onehot = torch.zeros(NK, L).to(disturb.device).scatter_(1, trigger_index.unsqueeze(1),1)  # NK x length
            onehot_expand = onehot.unsqueeze(-1).expand(-1, -1, L)  # NK x length x length
            disturb_expand = disturb.unsqueeze(1).expand(-1, L, -1)  # NK x length x dim
            disturb_expand = torch.matmul(onehot_expand, disturb_expand) / L  # NK x length x dim
            word_embedding = word_embedding + disturb_expand

        if self.opt.add_trigger_embedding:
            trigger_embedding = self.select_single_token_rep(word_embedding, trigger_index)
            word_embedding = word_embedding + trigger_embedding.unsqueeze(1).expand(-1, word_embedding.size(1), -1)

        tout, _ = self.sen_encoder(inputs_embeds=word_embedding, attention_mask=encoder_padding_mask, return_dict=False)
        if self.opt.model.lower() in ['mlman', 'ouralpha', 'ourbeta']:
            max_length = encoder_padding_mask.long().sum(1).max().item()
            return tout, encoder_padding_mask.float(), max_length
        else:
            return tout


    def get_feature(self, token_ids, trigger_index, length, disturb=None):
        """

        Args:
            token_ids: NK x length
            trigger_index: NK
            length: NL
            disturb: NK x dim
        Returns:
            tout: NK x length x hidden_size
            input_mask: NK x length
            max_length: int
        """
        word_embeddings = self.word_emb(token_ids)
        if disturb is not None:
            NK, L = token_ids.size()
            if self.opt.avg == 'head_marker':
                onehot = torch.zeros(NK, L).to(disturb.device).scatter_(1, (trigger_index + 1).unsqueeze(1), 1) # NK x length
            else:
                onehot = torch.zeros(NK, L).to(disturb.device).scatter_(1, trigger_index.unsqueeze(1), 1) # NK x length
            onehot_expand = onehot.unsqueeze(-1).expand(-1, -1, L) # NK x length x length
            disturb_expand = disturb.unsqueeze(1).expand(-1, L, -1) # NK x length x dim
            disturb_expand = torch.matmul(onehot_expand, disturb_expand) / L  # NK x length x dim
            word_embeddings = word_embeddings + disturb_expand

        if self.opt.add_trigger_embedding:
            if self.opt.avg == 'head_marker':
                trigger_embedding = self.select_single_token_rep(word_embeddings, trigger_index + 1)
            else:
                trigger_embedding = self.select_single_token_rep(word_embeddings, trigger_index)
            word_embeddings = word_embeddings + trigger_embedding.unsqueeze(1).expand(-1, word_embeddings.size(1), -1)

        pos_index = self.get_pos_index(trigger_index, token_ids.size(1))  # N*K x max_sen_length
        pos_emb = self.pos_emb(pos_index)
        embeddings = torch.cat([word_embeddings, pos_emb], dim=-1)
        embeddings = self.dropout(embeddings)

        if self.opt.encoder == 'transformer':
            input_mask: torch.BoolTensor = (token_ids != 0).bool()
            embeddings = self.linear_berfore(embeddings).relu()
            tout = self.sen_encoder(embeddings, input_mask)
        else:
            tout = self.sen_encoder(embeddings, length)  # N*K x max_sen_length x dim
        tout = self.dropout(tout)

        if self.opt.model.lower() in ['mlman', 'ouralpha', 'ourbeta']:
            input_mask = (token_ids != 0).float()
            max_length = input_mask.long().sum(1).max().item()
            return tout, input_mask, max_length
        else:
            return tout

    def get_pos_index(self, trigger_indexs, batch_sen_length):
        """
        根据trigger位置生成相对pos_index, trigger位置的pos index为self.max_sen_length
        input:
            trigger_index: N*K
            batch_sen_length: sentence length after padding in this batch
        return:
            pos_index: NK x max_sen_length
        """
        NK = len(trigger_indexs)
        anchor = torch.arange(batch_sen_length).unsqueeze(0).expand(NK, -1)  # NK x max_sen_length
        if trigger_indexs.is_cuda:
            anchor = anchor.cuda()
        shift = (self.max_sen_length - trigger_indexs).unsqueeze(1).expand(-1, batch_sen_length)
        pos_index = anchor + shift
        return pos_index

    def select_single_token_rep(self, batch_rep, token_pos):
        """
        batch_rep: B x L x dim
        token_pos: B

        Returns:
            B x dim
        """
        B, L, dim = batch_rep.size()
        shift = torch.arange(B) * L
        if self.opt.use_gpu:
            shift = shift.cuda()
        token_pos = token_pos + shift
        res = batch_rep.contiguous().view(-1, dim)[token_pos]
        return res

    def cross_entopy_loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size.
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).float())


    def __dist__(self, x, y, dim, score=None):
        if score is None:
            return (torch.pow(x - y, 2)).sum(dim)
        else:
            return (torch.pow(x - y, 2) * score).sum(dim)

    def __cos_dist__(self, x, y):
        # query_num x *N x dim
        return torch.nn.functional.cosine_similarity(x, y, dim=-1)