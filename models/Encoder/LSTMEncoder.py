#encoding:utf-8
"""
@Time: 2020/11/17 17:31
@Author: Wang Peiyi
@Site : 
@File : LSTMEncoder.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class LSTMEncoder(nn.Module):
    def __init__(self, din, dout, num_layers=1,bidirectional=True, batch_first=True):
        super(LSTMEncoder, self).__init__()
        self.rnn = nn.LSTM(din, dout, num_layers=num_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.init_weight()
    def init_weight(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            else:
                nn.init.constant_(param, 0.0)
    def forward(self, inputs, lengths=None):
        """
        Args:
            inputs: B x L x din
            lengths:

        Returns: B x L x hidden_size

        """
        if lengths is None:
            rnn_output, _ = self.rnn(inputs)
        else:
            x = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
            x, _ = self.rnn(x)
            rnn_output, length = pad_packed_sequence(x, batch_first=True)
        return rnn_output  # (B, L, 2/1hidden_