#encoding:utf-8
"""
@Time: 2020/11/17 17:26
@Author: Wang Peiyi
@Site : 
@File : CNNEncoder.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self, filters_num, filters, din):
        super(CNNEncoder, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(1, filters_num, (k, din), padding=(int(k / 2), 0)) for k in filters])
        self.init_model_weight()

    def init_model_weight(self):
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0.0)

    def forward(self, inputs, lengths=None):
        """

        Args:
            inputs: B x L x din
            lengths:

        Returns: B x L x filter_num*len(filgers)

        """
        x = inputs.unsqueeze(1)
        x = [conv(x).relu().squeeze(3).permute(0, 2, 1) for conv in self.convs]
        x = torch.cat(x, 2)
        return x