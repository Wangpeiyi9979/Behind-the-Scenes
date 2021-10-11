#encoding:utf-8
from .BaseConfig import BaseConfig
class ProtoConfig(BaseConfig):
    model = 'Proto'

    # cnn config
    filter_num = 300
    filters = [3]

    # lstm config
    lstm_dout = 300
    lstm_num_layers = 1

    # transformer
    d_model=300
    nhead = 4
    transformer_num_layers = 2

    # bert
    bert_hidden_size=768

    avg = 'max'

    use_glove = True