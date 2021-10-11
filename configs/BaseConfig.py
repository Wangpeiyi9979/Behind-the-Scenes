#encoding:utf-8
"""
@Time: 2020/11/19 9:21
@Author: Wang Peiyi
@Site : 
@File : BaseConfig.py
"""
class BaseConfig(object):
    model = 'Base'
    word2vec_npy_path = './tool_data/word_embedding.npy'
    bert_model_path = "bert-base-uncased"
    data_dir = './dataset/maven_processed_data'
    dataset = 'maven'
    optimizer = 'adam'
    ckpt_dir = './checkpoints'
    log_dir = ''
    encoder='cnn'
    test_sample_method = 'blurry_uniform'
    glove_vocab_size = 400004
    pretrain_model = None
    add_trigger_embedding = False
    seed = 9979
    gpu_id = 3
    use_gpu = True
    dropout = 0.2
    weight_decay = 1e-6
    lr_step_size = 200
    grad_iter = 1
    val_iter = 2000
    val_step = 300
    train_iter = 10000
    test_iter = 10000
    word_dim = 300
    pos_dim = 50
    max_sen_length = 128
    save_opt = None
    early_stop = 2
    N_train = 5
    blurry_p = 1
    adv_train = False
    int_loss=False
    ephsion = 0.5
    dist = 'l2'
    freeze=False
    N = 5
    K = 5
    lr = 1e-4
    tao = 1
    out_sample=False
    debias=False
    label_smoothing = 0.1
    w_rec = 1e-1
    bert_hidden_size = 768
    rec=False
    hidden_size=0
    query_num=5 # BERT爆显存, 降低每次的query_num, 训练时
    def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)
        setattr(self, 'dataset', kwargs.get('dataset', 'maven'))
        setattr(self, 'data_dir', './dataset/{}_processed_data'.format(self.dataset))
        if 'save_opt' not in kwargs:
            setattr(self, 'save_opt', "{}_{}_{}_{}".format(self.dataset, self.model, self.N, self.K))
        if 'log_dir' not in kwargs:
            setattr(self, 'log_dir', "{}_log.txt".format(self.model))
        if 'query_num' not in kwargs:
            setattr(self, 'query_num', self.N_train)

        if kwargs.get('model') not in ['Glove', 'Hard', 'Match']:
            if kwargs.get('encoder', 'cnn') == 'cnn':
                setattr(self, 'hidden_size', self.filter_num * len(self.filters))
            elif kwargs.get('encoder', 'cnn') == 'lstm':
                setattr(self, 'hidden_size', self.lstm_dout)
            elif kwargs.get('encoder', 'cnn') == 'bert':
                setattr(self, 'hidden_size', self.bert_hidden_size)

    def __repr__(self):
        info = "*"*10 + 'model config' + "*" * 10 + '\n'
        info += '\n'.join(['{}: {}'.format(k, getattr(self, k)) for k in dir(self) if k[0] != '_' and k != 'parse'])
        info += '\n' + ''"*"*34
        return info
