#encoding:utf-8
import fire
import json
import sys
import torch
import os
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel
import random
from tqdm import trange
from torch.optim.lr_scheduler import LambdaLR
from utils import RunningAverage
from torch.utils.data import DataLoader
import datamodels
import configs
import models

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def select_batch_trigger(batch_token_ids, trigger_pos):
    B, L = batch_token_ids.size()
    shift = torch.arange(B) * L
    if batch_token_ids.is_cuda:
        shift = shift.cuda()
    trigger_pos = trigger_pos + shift
    res = batch_token_ids.contiguous().view(-1)[trigger_pos]
    return res


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run(**keward):
    opt = getattr(configs, keward['model']+'Config')()
    opt.parse(keward)
    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)
    setup_seed(opt.seed)

    model = getattr(models, opt.model)(opt)
    print(opt)
    if opt.use_gpu:
        model.cuda()

    bert_parameters = []
    for m in model.modules():
        if isinstance(m, BertModel):
            bert_parameters += m.parameters()
    rest_params = filter(lambda p: id(p) not in list(map(id, bert_parameters)), model.parameters())

    DataLoader = getattr(datamodels, 'FewECDataLoader')
    train_data_loader = DataLoader(opt, case='train')
    dev_data_loader = DataLoader(opt, case='val')
    test_data_loader = DataLoader(opt, case='test')

    device_anchor = torch.zeros(1)
    if opt.use_gpu: device_anchor = device_anchor.cuda()
    device_anchor = device_anchor.device
    print("Start training...")
    if opt.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = optim.Adam([
            {'params': bert_parameters, 'lr': 3e-5},
            {'params': rest_params, 'lr': opt.lr}])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size)
    model = model.to(device_anchor)
    model.train()
    print("parameter num:")
    print(get_parameter_number(model))
    if opt.label_smoothing > 0:
        myloss = LabelSmoothingLoss(opt.label_smoothing, opt.N_train).cuda()

    # Training
    best_acc = 0
    best_test_acc = 0
    iter_loss = RunningAverage()
    iter_right = RunningAverage()
    early_stop = 0
    it = 0

    alpha = 1 if opt.debias else None
    while early_stop < opt.early_stop and it <= opt.train_iter:
        batch, _ = train_data_loader.next_one('normal', out_sample=opt.out_sample, train=True)

        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device_anchor)

        return_data = model(batch, opt.N_train, opt.K, train=True, alpha=alpha)
  
        query_label_list = torch.arange(batch['query_token_ids'].size(0))
        query_label_list = query_label_list.to(device_anchor)

        if opt.label_smoothing == 0:
            allloss = model.cross_entopy_loss(return_data['logits'] / opt.tao, query_label_list)
        else:
            allloss = myloss(return_data['logits'] / opt.tao, query_label_list)

        right = model.accuracy(return_data['pred'], query_label_list)

        if 'J_rec' in return_data:
            allloss = allloss + opt.w_rec * return_data['J_rec']
        for key in return_data:
            if 'loss' in key:
                allloss = allloss + return_data[key]
        try:
            allloss.backward()  # Glove:  freeze glove, wrong
        except:
            pass

        if opt.adv_train:
            if opt.encoder != 'bert':
                support_trigger_token = select_batch_trigger(batch['support_token_ids'], batch['support_trigger_indices'])
                query_trigger_token = select_batch_trigger(batch['query_token_ids'], batch['query_trigger_indices'])

                support_trigger_grad = model.word_emb.weight.grad[support_trigger_token]  # N*K, word_dim
                query_trigger_grad = model.word_emb.weight.grad[query_trigger_token] #N, word_dim
            else:
                support_trigger_token = select_batch_trigger(batch['support_bert_token_ids'], batch['support_bert_trigger_indices'])
                query_trigger_token = select_batch_trigger(batch['query_bert_token_ids'], batch['query_bert_trigger_indices'])

                support_trigger_grad = model.sen_encoder.embeddings.word_embeddings.weight.grad[support_trigger_token]  # N*K, word_dim
                query_trigger_grad = model.sen_encoder.embeddings.word_embeddings.weight.grad[query_trigger_token]  # N, word_dim

            support_disturb = opt.ephsion * support_trigger_grad / torch.norm(support_trigger_grad, p=2, dim=-1, keepdim=True)
            query_disturb = opt.ephsion * query_trigger_grad / torch.norm(query_trigger_grad, p=2, dim=-1, keepdim=True)
            return_data = model(batch, opt.N_train, opt.K, support_disturb=support_disturb, query_disturb=query_disturb, train=True, alpha=alpha)
            if opt.label_smoothing == 0:
                allloss_disturb = model.cross_entopy_loss(return_data['logits'] / opt.tao, query_label_list)
            else:
                allloss_disturb = myloss(return_data['logits'] / opt.tao, query_label_list)
            if 'J_incon' in return_data:
                allloss_disturb = allloss_disturb + return_data['J_incon']
            for key in return_data:
                if 'loss' in key:
                    allloss_disturb = allloss_disturb + return_data[key]

            allloss_disturb.backward()
            right_distrub = model.accuracy(return_data['pred'], query_label_list)
            iter_right.update(right_distrub.item())
            iter_loss.update(allloss_disturb.item())

        if it % opt.grad_iter == 0:
            optimizer.step()
            optimizer.zero_grad()

        iter_loss.update(allloss.item())
        iter_right.update(right.item())

        sys.stdout.write(
            'step: {} | loss: {:.6f}/{:.6f} acc: {:.3f}% | best dev accuracy:{:.3f}%'.format(it + 1, iter_loss(),
                                                                                      allloss.item(),
                                                                                      100 * iter_right(),
                                                                                      best_acc) + '\r')
        sys.stdout.flush()
        if (it + 1) % opt.val_step == 0:
            acc = eval(model, opt, device_anchor, dev_data_loader)
            model.train()
            early_stop += 1
            if acc > best_acc:
                early_stop = 0
                best_acc = acc
                print('Best checkpoint')
                print('[Save]: {} to checkpoints/{}_best.pt'.format(model.model_name, opt.save_opt))
                torch.save({'opt': opt, 'parameters': model.state_dict(), 'optimizer': scheduler.state_dict(),
                            'current_acc': acc, 'best_acc': best_acc}, 'checkpoints/{}_best.pt'
                           .format(opt.save_opt))
            iter_loss.clear()
            iter_right.clear()

        it += 1
    print("\n####################\n")
    print("Finish training")
    test_acc = test(model, opt, device_anchor, test_data_loader,
                         ckpt='checkpoints/{}_best.pt'.format(opt.save_opt))
    print("{0:}-way-{1:}-shot test   Test accuracy: {2:3.2f}".format(opt.N, opt.K, test_acc))

    if not os.path.exists('result'):
        os.mkdir('result')
    with open(os.path.join('result', opt.log_dir), 'a') as f:
        f.write("{} {:.3f} {:.3f}\n".format(opt.save_opt, best_acc, test_acc))
    os.system('rm checkpoints/{}_best.pt'.format(opt.save_opt))

def test(model,opt,
         device_anchor,
         data_loader,
         ckpt):
    print("Use test dataset")
    iter_num = opt.test_iter
    try:
        checkpoint = torch.load(ckpt, map_location='cpu')
        try:
            model.load_state_dict(checkpoint['parameters'])
        except KeyError:
            model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print('No {}'.format(ckpt))
    iter_right = RunningAverage()
    model.eval()
    with torch.no_grad():
        for it in range(iter_num):
            batch, _ = data_loader.next_one(opt.test_sample_method, blurry_p=opt.blurry_p,
                                            out_sample=opt.out_sample)
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device_anchor)

            return_data = model(batch, opt.N, opt.K, train=False)
            query_label_list = torch.arange(opt.N)
            query_label_list = query_label_list.to(device_anchor)
            right = model.accuracy(return_data['pred'], query_label_list)
            iter_right.update(right.item())
            sys.stdout.write(
                '[EVAL] step: {:4} | accuracy: {:3.2f}%'.format(it + 1,
                                                                100 * iter_right()) + '\r')
            sys.stdout.flush()
        print("")
    return 100 * iter_right()

def eval(model,opt,
         device_anchor,
         data_loader):
    print("")
    model.eval()
    iter_num = opt.val_iter
    iter_right = RunningAverage()
    with torch.no_grad():
        for it in range(iter_num):
            batch, _ = data_loader.next_one(opt.test_sample_method, blurry_p=opt.blurry_p, out_sample=opt.out_sample)
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device_anchor)

            return_data = model(batch, opt.N, opt.K, train=False)
            query_label_list = torch.arange(opt.N)
            query_label_list = query_label_list.to(device_anchor)
            right = model.accuracy(return_data['pred'], query_label_list)
            iter_right.update(right.item())
            sys.stdout.write(
                '[EVAL] step: {:4} | accuracy: {:3.2f}%'.format(it + 1,
                                                                100 * iter_right()) + '\r')
            sys.stdout.flush()
        print("")
    return 100*iter_right()

class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, N, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (N - 1)
        one_hot = torch.full((N,), smoothing_value)
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        output = F.log_softmax(output, dim=-1)
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='batchmean')

if __name__ == '__main__':
    fire.Fire()
