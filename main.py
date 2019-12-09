from __future__ import division
import numpy as np
import tensorflow as tf
import random
from model import LGSR
from utils import split_validation_v1, split_validation_v2, Data, Graph
import logging
import pickle
import argparse
import time
import sys
import json
import pdb


def m_print(log):
    logging.info(log)
    print(log)
    return


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose1_64', help='dataset diginetica/yoochoose1_4/yoochoose1_64')
parser.add_argument('--method', type=str, default='ha', help='recommendation module method ha/sr_gnn')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size')
parser.add_argument('--emb_size', type=int, default=100, help='hidden state size')
parser.add_argument('--l2', type=float, default=0.0, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps for sr_gnn')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--max_len', type=int, default=19, help='sequence max length')

parser.add_argument('--cide', type=int, default=2, help='the train frequency of cross-session item dependency encoder (cide)')
parser.add_argument('--cide_batch_size', type=int, default=8192, help='the batch size of cide')
parser.add_argument('--num_length', type=int, default=50, help='the number of length in random walk')
parser.add_argument('--num_walks', type=int, default=13, help='the number of walk in random walk')
parser.add_argument('--skip_window', type=int, default=8, help='the window size in skip-gram')
parser.add_argument('--n_sample', type=int, default=512, help='the negative sample size in skip-gram')
parser.add_argument('--rand_seed', type=int, default=1111)
parser.add_argument('--log_file', type=str, default='./default.txt')


opt = parser.parse_args()
random.seed(opt.rand_seed)
np.random.seed(opt.rand_seed)
tf.set_random_seed(opt.rand_seed)
logging.basicConfig(level=logging.INFO, format='%(message)s', filename=opt.log_file, filemode='w')

all_train_seq = pickle.load(open('./datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
if opt.validation:
    if opt.cide > 0:
        train_data, test_data, all_train_seq = split_validation_v2(all_train_seq, frac=0.1)
    else:
        train_data = pickle.load(open('./datasets/' + opt.dataset + '/train.txt', 'rb'))
        train_data, test_data = split_validation_v1(train_data, frac=0.1)
else:
    train_data = pickle.load(open('./datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('./datasets/' + opt.dataset + '/test.txt', 'rb'))

if opt.dataset == 'diginetica':
    n_node = 43098
elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
    n_node = 37484

train_data = Data(train_data, all_seqs=all_train_seq, method=opt.method, shuffle=True, maxlen=opt.max_len)
test_data = Data(test_data, method=opt.method, shuffle=False, maxlen=opt.max_len)
graph = Graph(all_train_seq, opt.num_length, opt.num_walks, opt.skip_window, opt.cide_batch_size)
g_node = graph.num_node

model = LGSR(hidden_size=opt.hidden_size,
             emb_size=opt.emb_size,
             n_node=n_node,
             method=opt.method,
             lr=opt.lr,
             l2=opt.l2,
             step=opt.step,
             decay=0.1 * len(train_data.inputs) / opt.batch_size,
             lr_dc=opt.lr_dc,
             dropout=opt.dropout,
             cide=opt.cide,
             g_node=g_node,
             n_sample=opt.n_sample,
             max_len=opt.max_len)

m_print(json.dumps(opt.__dict__, indent=4))
m_print('train len: %d, test len: %d' % (train_data.length, test_data.length))
best_result = [0, 0]
best_epoch = [0, 0]
early_stop = False
pre_cite_loss = 0.0
walks = graph.construct_walks()

for epoch in range(opt.epoch):
    m_print('<==================epoch: %d==================>' % epoch)
    if not early_stop:
        for cide_step in range(opt.cide):
            e_loss = []
            cide_batch = graph.generate_batch(walks)
            fetches_node = [model.cide_loss, model.cide_opt, model.global_step_cide]
            train_start = time.time()
            for _, x, y in cide_batch:
                # pdb.set_trace()
                s_loss, _, _ = model.run_cide(fetches_node, x, y)
                e_loss.append(s_loss)
            cost_time = time.time() - train_start
            m_print('Step: %d, Train cide_Loss: %.4f, Cost: %.2f' % (cide_step, np.mean(e_loss), cost_time))
            if abs(pre_cite_loss - np.mean(e_loss)) <= 0.0005:
                early_stop = True
            pre_cite_loss = np.mean(e_loss)

    # pdb.set_trace()
    slices = train_data.generate_batch(opt.batch_size)
    fetches = [model.rec_opt, model.rec_loss, model.global_step]
    m_print('start train: ' + time.strftime('%m-%d %H:%M:%S ', time.localtime(time.time())))
    loss_ = []
    for i, j in zip(slices, np.arange(len(slices))):
        batch_input = train_data.get_slice(i)
        _, loss, _ = model.run_rec(fetches, batch_input)
        loss_.append(loss)
    loss = np.mean(loss_)

    slices = test_data.generate_batch(opt.batch_size)
    m_print('start predict:' + time.strftime('%m-%d %H:%M:%S ', time.localtime(time.time())))
    hit, mrr, test_loss_, sa_wei, a_wei, ans = [], [], [], [], [], []
    for i, j in zip(slices, np.arange(len(slices))):
        batch_input = test_data.get_slice(i)
        scores, test_loss, tk, satt, att = model.run_rec(
            [model.logits, model.rec_loss, model.top_k, model.satt, model.att], batch_input, is_train=False)
        test_loss_.append(test_loss)
        sa_wei.append(satt)
        a_wei.append(att)
        ans.append(tk)

        targets = batch_input[-1]
        # index = np.argsort(scores, 1)[:, -20:]
        for score, target in zip(tk, targets):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (1 + np.where(score == target - 1)[0][0]))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    test_loss = np.mean(test_loss_)
    if hit >= best_result[0]:
        best_result[0] = hit
        best_epoch[0] = epoch
    if mrr >= best_result[1]:
        best_result[1] = mrr
        best_epoch[1] = epoch
    m_print('train_loss: %.4f, test_loss: %.4f, Recall@20: %.4f, MMR@20: %.4f' % (loss, test_loss, hit, mrr))
    m_print('Best Recall@20: %.4f, Best MMR@20: %.4f, Epoch: %d, %d\n' %
            (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
