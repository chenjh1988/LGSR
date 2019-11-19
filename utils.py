#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/9/23 2:52
# @Author : {ZM7}
# @File : utils.py
# @Software: PyCharm

import networkx as nx
import numpy as np
import pdb
import node2vec
import random
import math
from collections import Counter


def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def build_graph_cjh(train_data, num_items):
    graph = nx.Graph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            # graph.add_edge(seq[i], seq[i + 1])
            if seq[i] != seq[i + 1]:
                graph.add_edge(seq[i], seq[i + 1])
    max_degree = 500
    adj = np.zeros((num_items, max_degree), dtype=int)
    deg = np.zeros((num_items, 1), dtype=int)

    for nodeid in graph.node:
        neighbors = np.array([neighbor for neighbor in graph.neighbors(nodeid)])
        deg[nodeid, 0] = min(max_degree, len(neighbors))
        if len(neighbors) == 0:
            continue
        if len(neighbors) > max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=False)
        elif len(neighbors) < max_degree:
            neighbors = np.pad(neighbors, (0, max_degree - neighbors.shape[0]), 'constant')
        adj[nodeid, :] = neighbors
    # pdb.set_trace()
    node_list, _ = np.where(deg > 0)
    return adj, deg, node_list


def build_graph_wei(train_data):
    graph = nx.Graph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            graph.add_edge(seq[i], seq[i + 1])
            # if seq[i] != seq[i + 1]:
            #     graph.add_edge(seq[i], seq[i + 1])
    max_degree = 500
    adj = np.zeros((37484, max_degree), dtype=int)
    deg = np.zeros((37484, 1), dtype=int)

    for nodeid in graph.node:
        neighbors = np.array([neighbor for neighbor in graph.neighbors(nodeid)])
        deg[nodeid, 0] = min(max_degree, len(neighbors))
        if len(neighbors) == 0:
            continue
        if len(neighbors) > max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=False)
        elif len(neighbors) < max_degree:
            neighbors = np.pad(neighbors, (0, max_degree - neighbors.shape[0]), 'constant')
        adj[nodeid, :] = neighbors
    # pdb.set_trace()
    node_list, _ = np.where(deg > 0)
    return adj, deg, node_list


def build_graph_node2vec(train_data):
    graph = nx.Graph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if seq[i] == seq[i + 1]:
                continue
            if graph.has_edge(seq[i], seq[i + 1]):
                graph[seq[i]][seq[i + 1]]['weight'] += 1
            else:
                graph.add_edge(seq[i], seq[i + 1], weight=1)
    return graph


def construct_node2vec(graph):
    # for edge in graph.edges():
    #     graph[edge[0]][edge[1]]['weight'] = 1
    nx_G = graph
    G = node2vec.Graph(nx_G, False, 1, 1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(1, 100)
    return walks


def calculate_centrality(G, mode='hits'):
    if mode == 'degree_centrality':
        a = nx.degree_centrality(G)
    else:
        h, a = nx.hits(G)

    max_a, min_a = 0, 100000
    for node in G.nodes():
        if max_a < a[node]:
            max_a = a[node]
        if min_a > a[node]:
            min_a = a[node]

    # pdb.set_trace()
    hits_dict = dict()
    for node in G.nodes():
        if max_a - min_a != 0:
            hits_dict[node] = (float(a[node]) - min_a) / (max_a - min_a)
        else:
            hits_dict[node] = 0
    return hits_dict


def cjh_random_walk(G, nodes, percentage, alpha=0, rand=random.Random(), start=None):
    """ Returns a truncated random walk.
        percentage: probability of stopping walking
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    if start:
        path = [start]
    else:
        path = [rand.choice(nodes)]

    while len(path) < 1 or random.random() > percentage:
        cur = path[-1]
        if len(G[cur]) > 0:
            if rand.random() >= alpha:
                add_node = rand.choice(list(G[cur].keys()))
                while add_node == cur:
                    add_node = rand.choice(list(G[cur].keys()))
                path.append(add_node)
            else:
                path.append(path[0])
        else:
            break
    return path


def build_deepwalk_corpus(G, percentage, maxT, minT, alpha=0, rand=random.Random()):
    walks = []
    hits_dict = calculate_centrality(G)
    # bb = np.array(list(hits_dict.values()))
    # pdb.set_trace()
    nodes = list(G.nodes())
    node_len = dict()
    for idx, node in enumerate(nodes):
        if len(G[node]) == 0:
            continue
        num_paths = max(int(math.ceil(maxT * hits_dict[node])), minT)
        node_len[node] = num_paths
        for cnt in range(num_paths):
            walks.append(cjh_random_walk(G, nodes, percentage, rand=rand, alpha=alpha, start=node))
        if (idx + 1) % 1000 == 0:
            print(idx + 1, 'over!')
    random.shuffle(walks)
    return walks, node_len


def prepare_data(seqs, labels, max_len=-1):
    n_samples = len(seqs)
    if max_len == -1:
        lengths = [len(s) for s in seqs]
    else:
        lengths = [min(len(s), max_len) for s in seqs]
    maxlen = max(lengths)
    labels = [i - 1 for i in labels]

    inputs = np.zeros((n_samples, maxlen)).astype('int64')
    lasts = np.zeros((n_samples)).astype('int64')
    items = []
    for idx, s in enumerate(seqs):
        inputs[idx, :lengths[idx]] = s[-lengths[idx]:]
        lasts[idx] = s[-1]
        items.extend(s)
    return inputs, lasts, labels, lengths


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return np.array(us_pois), np.array(us_msks), len_max


def split_validation_v2(all_train_seq, frac=0.1):
    train_len = int(len(all_train_seq) * (1 - frac))
    train_seq = all_train_seq[:train_len]
    valid_seq = all_train_seq[train_len:]

    def generate_seq(seqs):
        set_x, set_y = [], []
        for seq in seqs:
            for i in range(1, len(seq)):
                set_x.append(seq[:-i])
                set_y.append(seq[-i])
        return set_x, set_y

    train_set_x, train_set_y = generate_seq(train_seq)
    valid_set_x, valid_set_y = generate_seq(valid_seq)
    return (train_set_x, train_set_y), (valid_set_x, valid_set_y), train_seq


def split_validation_v1(train_set, frac=0.1):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - frac)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, all_seqs=None, sub_graph=False, method='ggnn', sparse=False, shuffle=False):
        # inputs = data[0]
        # inputs, mask, len_max = data_masks(inputs, [0])
        # self.inputs = np.asarray(inputs)
        # self.mask = np.asarray(mask)
        # self.len_max = len_max
        self.inputs = np.asarray(data[0])
        self.targets = np.asarray(data[1])
        self.length = len(self.inputs)
        self.shuffle = shuffle
        self.sub_graph = sub_graph
        self.sparse = sparse
        self.method = method
        self.remap = self.construct_remap(all_seqs)
        self.skip_window = 1

    def construct_remap(self, all_seqs):
        if all_seqs is None:
            return None
        seqs = []
        for i in all_seqs:
            seqs.extend(i)
        cnt = Counter(seqs)
        cc = cnt.most_common()
        nodes = [c[0] for c in cc]
        remap = dict(zip(nodes, range(len(nodes))))
        return remap

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            # self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        remain = self.length % batch_size
        if remain != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length - remain, self.length)
        return slices

    def get_slice(self, index):
        if self.method == 'ggnn':
            return self.prepare_data_ggnn(index)
        elif self.method == 'narm':
            return self.prepare_data_narm(self.inputs[index], self.targets[index])
        elif self.method == 'gru':
            return self.prepare_data_narm(self.inputs[index], self.targets[index])
        elif self.method == 'stamp':
            return self.prepare_data_stamp(self.inputs[index], self.targets[index])
        elif self.method == 'sa' or self.method == 'pa':
            return self.prepare_data_sa(self.inputs[index], self.targets[index])

    def prepare_data_narm(self, seqs, y, maxlen=19):
        n_samples = len(seqs)
        inputs = np.zeros((n_samples, maxlen)).astype('int64')
        seq_len = []
        for idx, s in enumerate(seqs):
            t_len = len(s)
            u_len = min(maxlen, t_len)
            seq_len.append(u_len)
            inputs[idx, :u_len] = s[t_len - u_len: t_len]
            # inputs[idx, :u_len] = s[:u_len]
        return inputs, seq_len, y

    def prepare_data_stamp(self, seqs, y):
        n_samples = len(seqs)
        # seq_len = [len(s) for s in seqs]
        seq_len = [min(len(s), 19) for s in seqs]
        maxlen = max(seq_len)
        inputs = np.zeros((n_samples, maxlen)).astype('int64')
        pos = np.zeros((n_samples, maxlen)).astype('int64')
        for idx, s in enumerate(seqs):
            inputs[idx, :seq_len[idx]] = s[:seq_len[idx]]
            pos[idx, :seq_len[idx]] = range(seq_len[idx], 0, -1)
            # inputs[idx, :u_len] = s[:u_len]
        return inputs, seq_len, pos, y

    def prepare_data_sa(self, seqs, y):
        n_samples = len(seqs)
        # seq_len = [len(s) for s in seqs]
        seq_len = [min(len(s), 19) for s in seqs]
        maxlen = max(seq_len)
        inputs = np.zeros((n_samples, maxlen)).astype('int64')
        pos = np.zeros((n_samples, maxlen)).astype('int64')
        for idx, s in enumerate(seqs):
            inputs[idx, :seq_len[idx]] = s[-seq_len[idx]:]
            pos[idx, :seq_len[idx]] = range(seq_len[idx], 0, -1)
            # inputs[idx, :u_len] = s[:u_len]
        return inputs, seq_len, pos, y

    def prepare_data_ggnn(self, index):
        items, n_node, A_in, A_out, alias_inputs = [], [], [], [], []
        u_inputs, mask, len_max = data_masks(self.inputs[index], [0])
        for u_input in u_inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in u_inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)

            A_in.append(u_A_in.transpose())
            A_out.append(u_A_out.transpose())
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return A_in, A_out, alias_inputs, items, mask, self.targets[index]


class Graph():
    def __init__(self, train_data, p, q, num_length, num_walks, direct, skip_window, batch_size):
        self.p = p
        self.q = q
        self.num_length = num_length
        self.num_walks = num_walks
        self.direct = direct
        self.skip_window = skip_window
        self.batch_size = batch_size
        self.train_data = train_data
        self.G = self.build_graph(train_data, direct)
        self.remap = self.construct_remap()
        self.num_node = len(self.remap)

    def build_graph(self, train_data, direct):
        graph = nx.DiGraph() if direct else nx.DiGraph()
        for seq in train_data:
            for i in range(len(seq) - 1):
                if seq[i] == seq[i + 1]:
                    continue
                if graph.has_edge(seq[i], seq[i + 1]):
                    graph[seq[i]][seq[i + 1]]['weight'] += 1
                    # graph[seq[i]][seq[i + 1]]['weight'] = 1
                else:
                    graph.add_edge(seq[i], seq[i + 1], weight=1)
        # pdb.set_trace()
        for node in graph.nodes():
            unnormal_weight = [graph[node][nbr]['weight'] for nbr in graph.neighbors(node)]
            total = sum(unnormal_weight)
            for nbr in graph.neighbors(node):
                graph[node][nbr]['weight'] /= total
        # test = []
        # for node in graph.nodes():
        #     test.append(sum([graph[node][nbr]['weight'] for nbr in graph.neighbors(node)]))
        # pdb.set_trace()
        return graph

    def construct_walks(self):
        n2v = node2vec.Graph(self.G, self.direct, self.p, self.q)
        n2v.preprocess_transition_probs()
        walks = n2v.simulate_walks(self.num_walks, self.num_length)
        return walks

    def construct_walks_orignal(self):
        walks = self.train_data
        return walks

    def construct_walks_bine(self):
        walks, node_len = build_deepwalk_corpus(
            self.G, percentage=0.15, maxT=32, minT=1, alpha=0, rand=random.Random()
        )
        return walks

    def construct_remap(self):
        nodes = self.G.degree().items()
        nodes = sorted(nodes, key=lambda x: x[1], reverse=True)
        nodes = [n[0] for n in nodes]
        remap = dict(zip(nodes, range(len(nodes))))
        return remap

    def generate_batch(self, data):
        random.shuffle(data)
        x, y = [], []
        for idx, t in enumerate(data):
            for i, s in enumerate(t):
                win = t[max(0, i - self.skip_window): i] + t[i + 1: min(len(t), i + self.skip_window)]
                for w in win:
                    x.append(w)
                    if self.remap is not None:
                        y.append([self.remap[s]])
                    else:
                        y.append([s])
                    if len(x) == self.batch_size:
                        yield idx, np.array(x), np.array(y)
                        x, y = [], []
        if x:
            yield idx, np.array(x), np.array(y)
