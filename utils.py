import networkx as nx
import numpy as np
import pdb
import node2vec
import random
import math
from collections import Counter


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
    def __init__(self, data, all_seqs=None, method='ha', shuffle=False, maxlen=19):
        self.inputs = np.asarray(data[0])
        self.targets = np.asarray(data[1])
        self.length = len(self.inputs)
        self.shuffle = shuffle
        self.method = method
        self.remap = self.construct_remap(all_seqs)
        self.skip_window = 1
        self.maxlen = maxlen

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
        if self.method == 'sr_gnn':
            return self.prepare_data_sr_gnn(index)
        elif self.method == 'ha':
            return self.prepare_data_ha(self.inputs[index], self.targets[index])

    def prepare_data_ha(self, seqs, y):
        n_samples = len(seqs)
        # seq_len = [len(s) for s in seqs]
        seq_len = [min(len(s), self.maxlen) for s in seqs]
        maxlen = max(seq_len)
        inputs = np.zeros((n_samples, maxlen)).astype('int64')
        pos = np.zeros((n_samples, maxlen)).astype('int64')
        for idx, s in enumerate(seqs):
            inputs[idx, :seq_len[idx]] = s[-seq_len[idx]:]
            pos[idx, :seq_len[idx]] = range(seq_len[idx], 0, -1)
            # inputs[idx, :u_len] = s[:u_len]
        return inputs, seq_len, pos, y

    def prepare_data_sr_gnn(self, index):
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
    def __init__(self, train_data, num_length, num_walks, skip_window, batch_size):
        self.num_length = num_length
        self.num_walks = num_walks
        self.skip_window = skip_window
        self.batch_size = batch_size
        self.train_data = train_data
        self.G = self.build_graph(train_data)
        self.remap = self.construct_remap()
        self.num_node = len(self.remap)

    def build_graph(self, train_data):
        graph = nx.DiGraph()
        for seq in train_data:
            for i in range(len(seq) - 1):
                if seq[i] == seq[i + 1]:
                    continue
                if graph.has_edge(seq[i], seq[i + 1]):
                    graph[seq[i]][seq[i + 1]]['weight'] += 1
                else:
                    graph.add_edge(seq[i], seq[i + 1], weight=1)
        # pdb.set_trace()
        for node in graph.nodes():
            unnormal_weight = [graph[node][nbr]['weight'] for nbr in graph.neighbors(node)]
            total = sum(unnormal_weight)
            for nbr in graph.neighbors(node):
                graph[node][nbr]['weight'] /= total
        return graph

    def construct_walks(self):
        n2v = node2vec.Graph(self.G, True, 1, 1)
        n2v.preprocess_transition_probs()
        walks = n2v.simulate_walks(self.num_walks, self.num_length)
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
