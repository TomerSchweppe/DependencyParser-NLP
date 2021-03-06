# !/usr/bin/env python
from chu_liu import *
import numpy as np
from random import shuffle


def tree_2_parent(tree):
    p_dict = dict()
    for p, children in tree.items():
        for c in children:
            p_dict[c] = p
    return p_dict


class Perceptron:
    """perceptron class"""

    def __init__(self, data, features):
        """init perceptron, extract all features"""
        self._data = data
        self._features = features
        self._f_dict_list = self.extract_features()
        self._window_list = self.window_list()
        self._full_graphs = dict()
        for sentence in self._data.sentences:
            if sentence.sentence_len not in self._full_graphs:
                self._full_graphs[sentence.sentence_len] = self.full_graph(sentence.sentence_len)

    def window_list(self):
        """save window list"""
        win_list = []
        for _, window in self._features(0, 1, self._data.sentences[0]):
            win_list.append(window)
        return win_list

    def features_dict(self, sentence):
        """generate features dictionary per sentence"""
        f_dict = dict()
        for h in range(sentence.sentence_len):
            for m in range(1, sentence.sentence_len):
                if h != m:
                    f_dict[(h, m)] = [shift for shift, _ in self._features(h, m, sentence)]
        return f_dict

    def extract_features(self):
        """extract features for all sentences"""
        f_dict_list = []
        for sentence in self._data.sentences:
            f_dict_list.append(self.features_dict(sentence))
        return f_dict_list

    def sentence_inference(self, w, sentence_len, f_dict):
        """inference on a given sentence"""

        def get_score(h, m):
            pos = 0
            score = 0
            for shift, window in zip(f_dict.get((h, m), -1), self._window_list):
                if shift != -1:
                    score += w[pos + shift]
                pos += window
            return score

        graph = Digraph(self._full_graphs[sentence_len], get_score)

        mst = graph.mst()
        return tree_2_parent(mst.successors)

    def full_graph(self, node_num):
        """generate full graph"""
        g = dict()
        g[0] = [m for m in range(1, node_num)]
        for h in range(1, node_num):
            g[h] = [m for m in range(1, node_num) if m != h]
        return g

    def update_weights(self, w, exact_d_tree, infer_d_tree, f_dict):
        """update weights"""
        for m, h in exact_d_tree.items():
            if infer_d_tree[m] != h:
                pos = 0
                for shift, window in zip(f_dict.get((h, m), -1), self._window_list):
                    if shift != -1:
                        w[pos + shift] += 1
                    pos += window

        for m, h in infer_d_tree.items():
            if exact_d_tree[m] != h:
                pos = 0
                for shift, window in zip(f_dict.get((h, m), -1), self._window_list):
                    if shift != -1:
                        w[pos + shift] += -1
                    pos += window

    def train(self, N):
        """
        train the model
        :param N: number of iterations
        :return w: learnt weights
        """
        w = np.zeros(self._features.features_len(), dtype=int)
        indices = [i for i in range(self._data.sentences_num)]
        for n in range(N):
            print('iteration', n + 1, '/', N)
            for idx in indices:
                sentence = self._data.sentences[idx]
                inference_d_tree = self.sentence_inference(w, sentence.sentence_len, self._f_dict_list[idx])
                if sentence.dependency_tree() != inference_d_tree:
                    self.update_weights(w, sentence.dependency_tree(), inference_d_tree, self._f_dict_list[idx])
            shuffle(indices)
        return w


if __name__ == '__main__':
    print('PASSED!')
