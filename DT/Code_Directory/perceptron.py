# !/usr/bin/env python
from chu_liu import *
import numpy as np
import time
from multiprocessing import cpu_count
from multiprocessing import Pool


class Perceptron:
    """perceptron class"""
    def __init__(self, data, features, edge_max_len = -1):
        """init perceptron, extract all features"""
        self._data = data
        self._features = features
        self._edge_max_len = edge_max_len
        self._f_dict_list = self.extract_features(data, features)


    def features_dict(self, sentence, features):
        """generate features dictionary per sentence"""
        f_dict = dict()
        for h in range(sentence.sentence_len):
            for m in range(1, sentence.sentence_len):
                if  self._edge_max_len == -1 or abs(h-m) <= self._edge_max_len:
                    f_dict[(h, m)] = features(h, m, sentence)
                else:
                    f_dict[(h, m)] = []
        return f_dict

    def extract_features(self, data, features):
        """extract features for all sentences"""
        f_dict_list = []
        for sentence in data.sentences:
            f_dict_list.append(self.features_dict(sentence, features))
        return f_dict_list

    def sentence_inference(self, w, sentence_len, f_dict):
        """inference on a given sentence"""
        def get_score(h, m):
            pos = 0
            score = 0
            for shift, window in f_dict[(h, m)]:
                if shift != -1:
                    score += w[pos + shift]
                pos += window
            return score

        graph = Digraph(self.full_graph(sentence_len), get_score)
        mst = graph.mst()
        return mst.successors

    def full_graph(self, node_num):
        """generate full graph"""
        g = dict()
        g[0] = [m for m in range(1, node_num)]
        for h in range(1, node_num):
            g[h] = [m for m in range(1, node_num) if m != h]
        return g

    def update_weights(self, w, d_tree, sentence, f_dict, sign):
        """update weights"""
        for h, m_list in d_tree.items():
            for m in m_list:
                pos = 0
                for shift, window in f_dict[(h, m)]:
                    if shift != -1:
                        w[pos + shift] += sign
                    pos += window

    def train(self, N):
        """
        train the model

        :param N: number of iterations
        :return w: learnt weights
        """
        w = np.zeros(self._features.features_len(), dtype=int)

        for n in range(N):
            print('iteration', n + 1,'/', N)
            for idx, sentence in enumerate(self._data.sentences):
                inference_d_tree = self.sentence_inference(w, sentence.sentence_len, self._f_dict_list[idx])
                if inference_d_tree != sentence.dependency_tree():
                    self.update_weights(w, sentence.dependency_tree(), sentence, self._f_dict_list[idx], 1)
                    self.update_weights(w, inference_d_tree, sentence, self._f_dict_list[idx], -1)
        return w


if __name__ == '__main__':
    print('PASSED!')



