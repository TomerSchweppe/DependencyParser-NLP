# !/usr/bin/env python
from chu_liu import *
import numpy as np
import time
from random import randint
from random import shuffle

class Perceptron:
    """perceptron class"""
    def __init__(self, data, features):
        """init perceptron, extract all features"""
        self._data = data
        self._features = features
        self._f_dict_list = self.extract_features(data, features)
        self._full_graphs = dict()
        for sentence in self._data.sentences:
            if sentence.sentence_len not in self._full_graphs:
                self._full_graphs[sentence.sentence_len] = self.full_graph(sentence.sentence_len)

    def prune(self, h, m, sentence):
        """prune implausible edges"""
        # source: http://aclweb.org/anthology/C10-1007
        h_pos = sentence(h)[1]
        m_pos = sentence(m)[1]

        if h_pos in ['"', '.', ';', '|', 'CC', 'PRP$', 'PRP', 'EX', '-RRB-', '-LRB-']: return True

        if m_pos in ['EX', 'LS', 'POS', 'PRP$'] and h < m: return True

        if m_pos in ['.', 'RP'] and h > m: return True

        if h == 0 and h_pos in [',', 'DT']: return True

        if h < m:
            if h_pos == 'DT' and m_pos in ['DT', 'JJ', 'NN', 'NNP', 'NNS', '.']: return True

            if h_pos == 'CD' and m_pos == 'CD': return True

            if h_pos == 'NN' and m_pos in ['DT', 'NNP']: return True

            if h_pos == 'NNP' and m_pos in ['DT', 'NN', 'NNS']: return True

        if h > m:
            if m_pos == 'DT' and h_pos in ['DT', 'IN', 'JJ', 'NN', 'NNP']: return True

            if m_pos == 'NNP' and h_pos == 'IN': return True

            if m_pos == 'IN' and h_pos == 'JJ': return True
        return False

    def features_dict(self, sentence, features):
        """generate features dictionary per sentence"""
        f_dict = dict()
        for h in range(sentence.sentence_len):
            for m in range(1, sentence.sentence_len):
                if  not self.prune(h, m ,sentence):
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

        graph = Digraph(self._full_graphs[sentence_len], get_score)

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
        indices = [i for i in range(self._data.sentences_num)]
        for n in range(N):
            print('iteration', n + 1,'/', N)
            for idx in indices:
                sentence = self._data.sentences[idx]
                inference_d_tree = self.sentence_inference(w, sentence.sentence_len, self._f_dict_list[idx])
                if inference_d_tree != sentence.dependency_tree():
                    self.update_weights(w, sentence.dependency_tree(), sentence, self._f_dict_list[idx], 1)
                    self.update_weights(w, inference_d_tree, sentence, self._f_dict_list[idx], -1)
            shuffle(indices)
        return w


if __name__ == '__main__':
    print('PASSED!')



