# !/usr/bin/env python
class Sentence:
    """sentence class"""

    def __init__(self, word_list, pos_list):
        """init sentence list of WordPos"""
        self._sentence = [('ROOT', 'ROOT')]
        for word, pos in zip(word_list, pos_list):
            self._sentence.append((word, pos))
        self.sentence_len = len(self._sentence)

    def __call__(self, index):
        """return word tag from sentence with index 'index'"""
        return self._sentence[index][0], self._sentence[index][1]


class LabeledSentence(Sentence):
    """labeled sentence class"""

    def __init__(self, word_list, pos_list, labels_list):
        """init dependency tree and labels list"""
        super(LabeledSentence, self).__init__(word_list, pos_list)
        self._labels_list = labels_list
        self._dt = dict()
        for m_1, h in enumerate(self._labels_list):
            m = m_1 + 1
            self._dt[m] = h

    def dependency_tree(self):
        """return dependency tree"""
        return self._dt


if __name__ == '__main__':
    word_list = ['ofir', 'tomer', 'nadav', 'roy']
    pos_list = ['S', 'S', 'T', 'T']

    # validate sentence
    sen = Sentence(word_list, pos_list)
    assert sen.sentence_len == 4 + 1
    assert sen._sentence == [('ROOT', 'ROOT'), ('ofir', 'S'), ('tomer', 'S'), ('nadav', 'T'), ('roy', 'T')]

    # validate labaled sentence
    labels_list = [0, 1, 2, 3]
    l_sen = LabeledSentence(word_list, pos_list, labels_list)
    assert l_sen._sentence == [('ROOT', 'ROOT'), ('ofir', 'S'), ('tomer', 'S'), ('nadav', 'T'), ('roy', 'T')]
    assert l_sen.dependency_tree() == {0: [1], 1: [2], 2: [3], 3: [4]}

    print('PASSED!')
