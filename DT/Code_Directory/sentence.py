class WordTag:
    """word-tag class"""
    def __init__(self, word, pos):
        """init word and tag"""
        self.word = word
        self.pos = pos


class Sentence:
    """sentence class"""
    def __init__(self, word_list, pos_list):
        """init sentence list of WordTag"""
        self._sentence = [WordTag('ROOT', 'ROOT')]
        for word, pos in zip(word_list, pos_list):
            self._sentence.append(WordTag(word, pos))
        self._sentence_len = len(self._sentence)

    def __call__(self, index):
        """return word tag from sentence with index 'index'"""
        return self._sentence[index]


class LabeledSentence(Sentence):
    """labeled sentence class"""
    def __init__(self, word_list, pos_list, labels_list):
        """init dependency tree and labels list"""
        super(LabeledSentence, self).__init__(word_list, pos_list)
        self._labels_list = labels_list
        self._dt = dict()
        for m_1, h in enumerate(self._labels_list):
            m = m_1 + 1
            if h in self._dt:
                self._dt[h].append(m)
            else:
                self._dt[h] = [m]

    def dependency_tree(self):
        """return dependency tree"""
        return self._dt


if __name__ == '__main__':
    # validate WordTag class
    word_tag = WordTag('pc', 'nn')
    assert word_tag.word == 'pc'
    assert word_tag.pos == 'nn'
    # validate Sentence class

    # validate LabeledSentence class

