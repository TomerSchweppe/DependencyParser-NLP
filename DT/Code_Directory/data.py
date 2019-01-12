# !/usr/bin/env python
from sentence import *


def sentence_preprocess(sentence_txt, is_labeled):
    """generate Sentence object from given sentence text"""
    word_list = []
    pos_list = []
    labels_list = []
    for line in sentence_txt.split('\n'):
        if line.strip():
            args = line.split()
            word_list.append(args[1])
            pos_list.append(args[3])
            labels_list.append(args[6])
    if is_labeled:
        return LabeledSentence(word_list, pos_list, labels_list)
    return Sentence(word_list, pos_list)


def word_pos_wordpos_lists(sentences):
    """
    generate word, pos, word-pos pairs lists
    """
    word_set = set('ROOT')
    pos_set = set('ROOT')
    word_pos_pairs = set([('ROOT', 'ROOT')])

    for sentence in sentences:
        for idx in range(sentence.sentence_len):
            word, pos = sentence(idx)
            word_set.add(word)
            pos_set.add(pos)
            word_pos_pairs.add((word, pos))
    return sorted(list(word_set)), sorted(list(pos_set)), sorted(list(word_pos_pairs))


class Data:
    """data class"""
    def __init__(self, file_name, is_labeled):
        """init sentences list, vocab list, pos list"""
        self.sentences = []
        with open(file_name, 'r') as fh:
            for sentence_txt in fh.read().split('\n\n'):
                self.sentences.append(sentence_preprocess(sentence_txt, is_labeled))
        self.vocab_list, self.pos_list, self.word_pos_pairs = word_pos_wordpos_lists(self.sentences)


if __name__ == '__main__':
    # validate Data class
    train = Data('train.labeled', is_labeled=True)
    test = Data('test.labeled', is_labeled=True)
    comp = Data('comp.unlabeled', is_labeled=False)

    # validate type
    assert type(train.sentences[0]) == LabeledSentence
    assert type(test.sentences[0]) == LabeledSentence
    assert type(comp.sentences[0]) == Sentence

    # validate word & pos
    assert train.sentences[0](0)[0] == 'ROOT'
    assert train.sentences[0](0)[1] == 'ROOT'
    assert train.sentences[0](1)[0] == 'Pierre'
    assert train.sentences[0](1)[1] == 'NNP'
    assert test.sentences[2](0)[0] == 'ROOT'
    assert test.sentences[2](0)[1] == 'ROOT'
    assert test.sentences[2](2)[0] == 'attorneys'
    assert test.sentences[2](2)[1] == 'NNS'
    assert comp.sentences[0](0)[0] == 'ROOT'
    assert comp.sentences[0](0)[1] == 'ROOT'
    assert comp.sentences[0](3)[0] == 'the'
    assert comp.sentences[0](3)[1] == 'DT'

    # validate dependency tree
    tmp_dict = test.sentences[0].dependency_tree()
    assert tmp_dict['4'] == [1, 2, 3]
    assert tmp_dict['6'] == [7]
    assert tmp_dict['5'] == [4, 6, 17]
    assert tmp_dict['7'] == [9]
    assert tmp_dict['16'] == [15]
    assert tmp_dict['10'] == [11]
    assert tmp_dict['0'] == [5]
    assert tmp_dict['9'] == [8, 10, 14]
    assert tmp_dict['14'] == [16]
    assert tmp_dict['11'] == [12, 13]
    assert len(tmp_dict) == 10

    # validate vocab
    assert 'ROOT' in train.vocab_list
    assert 'ROOT' in test.vocab_list
    assert 'ROOT' in comp.vocab_list
    assert 'chairman' in train.vocab_list
    assert 'Minpeco' in test.vocab_list
    assert 'Washington' in comp.vocab_list

    # validate pos
    assert 'ROOT' in train.pos_list
    assert 'ROOT' in test.pos_list
    assert 'ROOT' in comp.pos_list
    assert 'VBD' in train.pos_list
    assert 'RB' in test.pos_list
    assert 'DT' in comp.pos_list

    # validate word pos pairs
    assert ('ROOT', 'ROOT') in train.word_pos_pairs
    assert ('ROOT', 'ROOT') in test.word_pos_pairs
    assert ('ROOT', 'ROOT') in comp.word_pos_pairs
    assert ('cancer', 'NN') in train.word_pos_pairs
    assert ('federal', 'JJ') in test.word_pos_pairs
    assert ('entertained', 'VBN') in comp.word_pos_pairs

    #print(len(train.vocab_list), len(train.pos_list), len(train.word_pos_pairs))

    print('PASSED!')
