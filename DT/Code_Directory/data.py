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


class Data:
    """data class"""
    def __init__(self, file_name, is_labeled):
        """init sentences list"""
        self.sentences = []
        with open(file_name, 'r') as fh:
            for sentence_txt in fh.read().split('\n\n'):
                self.sentences.append(sentence_preprocess(sentence_txt, is_labeled))


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
    assert train.sentences[0](0).word == 'ROOT'
    assert train.sentences[0](0).pos == 'ROOT'
    assert train.sentences[0](1).word == 'Pierre'
    assert train.sentences[0](1).pos == 'NNP'
    assert test.sentences[2](0).word == 'ROOT'
    assert test.sentences[2](0).pos == 'ROOT'
    assert test.sentences[2](2).word == 'attorneys'
    assert test.sentences[2](2).pos == 'NNS'
    assert comp.sentences[0](0).word == 'ROOT'
    assert comp.sentences[0](0).pos == 'ROOT'
    assert comp.sentences[0](3).word == 'the'
    assert comp.sentences[0](3).pos == 'DT'

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

    print('PASSED!')
