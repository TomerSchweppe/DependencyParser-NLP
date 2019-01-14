# !/usr/bin/env python
from sentence import *

class Feature:
    """base feature class"""

    def __init__(self, vocab_list, pos_list, word_pos_pairs):
        """store vocab, pos, word-pos lists and generate index dictionaries"""
        self._vocab_list = vocab_list
        self._pos_list = pos_list
        self._word_pos_pairs = word_pos_pairs
        self._word_idx = {word: idx for idx, word in enumerate(self._vocab_list)}
        self._pos_idx = {pos: idx for idx, pos in enumerate(self._pos_list)}
        self._word_pos_pairs_idx = {(word, pos): idx for idx, (word, pos) in enumerate(self._word_pos_pairs)}

    def __call__(self):
        """call function"""
        pass

    def feature_vec(self):
        """return tuple of '1' position and feature length """
        pass


class WordPos(Feature):
    """word pos feature class"""
    def __init__(self, vocab_list, pos_list, word_pos_pairs):
        """init"""
        super(WordPos, self).__init__(vocab_list, pos_list, word_pos_pairs)

    def __call__(self, word, pos):
        """generate feature tuple"""
        return self._word_pos_pairs_idx.get((word, pos), -1), len(self._word_pos_pairs)


class Word(Feature):
    """word feature class"""
    def __init__(self, vocab_list, pos_list, word_pos_pairs):
        """init"""
        super(Word, self).__init__(vocab_list, pos_list, word_pos_pairs)

    def __call__(self, word):
        """generate feature tuple"""
        return self._word_idx.get(word, -1), len(self._word_idx)


class Pos(Feature):
    """pos feature class"""
    def __init__(self, vocab_list, pos_list, word_pos_pairs):
        """init"""
        super(Pos, self).__init__(vocab_list, pos_list, word_pos_pairs)

    def __call__(self, pos):
        """generate feature tuple"""
        return self._pos_idx.get(pos, -1), len(self._pos_idx)


class WordPosPos(Feature):
    """word pos pos feature class"""
    def __init__(self, vocab_list, pos_list, word_pos_pairs):
        """init"""
        super(WordPosPos, self).__init__(vocab_list, pos_list, word_pos_pairs)

    def __call__(self, word, pos, other_pos):
        """generate feature tuple"""
        other_pos_idx = self._pos_idx.get(other_pos, -1)
        word_pos_idx = self._word_pos_pairs_idx.get((word, pos), -1)
        if other_pos_idx == -1 or word_pos_idx == -1:
            return -1, len(self._word_pos_pairs_idx) * len(self._pos_idx)
        return other_pos_idx * len(self._word_pos_pairs_idx) + word_pos_idx, len(self._word_pos_pairs_idx) * len(self._pos_idx)


class PosPos(Feature):
    """pos pos feature class"""
    def __init__(self, vocab_list, pos_list, word_pos_pairs):
        """init"""
        super(PosPos, self).__init__(vocab_list, pos_list, word_pos_pairs)

    def __call__(self, pos, other_pos):
        """generate feature tuple"""
        pos_idx = self._pos_idx.get(pos, -1)
        other_pos_idx = self._pos_idx.get(other_pos, -1)
        if pos_idx == -1 or other_pos_idx == -1:
            return -1, len(self._pos_idx)**2
        return other_pos_idx * len(self._pos_idx) + pos_idx, len(self._pos_idx)**2


class BasicFeatures:
    """basic features class"""

    def __init__(self, vocab_list, pos_list, word_pos_pairs):
        """init features"""
        self._f_word_pos = WordPos(vocab_list, pos_list, word_pos_pairs)
        self._f_word = Word(vocab_list, pos_list, word_pos_pairs)
        self._f_pos = Pos(vocab_list, pos_list, word_pos_pairs)
        self._f_word_pos_pos = WordPosPos(vocab_list, pos_list, word_pos_pairs)
        self._f_pos_pos = PosPos(vocab_list, pos_list, word_pos_pairs)

    def features_num(self):
        """return the number of features"""
        return len(self(0, 0, Sentence(['', ''], ['', ''])))


    def features_len(self):
        """return the number of feature bits"""
        sum = 0
        for _, size in self(0,0,Sentence(['',''],['',''])):
            sum += size
        return sum


    def __call__(self, h, m ,sentence):
        """return list of all features"""
        p_word = sentence(h)[0]
        c_word = sentence(m)[0]
        p_pos = sentence(h)[1]
        c_pos = sentence(m)[1]

        f_p_word_p_pos = self._f_word_pos(p_word, p_pos)
        f_p_word = self._f_word(p_word)
        f_p_pos = self._f_pos(p_pos)
        f_c_word_c_pos = self._f_word_pos(c_word, c_pos)
        f_c_word = self._f_word(c_word)
        f_c_pos = self._f_pos(c_pos)
        f_c_word_c_pos_p_pos = self._f_word_pos_pos(c_word, c_pos, p_pos)
        f_p_word_p_pos_c_pos = self._f_word_pos_pos(p_word, p_pos, c_pos)
        f_p_pos_c_pos = self._f_pos_pos(p_pos, c_pos)

        return [f_p_word_p_pos, f_p_word, f_p_pos, f_c_word_c_pos, f_c_word, f_c_pos, f_c_word_c_pos_p_pos, f_p_word_p_pos_c_pos, f_p_pos_c_pos]


class ComplexFeatures:
    pass


if __name__ == '__main__':
    vocab_list = ['ofir', 'tomer', 'nadav', 'roy']
    pos_list = ['S', 'T']
    word_pos_pairs = [('ofir', 'S'), ('tomer', 'S'), ('nadav', 'T'), ('roy', 'T')]

    # validate word pos
    word_pos = WordPos(vocab_list, pos_list, word_pos_pairs)
    assert word_pos('ofir', 'S') == (0, 4)
    assert word_pos('tomer', 'S') == (1, 4)
    assert word_pos('nadav', 'T') == (2, 4)
    assert word_pos('roy', 'T') == (3, 4)
    assert word_pos('tomer', 'T') == (-1, 4)

    # validate word
    word = Word(vocab_list, pos_list, word_pos_pairs)
    assert word('ofir') == (0, 4)
    assert word('tomer') == (1,4)
    assert word('nadav') == (2,4)
    assert word('roy') == (3,4)
    assert word('test') == (-1, 4)

    # validate pos
    pos = Pos(vocab_list, pos_list, word_pos_pairs)
    assert pos('S') == (0,2)
    assert pos('T') == (1,2)
    assert pos('F') == (-1,2)

    # validate word pos pos
    word_pos_pos = WordPosPos(vocab_list, pos_list, word_pos_pairs)
    assert word_pos_pos('ofir', 'S', 'S') == (0, 8)
    assert word_pos_pos('tomer', 'S', 'S') == (1,8)
    assert word_pos_pos('nadav', 'T', 'S') == (2,8)
    assert word_pos_pos('roy', 'T','S') == (3,8)
    assert word_pos_pos('ofir', 'S', 'T') == (4, 8)
    assert word_pos_pos('tomer', 'S', 'T') == (5, 8)
    assert word_pos_pos('nadav', 'T', 'T') == (6, 8)
    assert word_pos_pos('roy', 'T','T') == (7, 8)
    assert word_pos_pos('test', 'S', 'S') == (-1, 8)
    assert word_pos_pos('roy', 'F', 'S') == (-1, 8)
    assert word_pos_pos('roy', 'S', 'F') == (-1, 8)

    # validate pos pos
    pos_pos = PosPos(vocab_list, pos_list, word_pos_pairs)
    assert pos_pos('S', 'S') == (0, 4)
    assert pos_pos('T', 'S') == (1,4)
    assert pos_pos('S', 'T') == (2, 4)
    assert pos_pos('T', 'T') == (3, 4)

    # validate basic features
    basic = BasicFeatures(vocab_list, pos_list, word_pos_pairs)
    sentence = Sentence(['alejandro'],['S'])
    assert basic(0, 1, sentence) == [(-1, 4), (-1, 4), (-1, 2), (-1, 4), (-1, 4), (0, 2), (-1, 8), (-1,8), (-1, 4)]
    assert basic.features_len() == 4 + 4 + 2 + 4 + 4 + 2 + 8 + 8 + 4

    print('PASSED!')
