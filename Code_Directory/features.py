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
        self._word_pos_5gram_pairs_idx = {(word[:5], pos): idx for idx, (word, pos) in enumerate(self._word_pos_pairs)}


class WordPos5gram(Feature):
    """word pos 5 gram feature class"""
    def __init__(self, vocab_list, pos_list, word_pos_pairs):
        """init"""
        super(WordPos5gram, self).__init__(vocab_list, pos_list, word_pos_pairs)

    def __call__(self, word, pos):
        """generate feature tuple"""
        return self._word_pos_5gram_pairs_idx.get((word[:5], pos), -1), len(self._word_pos_5gram_pairs_idx)


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
            return -1, len(self._pos_idx) ** 2
        return other_pos_idx * len(self._pos_idx) + pos_idx, len(self._pos_idx)**2


class PosPosPosPos(Feature):
    """pos pos pos pos feature class"""
    def __init__(self, vocab_list, pos_list, word_pos_pairs):
        """init"""
        super(PosPosPosPos, self).__init__(vocab_list, pos_list, word_pos_pairs)

    def __call__(self, pos1, pos2, pos3, pos4):
        """generate feature tuple"""
        pos1_idx = self._pos_idx.get(pos1, -1)
        pos2_idx = self._pos_idx.get(pos2, -1)
        pos3_idx = self._pos_idx.get(pos3, -1)
        pos4_idx = self._pos_idx.get(pos4, -1)

        if -1 in [pos1_idx, pos2_idx, pos3_idx, pos4_idx]:
            return -1, len(self._pos_idx) ** 4
        return  pos1_idx * (len(self._pos_idx) ** 3) + pos2_idx * (len(self._pos_idx) ** 2) + pos3_idx * len(self._pos_idx) + pos4_idx, len(self._pos_idx) ** 4


class Direction(Feature):
    """direction feature class"""
    def __init__(self, vocab_list, pos_list, word_pos_pairs):
        """init"""
        super(Direction, self).__init__(vocab_list, pos_list, word_pos_pairs)

    def __call__(self, h, m):
        """generate feature tuple"""
        if h < m: # right edge
            return 1, 2
        return 0, 2 # left edge


class Distance(Feature):
    """distance feature class"""
    def __init__(self, vocab_list, pos_list, word_pos_pairs, max_len):
        """init"""
        super(Distance, self).__init__(vocab_list, pos_list, word_pos_pairs)
        self._max_len = max_len

    def __call__(self, h, m): # h != m
        """generate feature tuple"""
        dist = abs(h - m)
        if dist < self._max_len:
            return dist, self._max_len
        return -1, self._max_len


class BetweenPos(Feature):
    """pos in between"""
    def __init__(self, vocab_list, pos_list, word_pos_pairs):
        """init"""
        super(BetweenPos, self).__init__(vocab_list, pos_list, word_pos_pairs)


    def __call__(self, h, m, sentence):
        """generate feature tuple"""
        between_pos_list = []
        for idx in range(min(h,m) + 1, max(h,m)):
            between_pos_list.append(sentence(idx)[1])
        ret = []
        for pos in self._pos_list:
            if pos in between_pos_list:
                ret.append((0, 1))
            else:
                ret.append((-1, 1))
        return ret



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
        for _, size in self(0 ,1, Sentence(['',''],['',''])):
            sum += size
        return sum


    def __call__(self, h, m, sentence):
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


class ComplexFeatures(BasicFeatures):
    """complex features class"""
    def __init__(self, vocab_list, pos_list, word_pos_pairs):
        """init features"""
        super(ComplexFeatures, self).__init__(vocab_list, pos_list, word_pos_pairs)
        self._f_pos_pos_pos_pos = PosPosPosPos(vocab_list, pos_list, word_pos_pairs)
        self._f_direction = Direction(vocab_list, pos_list, word_pos_pairs)
        self._f_distance = Distance(vocab_list, pos_list, word_pos_pairs, 60) # it's a kind of magic
        self._f_between_pos = BetweenPos(vocab_list, pos_list, word_pos_pairs)
        self._f_5gram = WordPos5gram(vocab_list, pos_list, word_pos_pairs)

    def __call__(self, h, m, sentence):
        """return list of all features"""
        basic_features = super(ComplexFeatures, self).__call__(h, m, sentence)

        p_word = sentence(h)[0]
        c_word = sentence(m)[0]
        p_pos = sentence(h)[1]
        c_pos = sentence(m)[1]

        if h == 0:
            p_pos_1 = None
        else:
            p_pos_1 = sentence(h - 1)[1]

        if m == 0:
            c_pos_1 = None
        else:
            c_pos_1 = sentence(m - 1)[1]

        if h == sentence.sentence_len - 1:
            p_pos1 = None
        else:
            p_pos1 = sentence(h + 1)[1]

        if m == sentence.sentence_len - 1:
            c_pos1 = None
        else:
            c_pos1 = sentence(m + 1)[1]


        p_pos_p_pos1_c_pos_1_c_pos = self._f_pos_pos_pos_pos(p_pos, p_pos1, c_pos_1, c_pos)
        p_pos_1_p_pos_c_pos_1_c_pos = self._f_pos_pos_pos_pos(p_pos_1, p_pos, c_pos_1, c_pos)
        p_pos_p_pos1_c_pos_c_pos1 = self._f_pos_pos_pos_pos(p_pos, p_pos1, c_pos, c_pos1)
        p_pos_1_p_pos_c_pos_c_pos1 = self._f_pos_pos_pos_pos(p_pos_1, p_pos, c_pos, c_pos1)

        p_c_dist = self._f_distance(h, m)
        p_c_direction = self._f_direction(h, m)

        p_pos_1_f = self._f_pos(p_pos_1)
        p_pos1_f = self._f_pos(p_pos1)
        c_pos_1_f = self._f_pos(c_pos_1)
        c_pos1_f = self._f_pos(c_pos1)

        return basic_features + [self._f_5gram(p_word, p_pos), self._f_5gram(c_word, c_pos), p_pos_1_f,p_pos1_f,c_pos_1_f, c_pos1_f, p_pos_p_pos1_c_pos_1_c_pos, p_pos_1_p_pos_c_pos_1_c_pos,
                                  p_pos_p_pos1_c_pos_c_pos1, p_pos_1_p_pos_c_pos_c_pos1, p_c_dist, p_c_direction] + self._f_between_pos(h, m, sentence)


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

    # validate pos pos pos pos
    pos_pos_pos_pos = PosPosPosPos(vocab_list, pos_list, word_pos_pairs)
    assert pos_pos_pos_pos('S', 'S', 'S', 'S') == (0, 16)
    assert pos_pos_pos_pos('S', 'S', 'S', 'T') == (1, 16)
    assert pos_pos_pos_pos('S', 'S', 'T', 'S') == (2, 16)
    assert pos_pos_pos_pos('S', 'S', 'T', 'T') == (3, 16)
    assert pos_pos_pos_pos('S', 'T', 'S', 'S') == (4, 16)
    assert pos_pos_pos_pos('S', 'T', 'S', 'T') == (5, 16)
    assert pos_pos_pos_pos('S', 'T', 'T', 'S') == (6, 16)
    assert pos_pos_pos_pos('S', 'T', 'T', 'T') == (7, 16)
    assert pos_pos_pos_pos('T', 'S', 'S', 'S') == (8, 16)
    assert pos_pos_pos_pos('T', 'S', 'S', 'T') == (9, 16)
    assert pos_pos_pos_pos('T', 'S', 'T', 'S') == (10, 16)
    assert pos_pos_pos_pos('T', 'S', 'T', 'T') == (11, 16)
    assert pos_pos_pos_pos('T', 'T', 'S', 'S') == (12, 16)
    assert pos_pos_pos_pos('T', 'T', 'S', 'T') == (13, 16)
    assert pos_pos_pos_pos('T', 'T', 'T', 'S') == (14, 16)
    assert pos_pos_pos_pos('T', 'T', 'T', 'T') == (15, 16)

    # validate direction
    direction = Direction(vocab_list, pos_list, word_pos_pairs)
    assert direction(0, 1) == (1, 2)
    assert direction(1, 0) == (0, 2)

    # validate distance
    dist = Distance(vocab_list, pos_list, word_pos_pairs, 5)
    assert dist(0, 1) == (1, 5)
    assert dist(1, 0) == (1, 5)
    assert dist(5, 9) == (4, 5)
    assert dist(9, 6) == (3, 5)
    assert dist(5, 10) == (-1, 5)

    # validate between pos
    between_pos = BetweenPos(vocab_list, pos_list, word_pos_pairs)
    sentence = Sentence(['ofir', 'roy','tomer'], ['S', 'T', 'S'])
    assert between_pos(1, 3, sentence) == [(-1, 1), (0, 1)]
    assert between_pos(0, 3, sentence) == [(0, 1), (0, 1)]

    print('PASSED!')
