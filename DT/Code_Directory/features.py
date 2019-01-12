# !/usr/bin/env python


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
        return other_pos_idx * word_pos_idx + word_pos_idx, len(self._word_pos_pairs_idx) * len(self._pos_idx)


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
        return pos_idx * other_pos_idx + pos_idx, len(self._pos_idx)**2


class BasicFeatures:
    """basic features class"""

    def __init__(self, vocab_list, pos_list, word_pos_pairs):
        """init features"""
        self._f_word_pos = WordPos(vocab_list, pos_list, word_pos_pairs)
        self._f_word = Word(vocab_list, pos_list, word_pos_pairs)
        self._f_pos = Pos(vocab_list, pos_list, word_pos_pairs)
        self._f_word_pos_pos = WordPosPos(vocab_list, pos_list, word_pos_pairs)
        self._f_pos_pos = PosPos(vocab_list, pos_list, word_pos_pairs)

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
        f_p_word_c_word_c_pos = self._f_word_pos_pos(c_word, c_pos, p_pos)
        f_p_word_p_pos_c_pos = self._f_word_pos_pos(p_word, p_pos, c_pos)
        f_p_pos_c_pos = self._f_pos_pos(p_pos, c_pos)

        return [f_p_word_p_pos, f_p_word, f_p_pos, f_c_word_c_pos, f_c_word, f_c_pos, f_p_word_c_word_c_pos, f_p_word_p_pos_c_pos, f_p_pos_c_pos]


class ComplexFeatures:
    pass


if __name__ == '__main__':
    print('PASSED!')
