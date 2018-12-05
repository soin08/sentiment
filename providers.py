import gensim
import json


class Word2VecProvider:
    def __init__(self):
        self.word2vec = None
        self.dimensions = 0

    def load(self, path_to_word2vec):
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(path_to_word2vec, binary=False)
        self.word2vec.init_sims(replace=True)
        self.dimensions = self.word2vec.vector_size
        print(type(self.word2vec))

    def get_vector(self, word):
        if word not in self.word2vec.vocab:
            return None
        return self.word2vec.syn0norm[self.word2vec.vocab[word].index]

    def get_similarity(self, word1, word2):
        if word1 not in self.word2vec.vocab or word2 not in self.word2vec.vocab:
            return None
        return self.word2vec.similarity(word1, word2)


class EmojiProvider:
    def __init__(self):
        self._emoji = None

    def load(self, file_path):
        with open(file_path, 'r') as f:
            self._emoji = json.load(f)

    @property
    def emoji(self):
        return self._emoji["positive"] + self._emoji["negative"]

    def is_positive_emoji(self, word):
        return word in self._emoji['positive']

    def is_negative_emoji(self, word):
        return word in self._emoji['negative']

    def count_emoji(self, text):
        count_pos = 0
        count_neg = 0
        for word in text.split(' '):
            if self.is_positive_emoji(word.strip()):
                count_pos += 1
            elif self.is_negative_emoji(word.strip()):
                count_neg += 1
        return count_pos, count_neg