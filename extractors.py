from providers import Word2VecProvider, EmojiProvider
from replacers import RepeatReplacer, RegexpReplacer, PolarityReplacer, StemReplacer, NEGATION_REPLACEMENT_PATTERNS
from sklearn.base import TransformerMixin, BaseEstimator
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import numpy as np


class ArrayTransposer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return np.transpose(np.matrix(data))


class Stacker(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return np.stack(data)


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class TweetFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, word2vec_provider: Word2VecProvider, emoji_provider: EmojiProvider):
        self._emoji_provider = emoji_provider
        self._repeat_replacer = RepeatReplacer()
        self._polarity_replacer = PolarityReplacer()
        self._replacement_patterns = NEGATION_REPLACEMENT_PATTERNS
        self._replacement_patterns.extend([
            # remove urls
            (r'((www\.[^\s]+)|(https?://[^\s]+))', ''),
            # remove usernames
            (r'@[^\s]+', ''),
            # remove # from hashtags
            (r'#([^\s]+)', r'\1'),
            # leave only letters
            (r'[^a-zA-Z]+', ' '),
            # remove months
            (r'(\b\d{1,2}\D{0,3})?\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|' +
             r'aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|(nov|dec)(?:ember)?)\D?(\d{1,2}(st|nd|rd|th)?)?(([,.\-\/])' +
             r'\D?)?((19[7-9]\d|20\d{2})|\d{2})*', '')
        ])
        self._regexp_replacer = RegexpReplacer(self._replacement_patterns)
        self._stem_replacer = StemReplacer()
        self._word2vec_provider = word2vec_provider
        self._stopwords = stopwords.words('english')
        # drop negation words from stopwords
        self._stopwords.extend(['NEG_' + word for word in self._stopwords])
        self._stopwords.extend(["'nt", "st", "nd", "rd", "th", "rt"])
        self._stopwords.extend(self._emoji_provider.emoji)

    @classmethod
    def _count_with_func(cls, tweet, func):
        count = 0
        for word in tweet.split(' '):
            if func(word):
                count += 1
        return count

    @classmethod
    def _count_occurrences(cls, tweet, letter):
        count = 0
        for l in tweet:
            if l == letter:
                count += 1
        return count

    @classmethod
    def _count_uppercase_words(cls, tweet):
        return cls._count_with_func(tweet, lambda word: word == word.upper())

    @classmethod
    def count_exclamation(cls, tweet):
        return cls._count_occurrences(tweet, '!')

    @classmethod
    def count_question_marks(cls, tweet):
        return cls._count_occurrences(tweet, '!')

    def count_positive_emoji(self, tweet):
        return self._count_with_func(tweet, lambda word: self._emoji_provider.is_positive_emoji(word.strip()))

    def count_negative_emoji(self, tweet):
        return self._count_with_func(tweet, lambda word: self._emoji_provider.is_negative_emoji(word.strip()))

    def clean_tweet(self, tweet):
        tweet = tweet.lower()
        # transform html encoded symbols
        tweet = BeautifulSoup(tweet, 'lxml').get_text()
        tweet = self._regexp_replacer.replace(tweet)
        tweet = word_tokenize(tweet)
        # eg loooove -> love
        tweet = self._repeat_replacer.replace(tweet)
        # replace negations
        tweet = self._stem_replacer.replace(tweet)
        tweet = self._polarity_replacer.mark_negations(tweet)
        return " ".join([word for word in tweet if word not in self._stopwords]).strip()

    def get_avg_word_similarity(self, tweet, main_word):
            current_similarities = set()
            for word in tweet.split(' '):
                sim = self._word2vec_provider.get_similarity(main_word, word.lower())
                if sim is not None:
                    current_similarities.add(sim)

            if len(current_similarities) == 0:
                return

            if len(current_similarities) == 1:
                return current_similarities.pop()

            # return np.mean(zscore(list(current_similarities)))

            # if len(current_similarities) == 1:
            #    return current_similarities[0 ]
            current_similarities = list(current_similarities)

            max_sim = np.max(current_similarities)
            min_sim = np.min(current_similarities)
            # normalize to <0;1>
            return list(np.mean([((sim - min_sim) / (max_sim - min_sim)) for sim in current_similarities]))

    def get_word2vec_vector(self, tweet):
        current_word2vec = []
        for word in tweet.split(' '):
            vec = self._word2vec_provider.get_vector(word.lower())
            if vec is not None:
                current_word2vec.append(vec)

        if len(current_word2vec) == 0:
            return np.zeros(200)

        return np.array(current_word2vec).mean(axis=0)

    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        features = np.recarray(shape=(len(texts), ),
                               dtype=[('pos_emoji_count', float),
                                      ('neg_emoji_count', float),
                                      ('uppercase_word_count', float),
                                      ('exclamation_count', float),
                                      ('question_mark_count', float),
                                      ('clean_text', object),
                                      ('word2vec', np.ndarray)
                                      ])

        for i, text in enumerate(texts):
            features['pos_emoji_count'][i] = self.count_positive_emoji(text)
            features['neg_emoji_count'][i] = self.count_negative_emoji(text)
            features['uppercase_word_count'][i] = self._count_uppercase_words(text)
            features['exclamation_count'][i] = self.count_exclamation(text)
            features['question_mark_count'][i] = self.count_question_marks(text)
            features['clean_text'][i] = self.clean_tweet(text)
            features['word2vec'][i] = self.get_word2vec_vector(text)

        return features
