from nltk import WordNetLemmatizer
from nltk.corpus import wordnet
import re
import enchant
from nltk.metrics import edit_distance
from nltk.stem import PorterStemmer


class AntonymReplacer:
    def replace(self, word, pos=None):
        antonyms = []
        for syn in wordnet.synsets(word, pos=pos):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms.append(antonym.name())
        if len(antonyms) > 0:
            return antonyms.pop(0)
        else:
            return None

    def replace_negations(self, sent):
        i, l = 0, len(sent)
        words = []
        while i < l:
            word = sent[i]
            if word == 'not' and i + 1 < l:
                ant = self.replace(sent[i + 1])
                if ant:
                    words.append(ant)
                    i += 2
                    continue
            words.append(word)
            i += 1
        return words
    

class PolarityReplacer:
    def __init__(self):
        self._keyword_set = {"isn't", "aren't", "wasn't", "weren't", "ain't","haven't","hasn't", 
                             "hadn't","won't", "wouldn't", "don't", "doesn't","didn't",                                                                      "can't","couldn't","shouldn't","mightn't", "mustn't", "never", 
                             "nothing", "nowhere", "noone", "none", "not"}
        
    def mark_negations(self, tweet_word_list):
        l = len(tweet_word_list)
        neg_count = [0] * l
        
        for neg in self._keyword_set:
            try:
                neg_ix = tweet_word_list.index(neg)
                for i in range(l):
                    if neg_ix <= i <= neg_ix + 4:
                        neg_count[i] += 1
            except ValueError:
                continue
                        
        for i in range(l):
            if neg_count[i] % 2 == 1:
                tweet_word_list[i] = 'NEG_' + tweet_word_list[i]
        
        return tweet_word_list
                

NEGATION_REPLACEMENT_PATTERNS = [
    (r'won\'t', 'will not'),
    (r'can\'t', 'cannot'),
    (r'i\'m', 'i am'),
    (r'ain\'t', 'is not'),
    (r'(\w+)\'ll', '\g<1> will'),
    (r'(\w+)n\'t', '\g<1> not'),
    (r'(\w+)\'ve', '\g<1> have'),
    (r'(\w+)\'s', '\g<1> is'),
    (r'(\w+)\'re', '\g<1> are'),
    (r'(\w+)\'d', '\g<1> would')
]


class RegexpReplacer:
    def __init__(self, patterns=NEGATION_REPLACEMENT_PATTERNS):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]

    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            s = re.sub(pattern, repl, s)
        return s


class RepeatReplacer:
    def __init__(self):
        self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl = r'\1\2\3'

    def _replace(self, word):
        if wordnet.synsets(word):
            return word
        repl_word = self.repeat_regexp.sub(self.repl, word)
        if repl_word != word:
            return self._replace(repl_word)
        else:
            return repl_word
        
    def replace(self, word_list):
        return [self._replace(word) for word in word_list]


class SpellingReplacer:
    def __init__(self, dict_name='en_US', max_dist=2):
        self.spell_dict = enchant.Dict(dict_name)
        self.max_dist = max_dist

    def replace(self, word):
        if self.spell_dict.check(word):
            return word
        suggestions = self.spell_dict.suggest(word)
        if suggestions and edit_distance(word, suggestions[0]) <= self.max_dist:
            return suggestions[0]
        else: 
            return word


class StemReplacer:
    def __init__(self):
        self._stemmer = PorterStemmer()

    def replace(self, word_list):
        return [self._stemmer.stem(word) for word in word_list]


class LemmaReplacer:
    def __init__(self):
        self._lemmatizer = WordNetLemmatizer()

    def replace(self, word_list):
        return [self._lemmatizer.lemmatize(word) for word in word_list]
