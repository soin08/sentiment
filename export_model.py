from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion

from extractors import TweetFeatureExtractor, ItemSelector, ArrayTransposer, Stacker
from providers import Word2VecProvider, EmojiProvider

import pandas as pd

import pickle

word2vec_provider = Word2VecProvider()
word2vec_provider.load('./data/glove.twitter.27B.200d.txt')

emoji_provider = EmojiProvider()
emoji_provider.load('./data/emoji.json')

feature_extractor = TweetFeatureExtractor(word2vec_provider=word2vec_provider, emoji_provider=emoji_provider)


classifier = Pipeline(
    [('features', feature_extractor),
        ('union', FeatureUnion(
          transformer_list=[
              ('words', Pipeline([
                  ('selector', ItemSelector(key='clean_text')),
                  ('tfidf', TfidfVectorizer(ngram_range=(1, 2), use_idf=True, norm='l2'))
              ])),
              ('pos_emoji_count', Pipeline([
                  ('selector', ItemSelector(key='pos_emoji_count')),
                  ('transposer', ArrayTransposer())
              ])),
              ('neg_emoji_count', Pipeline([
                  ('selector', ItemSelector(key='neg_emoji_count')),
                  ('transposer', ArrayTransposer())
              ])),
              ('uppercase_word_count', Pipeline([
                  ('selector', ItemSelector(key='uppercase_word_count')),
                  ('transposer', ArrayTransposer())
              ])),
              ('word2vec', Pipeline([
                  ('selector', ItemSelector(key='word2vec')),
                  ('stacker', Stacker())
              ])),
          ])),
        ('clf-sgd', SGDClassifier(loss='modified_huber', random_state=43))
     ])

df = pd.read_csv('./research/data/twitter_with_text.csv')
df = df[df.sentiment.isin(["positive", "negative"])]


def sentiment_encoder(v):
    if v == "positive":
        return 1
    if v == "negative":
        return -1
    return 0


df.rename(columns={'sentiment': 'label'}, inplace=True)
df["label"] = df.label.apply(sentiment_encoder)
# make equal # of positive and negative sentiment

i = -1


def mark_positive(label):
    global i
    if label == 1:
        i += 1
        return i
    return 0


df['positive_cnt'] = df.label.apply(mark_positive)
df = df[df.positive_cnt <= 2400]


X_train, X_test, y_train, y_test = train_test_split(df.text, df.label, test_size=0.33, random_state=43)
classifier.fit(X_train, y_train)


with open('./data/model.pickle', 'wb') as f:
    pickle.dump(classifier, f)

# api = twitter.Api(consumer_key='consumer_key',
#                  consumer_secret='consumer_secret',
#                  access_token_key='access_token',
#                  access_token_secret='access_token_secret')
