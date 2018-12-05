import pickle
import json
import rethinkdb
import hashlib
from configparser import ConfigParser
from extractors import TweetFeatureExtractor
from providers import Word2VecProvider, EmojiProvider
from streamers import TwitterStreamer
from tweepy.streaming import StreamListener
from datetime import datetime, timedelta

with open('./data/model.pickle', 'rb') as f:
    tweet_classifier = pickle.load(f)


class TwitterStreamListener(StreamListener):
    def __init__(self, connection, keywords: list, exclude_keywords: list=[], strict_kw_check=True):
        super().__init__()
        self._snooze_time = timedelta(seconds=3)
        self._timestamp = datetime.now()
        self._keywords = keywords
        self._exclude_keywords = exclude_keywords
        self._strict_kw_check = strict_kw_check
        self._connection = connection
        emoji_provider = EmojiProvider()
        emoji_provider.load('./data/emoji.json')
        self._extractor = TweetFeatureExtractor(word2vec_provider=Word2VecProvider(),
                                                emoji_provider=emoji_provider)

    def _contains_keywords(self, text):
        text = text.lower()
        for kw in self._keywords:
            if kw not in text:
                return False

        for kw in self._exclude_keywords:
            if kw in text:
                return False
        return True

    def _get_hashtags(self, data):
        return [t['text'] for t in data['entities']['hashtags']]

    def _push_tweet(self, data, clean_text, label):
        payload = {
            'id': data['id'],
            'timestamp': data['timestamp_ms'],
            'text': data['text'],
            'clean_text': clean_text,
            'clean_text_hash': hashlib.md5(clean_text.encode('utf-8')).hexdigest(),
            'hashtags': self._get_hashtags(data),
            'predicted_label': label,
            'keywords': self._keywords
        }
        print(payload['text'])
        print(payload['predicted_label'])
        # rethinkdb.table('twitter').insert(payload).run(self._connection)

    def on_data(self, data):
        if datetime.now() - self._timestamp >= self._snooze_time:
            data = json.loads(data)
            if "text" in data.keys():
                if self._strict_kw_check and not self._contains_keywords(data['text']):
                    return
                clean_text = self._extractor.clean_tweet(data['text'])
                if not clean_text.strip() == "":
                    self._push_tweet(data, clean_text, int(tweet_classifier.predict([data['text']])[0]))
                    self._timestamp = datetime.now()

    def on_error(self, status):
        print(status)


if __name__ == '__main__':
    config = ConfigParser()
    config.read('./settings.ini')

    # connection = rethinkdb.connect(host=config['rethinkdb']['host'], port=config['rethinkdb']['port'])
    connection = None

    twitter_streamer = TwitterStreamer(classifier=None,
                                       consumer_key=config['twitter']['consumer_key'],
                                       consumer_secret=config['twitter']['consumer_secret'],
                                       access_token_key=config['twitter']['access_token_key'],
                                       access_token_secret=config['twitter']['access_token_secret'])

    stream_listener = TwitterStreamListener(connection=connection,
                                            keywords=['trump'],
                                            exclude_keywords=[],
                                            strict_kw_check=True)

    twitter_streamer.stream(keywords=['apple'], listener=stream_listener)

    connection.close()
