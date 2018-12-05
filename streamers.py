from bokeh.protocol import ProtocolError
from tweepy import OAuthHandler, Stream
from tweepy.streaming import StreamListener


class TwitterStreamer:
    def __init__(self, classifier, consumer_key, consumer_secret, access_token_key, access_token_secret):
        self._auth = OAuthHandler(consumer_key, consumer_secret)
        self._auth.set_access_token(access_token_key, access_token_secret)
        self._classifier = classifier
        self._stream = None

    def stream(self, keywords: list, listener: StreamListener):
        self._stream = Stream(self._auth, listener)
        try:
            self._stream.filter(track=keywords, languages=['en'])
        except ProtocolError:
            self.stream(keywords)





