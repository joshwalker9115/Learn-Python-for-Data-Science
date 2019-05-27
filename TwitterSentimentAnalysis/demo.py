import tweepy
from textblob import TextBlob
import csv

from keys import *

auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Trump')

with open('test.csv', 'w') as SAFile:
    SAFile_writer = csv.writer(SAFile, delimiter=',')
    SAFile_writer.writerow(['Sentiment', 'Score', 'Tweet'])

    for tweet in public_tweets:
        analysis = TextBlob(tweet.text)
        if analysis.sentiment.polarity > 0:
            SAFile_writer.writerow(['Positive', analysis.sentiment, tweet.text])
        elif analysis.sentiment.polarity < 0:
            SAFile_writer.writerow(['Negative', analysis.sentiment, tweet.text])
        else:
            SAFile_writer.writerow(['Neutral', analysis.sentiment, tweet.text])