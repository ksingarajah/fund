import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob

class TwitterClient():
    def __init__(self):

        consumerKey = 'Df3PIQAqPPJSJHbCVWTn76Xou'
        consumerSecret = 'xa65sxdNyAxK3nz3k8y6Wkn2uDBgOFYtHsVG28gEXOWFT9l0fX'
        accessToken = '2231597743-1YVTwe1RNihat4kRhdhpza2vFLP1DSDxaG7Ijxs'
        accessTokenSecret = 's464p6YaXAP79SReP5E95OeRXqkYfxyMvTc6cMv3pcfzA'

        try:
            self.auth = OAuthHandler(consumerKey, consumerSecret)
            self.auth.set_access_token(accessToken, accessTokenSecret)
            self.api = tweepy.API(self.auth)
        except:
            print("Authentication FAILURE")

    def cleanTweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def getSentiment(self, tweet):
        info = TextBlob(self.cleanTweet(tweet))
        if info.sentiment.polarity > 0:
            return '+'
        elif info.sentiment.polarity == 0:
            return '%'
        else:
            return '-'

    def getTweets(self, query, count):
        listOfTweets = []
        try:
            queryResults = self.api.search(q = query, count = count)
            for tweet in queryResults:
                tweetInfo = {}
                tweetInfo['text'] = tweet.text
                tweetInfo['sentiment'] = self.getSentiment(tweet.text)
                if tweet.retweet_count > 0:
                    if tweetInfo not in listOfTweets:
                        listOfTweets.append(tweetInfo)
                else:
                    listOfTweets.append(tweetInfo)

            return listOfTweets
        except:
            print("Error")

def main():
    api = TwitterClient()
    tweets = api.getTweets(query = 'Google', count = 150)
    pos = [tweet for tweet in tweets if tweet['sentiment'] == '+']
    neu = [tweet for tweet in tweets if tweet['sentiment'] == '%']
    neg = [tweet for tweet in tweets if tweet['sentiment'] == '-']

    print "Percentage of tweets that are positive: {}".format(100*len(pos)/len(tweets))
    print "Percentage of tweets that are neutral: {}".format(100*len(neu)/len(tweets))
    print "Percentage of tweets that are negative: {}".format(100*len(neg)/len(tweets))

    print "\n\nFirst 10 Positive Tweets:"
    for tweet in pos[:10]:
        print tweet['text']

    print "\n\nFirst 10 Neutral Tweets:"
    for tweet in neu[:10]:
        print tweet['text']

    print "\n\nFirst 10 Negative Tweets:"
    for tweet in neg[:10]:
        print tweet['text']

if __name__ == "__main__":
    main()
