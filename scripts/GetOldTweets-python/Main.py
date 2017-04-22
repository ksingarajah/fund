import got
import csv
from textblob import TextBlob

def main():

	def getTweetSentiment(tw):
		info = TextBlob(tw.text)
		sentiment = info.sentiment.polarity
		infoList = [tw.date.strftime("%B %d, %Y"), sentiment, tw.text]
		return infoList

	def tweetOkay(t):
		for ch in t.text:
			try:
				ch.decode('ascii')
			except:
				return False
		return True

	#Get tweets by query search and bound dates (feb 18 2015 - now)
	q = '#OpenSesame'
	tweetCriteria = got.manager.TweetCriteria().setQuerySearch(q).setSince("2015-02-18").setUntil("2017-03-31")
	tweets = got.manager.TweetManager.getTweets(tweetCriteria)

	#output filename will be the query minus the hashtag
	filename = '{}'.format(q[1:]) + ".csv"

	with open(filename, 'wb') as csvfile:
		writeController = csv.writer(csvfile)
		for tweet in tweets:
			if tweetOkay(tweet):
				writeController.writerow(getTweetSentiment(tweet))

	# Example 1 - Get tweets by username
	#tweetCriteria = got.manager.TweetCriteria().setUsername('barackobama').setMaxTweets(1)
	#tweet = got.manager.TweetManager.getTweets(tweetCriteria)[0]

	#printTweet("### Example 1 - Get tweets by username [barackobama]", tweet)

	# Example 2 - Get tweets by query search
	#tweetCriteria = got.manager.TweetCriteria().setQuerySearch('europe refugees').setSince("2015-05-01").setUntil("2015-09-30").setMaxTweets(1)
	#tweet = got.manager.TweetManager.getTweets(tweetCriteria)[0]

	#printTweet("### Example 2 - Get tweets by query search [europe refugees]", tweet)

	# Example 3 - Get tweets by username and bound dates
	#tweetCriteria = got.manager.TweetCriteria().setUsername("barackobama").setSince("2015-09-10").setUntil("2015-09-12").setMaxTweets(1)
	#tweet = got.manager.TweetManager.getTweets(tweetCriteria)[0]

	#printTweet("### Example 3 - Get tweets by username and bound dates [barackobama, '2015-09-10', '2015-09-12']", tweet)

if __name__ == '__main__':
	main()
