import tweepy
from textblob import TextBlob
import csv
import math
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense

from keys import *

auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Pareteum')

sentCount = 0
sentTotal = 0

for tweet in public_tweets:
    sentCount += 1
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)
    sentTotal += analysis.sentiment.polarity

sentAvg = sentTotal / sentCount
print(sentAvg)

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

def predict_prices(filename, x):
	# Multilayer Perceptron to Predict International Airline Passengers (t+1, given t)
	# fix random seed for reproducibility
	numpy.random.seed(7)
	# load the dataset
	dataframe = pandas.read_csv(filename, usecols=[1], engine='python', parse_dates=True, infer_datetime_format=True)
	dataset = dataframe.values
	dataset = dataset.astype('float32')
	# split into train and test sets
	train_size = int(len(dataset) * 0.67)
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
	print(len(train), len(test))
	# reshape into X=t and Y=t+1
	look_back = 1
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)
	# create and fit Multilayer Perceptron model
	model = Sequential()
	model.add(Dense(8, input_dim=look_back, activation='relu'))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(trainX, trainY, epochs=200, batch_size=2, verbose=2)
	# Estimate model performance
	trainScore = model.evaluate(trainX, trainY, verbose=0)
	print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
	testScore = model.evaluate(testX, testY, verbose=0)
	print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
	# generate predictions for training
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)
	# shift train predictions for plotting
	trainPredictPlot = numpy.empty_like(dataset)
	trainPredictPlot[:, :] = numpy.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
	# shift test predictions for plotting
	testPredictPlot = numpy.empty_like(dataset)
	testPredictPlot[:, :] = numpy.nan
	testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
	# plot baseline and predictions
	plt.plot(dataset)
	plt.plot(trainPredictPlot)
	plt.plot(testPredictPlot)
	plt.show()

	return trainPredict, testPredict

predicted_price = predict_prices('TEUM.csv', 29)
print(predicted_price)