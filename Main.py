import csv
import numpy as np
import random
import NeuralNet
import sys

def vectorize(x):
	vect = np.zeros(10)
	vect[int(x)] = 1
	return vect

# Read data and format for the neural network
# train flag specifies if data being read is
# training data with correct answers appended
def getData(d, train=False):
	data = list()
	f = open (d, 'r')
	reader = csv.reader(f)
	if train:
		for row in reader:
			row = [float(x) for x in row]
			rd = row[0:len(row)-1]
			ans = row[-1]
			data.append((np.array(rd), vectorize(ans)))
	else:
		for row in reader:
			data.append(np.array([float(x) for x in row]))
	f.close()
	return data

def getHighest(data):
	m = 0
	index = 0
	for i in range(len(data)):
		if (data[i] > m):
			m = data[i]
			index = i
	return index

def percentDiff(exp, theory):
	return abs(theory - exp) / theory 

if __name__ == '__main__':
	training = getData('optdigits_train.txt', True)
	structure = [len(training[0][0]), 10]
	#structure = [len(training[0][0]), len(training[0][0]) - 4, 10]
	n = NeuralNet.NeuralNet(64, 10)
	n.train(training, .01, 50)
	test = getData('optdigits_test.txt', True)
	for data, y in test:
		out = n.run(data)
		for i in range(len(out)):
			sys.stdout.write(str(i) + ": ")
			sys.stdout.write(str(out[i]))
			print()
		print("found:" + str(getHighest(out)))
		print("actual: " + str(getHighest(y)))
		print()
