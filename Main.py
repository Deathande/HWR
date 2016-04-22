import csv
import numpy as np
import random
import NeuralNet
import sys
import pdb
import BubbleSort
import IO

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

def average(val):
	avg = 0
	#pdb.set_trace()
	for i in val:
		avg += i
	return avg / len(val)

def rate(nn, datas):
	averages = list()
	for data, y in datas:
		out = nn.run(data)
		averages.append(average(abs(y - out)))
	return average(averages)

if __name__ == '__main__':
	"""
	training = getData('rec/optdigits_train.txt', True)
	testing = getData('rec/optdigits_test.txt', True)
	GeneticAlg.run(training, testing, 10, 10)
	"""
	training = IO.getData('rec/optdigits_train.txt', True)
	n = NeuralNet.NeuralNet(64, 10, [20])
	for i in n.layers:
		print(i.shape)
	n.train(training, .01, 80)
	"""
	test = getData('rec/optdigits_test.txt', True)
	print(rate(n, test))
	for data, y in test:
		out = n.run(data)
		for i in range(len(out)):
			sys.stdout.write(str(i) + ": ")
			sys.stdout.write(str(out[i]))
			print()
		print("found:" + str(getHighest(out)))
		print("actual: " + str(getHighest(y)))
		print("Average difference: " + str(average(abs(y - out))))
		print()
		"""
