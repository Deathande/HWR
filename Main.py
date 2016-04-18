import csv
import numpy as np
import GeneticAlg
import random
import NeuralNet
import sys
import pdb
import BubbleSort

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

def cross(l1, l2):
	if len(l1) == len(l2):
		pivot = np.random.randint(0, len(l1))
		ltemp1 = l1[pivot:]
		ltemp2 = l2[pivot:]
		l1 = l1[0:pivot]
		l2 = l2[0:pivot]
		l1.extend(ltemp2)
		l2.extend(ltemp1)
	return [l1, l2]

def mutate(l, num):
	new = list()
	for i in range(num):
		index = np.randomint(0, len(l))
		member = l[index]
		index = np.random.randint(0, len(l))
		val = 0
	return new

def mix(l, x=0):
	newlist  = list()
	olength = len(l)
	hl = list()
	for val in l:
		hl.append(val[2])
	if (x == 0):
		mix(hl, 1)
	for i in range(floor(len(l) / 2)):
		index = np.random.randint(0, len(l))
		l1 = l.pop(index)
		index = np.random.randint(0, len(l))
		l2 = l.pop(index)
		newlist.extend(corss(l1, l2))

def genetic(it, threads=0):
	inputs = 64
	outputs = 10
	nets = list()
	params = list()
	training = getData('rec/optdigits_train.txt', True)
	testing = getData('rec/optdigits_test.txt', True)
	for i in range(10):
		alpha = np.random.rand()
		nhl = np.random.randint(0,5)
		iteration = np.random.randint(3, 100)
		hl = list()
		for j in range(nhl):
			hl.append(np.random.randint(1, 85))
		params.append((alpha, iteration, hl, nhl))
	
	"""
	for n in range(it):
		errors = list()
		nets = list()
		for a, it, hl in params:
			nets.append(NeuralNet.NeuralNet(inputs, outputs, hl))
		for i in range(len(nets)):
			nets[i].train(training, params[i][0], params[i][1])
		for i in range(nets):
			errors.append(rate(nets[i], testing))
		BubbleSort.sort(errors)
	"""
		
			
if __name__ == '__main__':
	training = getData('rec/optdigits_train.txt', True)
	testing = getData('rec/optdigits_test.txt', True)
	GeneticAlg.run(training, testing, 10, 10)
	"""
	training = getData('rec/optdigits_train.txt', True)
	n = NeuralNet.NeuralNet(64, 10)
	n.train(training, .01, 10)
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
		rint()
	"""
