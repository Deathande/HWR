import NeuralNet
import IO
import numpy as np
import datetime
from multiprocessing import Pool

def __cross(l1, l2):
	pivot = np.random.randint(1, np.minimum(len(l1), len(l2)))
	ltemp1 = l1[pivot:]
	ltemp2 = l2[pivot:]
	l1 = l1[0:pivot]
	l2 = l2[0:pivot]
	l1.extend(ltemp2)
	l2.extend(ltemp1)
	return [l1, l2]

def __mix(l, x=0):
	newlist = list()
	half = int(len(l) / 2)
	clist = l[0:half]
	olength = len(l)
	while len(clist) > 1:
		print(np.floor(len(l) / 2))
		index = np.random.randint(0, len(clist))
		l1 = clist.pop(index)
		index = np.random.randint(0, len(clist))
		l2 = clist.pop(index)
		crossed = __cross(l1, l2)
		newlist.extend([crossed[0], crossed[1]])
	newlist.extend(l[len(newlist):])
	return newlist

def __fix(l):
	if len(l[1]) > l[-1]:
		while len(l[1]) > l[-1]:
			index = np.random.randint(0, len(l[1]))
			l[1].pop(index)
	if len(l[1]) < l[-1]:
		while len(l[1]) < l[-1]:
			if len(l[1]) != 0:
				index = np.random.randint(0, len(l[1]))
			else:
				index = 0
			l[1].insert(index, np.random.randint(3, 85))

def __mutate(l):
	pct = np.random.rand()
	num = int(np.round(len(l) * pct))
	for x in range(num):
		i = np.random.randint(0, len(l))
		j = np.random.randint(0, len(l[i]))
		pct2 = np.random.random()
		val = l[i][j] * pct2
		if np.random.randint(0,1) == 1:
			if isinstance(l[i][j], float):
				l[i][j] += val
			else:
				l[i][j] += int(np.round(val))
		else:
			if isinstance(l[i][j], float):
				l[i][j]-= val
			else:
				l[i][j] -= int(np.round(val))

def __getHighest(data, num=1):
	m = 0
	index = 0
	for i in range(len(data)):
		if (data[i] > m):
			m = data[i]
			index = i

def average(val):
	
	for i in val:
		avg += i
	return avg / len(val)

def rate(nn, datas):
	averages = list()
	for data, y in datas:
		out = nn.run(data)
		averages.append(average(abs(y-out)))
	return average(averages)

def sortB(params, nets, data):
	length = len(params)
	for i in range(length):
		for j in range(length - i - 1):
			if rate(nets[j], data) > rate(nets[j+1], data):
				tempp = params[j]
				tempnet = nets[j]
				params[j] = params[j+1]
				nets[j] = nets[j+1]
				params[j+1] = tempp
				nets[j+1] = tempnet
	return params, nets

def run(d1, d2, it, ss, threads=0):
	inputs = 64
	outputs = 10
	print("Running genetic algorithm for Neural Network with")
	print("genetic iterations: " + str(it))
	print("samples: " + str(ss))
	print("inputs: " + str(inputs))
	print("outputs: " + str(outputs))
	params = list()
	nets = list()
	dataPoints = list()
	for i in range(ss):
		a = np.random.rand()
		nhl = np.random.randint(0, 5)
		iteration = np.random.randint(3, 100)
		hl = list()
		for j in range(nhl):
			hl.append(np.random.randint(3, 85))
		params.append([a, hl, iteration, nhl])
	
	for val in range(it):
		for i in params:
			nets.append(NeuralNet.NeuralNet(inputs, outputs, i[1]))
		for i in range(len(nets)):
			nets[i].train(d1, params[i][0], params[i][2])
		params, nets = sortB(params, nets, d2)
		dataPoints.append(rate(nets[0], d2))
		params = __mix(params)
		__mutate(params)
		for i in range(ss):
			__fix(params[i])
	data = np.array(dataPoints)
	time = datetime.datetime.now()
	fn = str(time.day) + "-" + str(time.month) + "-" + str(time.day) + "-" + str(time.hour) + ":" + str(time.minute) + ":" + str(time.second)
	np.save(fn, data)
	
if __name__ == '__main__':
	train = IO.getData('rec/optdigits_train.txt', True)
	test = IO.getData('rec/optdigits_test.txt', True)
	run(train, test, 10, 10)
