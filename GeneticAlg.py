import NeuralNet
import numpy as np

def __mix(l, x=0):
	newlist = list()
	olenght = len(l)
	hl = list()
	if (x == 0):
		for val in l:
			hl.append(val[1])
		__mix(hl, 1)
		for i in range(np.floor(len(l) / 2)):
			index = np.random.randint(0, len(l))
			l1 = l.pop(index)
			index = np.random.rand(0, len(1))
			l2 = l.pop(index)
			newlist.extend(cross(l1, l2))

def run(d1, d2, it, ss, threads=0):
	inputs = 64
	outputs = 10
	params = list()
	for i in range(ss):
		a = np.random.rand()
		nhl = np.random.randint(0, 5)
		iteration = np.random.randint(3, 100)
		hl = list()
		for j in range(nhl):
			hl.append(np.random.randint(1, 85))
		params.append((a, hl, iteration, nhl))
	for i in range(ss):
		print(params[i])
	__mix(params)
