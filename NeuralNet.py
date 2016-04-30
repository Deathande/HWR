import pdb
import sys
import numpy
import matplotlib as plt

class NeuralNet:
	def __init__(self,inl, ol, netSpec=[]):
		self.runQuality = list()
		#numpy.random.seed(1)
		self.layers = list()
		netSpec.append(ol)
		netSpec.insert(0, inl)
		for i in range(1, len(netSpec)):
			self.layers.append(numpy.array(numpy.random.rand(netSpec[i-1]+1, netSpec[i]), dtype=numpy.float64) * .1)
	
	def sigmoid(x):
		return 1 / (1 + numpy.e ** -x)
	
	def dsigmoid(x):
		s = NeuralNet.sigmoid(x)
		return s * (1 - s)
	
	def run(self, data):
		a = list([data])
		for w in self.layers:
			a[-1] = numpy.insert(a[-1], 0, 1)
			i = numpy.dot(a[-1], w)
			a.append(NeuralNet.sigmoid(i))
		return a[-1]
	
	def average(self, val):
		avg = 0
		for i in val:
			avg += i
		return avg / len(val)
	
	def export(self, fn):
		numpy.save(fn, self.layers)
	
	def load(self, nn):
		if isinstance(nn, str):
			self.layers = numpy.load(nn)
		else:
			self.layers = nn
	
	def getHighest(data):
		m = 0
		index = 0
		for i in range(len(data)):
			if data[i] > m:
				m = data[i]
				index = i
		return index
	
	def train(self, data, alpha, it=3):
		pct = len(data) * it
		print()
		print("percent complete: ")
		x = 0
		self.num_correct = []
		for iterate in range(it):
			correct = 0
			for ds, y in data:
				a = list([ds])
				inputs = list()
				# Forward Feed
				for w in self.layers:
					a[-1] = numpy.insert(a[-1], 0, 1)
					inputs.append(numpy.dot(a[-1], w))
					a.append(NeuralNet.sigmoid(inputs[-1]))
				correct += y[NeuralNet.getHighest(a[-1])]
				# Back Propagate
				dk = (y - a[-1]) * NeuralNet.dsigmoid(inputs[-1])
				deltas = [dk]
				for i in range(len(self.layers)-2, -1, -1):
					dk = NeuralNet.dsigmoid(inputs[i])
					selection = self.layers[i+1].T[:, 1:]#self.layers[i+1].shape[0]]
					dk = dk * numpy.dot(deltas[-1], selection)
					deltas.append(dk)
				#print(len(deltas))
				deltas.reverse()
				for k in range(len(self.layers)):
					for j in range(self.layers[k].shape[1]):
						self.layers[k][:,j] = self.layers[k][:,j] + (alpha * a[k] * deltas[k][j])
				x += 1
				"""
				print("1:")
				print(self.layers[0][0][0])
				print("2:")
				print(self.layers[1][2][2])
				"""
				sys.stdout.write(str(round(x / pct * 100)) + "%      \r")

			self.num_correct.append(correct / len(data))
				
if __name__ == '__main__':
	pass
