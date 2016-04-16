import numpy

class NeuralNet:
	def __init__(self, netSpec):
		self.layers = list()
		for i in range(1, len(netSpec)):
			self.layers.append(numpy.random.rand(netSpec[i-1], netSpec[i]) * .1)
			
	def sigmoid(x):
		return 1 / (1 + numpy.e ** -x) # This does not work...
	
	def dsigmoid(x):
		s = NeuralNet.sigmoid(x)
		return s * (1 - s)
	
	def train(self, data):
		for ds, y in data:
			a = list([ds])
			# Forward Feed
			for w in self.layers:
				i = numpy.dot(a[-1], w)
				a.append(NeuralNet.sigmoid(i))
			# Back Propagate
			init = list()
			for i in range(self.layers[-1].shape[1]):
				row = self.layers[-1].T[i]
				init.append((y[i] - a[-1][i]) * NeuralNet.dsigmoid(numpy.dot(a[-2], row[i])))
			deltas = [numpy.array(init)]
			l = self.layers
			l.pop()
			l.reverse()
			ta = a
			ta.pop()
			ta.reverse()
			for w in l:
				pass
			exit(1)
if __name__ == '__main__':
	nn = NeuralNet([6, 4, 5])
	print(nn.layers[1].shape)
