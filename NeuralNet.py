import pdb
import numpy

class NeuralNet:
	def __init__(self, netSpec):
		self.layers = list()
		for i in range(1, len(netSpec)):
			self.layers.append(numpy.random.rand(netSpec[i-1], netSpec[i]) * .1)
			print(self.layers[-1].shape)
			
	def sigmoid(x):
		return 1 / (1 + numpy.e ** -x) # This does not work...
	
	def dsigmoid(x):
		s = NeuralNet.sigmoid(x)
		return s * (1 - s)
	
	def run(self, data):
		a = list([data])
		for w in self.layers:
			i = numpy.dot(a[-1], w)
			a.append(NeuralNet.sigmoid(i))
		return a[-1]
	
	def train(self, data, alpha, it=3):
		for ds, y in data:
			a = list([ds])
			# Forward Feed
			for w in self.layers:
				i = numpy.dot(a[-1], w)
				a.append(NeuralNet.sigmoid(i))
			# Back Propagate
			deltas = list()
			init = list()
			for i in range(self.layers[-1].shape[1]):
				row = self.layers[-1].T[i]
				init.append((y[i] - a[-1][i]) * NeuralNet.dsigmoid(numpy.dot(a[-2], row)))
			deltas.append(numpy.array(init))
			for k in range(len(self.layers)-1, 0, -1):
				v = numpy.dot(deltas[-1], self.layers[k].T)
				injk = numpy.dot(a[k-1], self.layers[k-1])
				init = list()
				for j in range(self.layers[k-1].shape[1]):
					#row = self.layers[k-1].T[i]
					#injk = numpy.dot(a[k-1], row)
					init.append(NeuralNet.dsigmoid(injk[j]) * v[j])
				deltas.append(numpy.array(init))
			deltas.reverse()
			for k in range(len(self.layers)-1, -1, -1):
				for j in range(self.layers[k].shape[1]):
					#print(self.layers[k].T[j].shape)
				#	print(str(a[k].shape) + " " + str(k))
					#print(a[k].shape)
					update = self.layers[k].T[j] + alpha * a[k] * deltas[k][j]
					self.layers[k].T[j] = update
				
if __name__ == '__main__':
	nn = NeuralNet([6, 4, 5])
	print(nn.layers[1].shape)
