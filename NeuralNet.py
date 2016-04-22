import pdb
import numpy

class NeuralNet:
	def __init__(self,inl, ol, netSpec=[]):
		self.runQuality = list()
		numpy.random.seed(1)
		self.layers = list()
		netSpec.append(ol)
		netSpec.insert(0, inl)
		print(range(1, len(netSpec)))
		for i in range(1, len(netSpec)):
			self.layers.append(numpy.random.rand(netSpec[i-1]+1, netSpec[i]) * .1)
	
	def sigmoid(x):
		return 1 / (1 + numpy.e ** -x) # This does not work...
	
	def dsigmoid(x):
		s = NeuralNet.sigmoid(x)
		return s * (1 - s)
	
	def run(self, data):
		a = list([data])
		for w in self.layers:
			a[-1] = numpy.append(a[-1], 1)
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
	
	def train(self, data, alpha, it=3):
		for iterate in range(it):
			avgError = list()
			for ds, y in data:
				a = list([ds])
				# Forward Feed
				for w in self.layers:
					a[-1] = numpy.append(a[-1], 1)
					i = numpy.dot(a[-1], w)
					a.append(NeuralNet.sigmoid(i))
				avgError.append(self.average(abs(y - a[-1])))
				#print(self.runQuality[-1])
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
						init.append(NeuralNet.dsigmoid(injk[j]) * v[j])
					deltas.append(numpy.array(init))
				deltas.reverse()
				for k in range(len(self.layers)-1, -1, -1):
					for j in range(self.layers[k].shape[1]):
						update = self.layers[k].T[j] + alpha * a[k] * deltas[k][j]
						self.layers[k].T[j] = update
			self.runQuality.append(self.average(avgError))
				
if __name__ == '__main__':
	nn = NeuralNet([6, 4, 5])
	print(nn.layers[1].shape)
