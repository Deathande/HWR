import MyMath
import cmath
import random
import sys

class NeuralNet:
	def __init__(self, netSpec):
		random.seed(1)
		self.net = list()
		for i in range(1, len(netSpec)):
			newMat = list()
			for j in range(netSpec[i-1]):
				newMat.append([])
				for k in range(netSpec[i]):
					newMat[j].append(random.random() / 10)
			self.net.append(MyMath.matrix(newMat))
	
	def sigmoid(x):
		return MyMath.matrix.invElements(1 + cmath.e ** -x)
	
	def dsigmoid(x):
		s = sigmoid(x)
		return s * (1 - s)
	
	def train(self, data):
		for ds in data:
			# Forward feed
			output = MyMath.matrix([ds[0]]) # Data should be a vector
			for syn in self.net:
				output = output * syn
			# Back propagate
			error = list()
			for val in output:
				error.append(dsigmoid(val) * (ds[1] - val)
if __name__ == '__main__':
	nn = NeuralNet([6, 4, 5])
