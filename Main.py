import csv
import sys
import NeuralNet
import pdb
import IO
import matplotlib.pyplot as plt
import datetime
import numpy as np

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
		averages.append(abs(y[getHighest(out)] - out[getHighest(out)]))
	return average(averages)

if __name__ == '__main__':
	# Parse the command line parameters
	hl = []
	alpha = .005
	iterations = 50
	hls = ""
	"""
	for i in range(len(sys.argv)):
		if sys.argv[i] == '-nn':
			sys.argv.pop(i)
			load = True
			fn = sys.argv[i]
		else:
			iterations = int(sys.argv.pop(i))
	"""
	sys.argv.pop(0)
	iterations = int(sys.argv.pop(0))
	alpha = float(sys.argv.pop(0))
	for arg in sys.argv:
		hl.append(int(arg))
	
	print("Running Neural network with parameters:")
	print("iterations over training data: " + str(iterations))
	print("inputs: 64")
	print("outputs: 10")
	print("alpha: " + str(alpha))
	sys.stdout.write("Hidden: ")
	hls = str(hl) 
	info = str(iterations)+ "_" + str(alpha) + "_" + hls
	print(hls)
	training = IO.getData('rec/optdigits_train.txt', True)
	n = NeuralNet.NeuralNet(64, 10, hl)
	n.train(training, alpha, iterations)
	n.export("data/nets/" + info)
	test = getData('rec/optdigits_test.txt', True)
	#print(rate(n, test))
	plt.plot(n.num_correct)
	plt.savefig("data/graphs/"+ info + ".jpg")
	correct = 0
	dat = ''
	for data, y in test:
		out = n.run(data)
		correct += y[NeuralNet.NeuralNet.getHighest(out)]
		for i in range(len(out)):
			sys.stdout.write(str(i) + ": ")
			dat += str(i) + ": "
			sys.stdout.write(str(out[i]))
			dat += str(out[i]) + "\n"
			print()
		print("found:" + str(getHighest(out)))
		print("actual: " + str(getHighest(y)))
		print()
		dat += "found: " + str(getHighest(out)) + "\n"
		dat += "actual: " + str(getHighest(y)) + "\n"
	print("Percent correct: " + str(correct / len(test)))
	dat += "Percent correct: " + str(correct / len(test)) + "\n"
	log = open("data/raw/" + info + ".out", "w")
	log.write(dat)
