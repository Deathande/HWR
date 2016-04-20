import csv
import numpy as np

def vectorize(x):
	vect = np.zeros(10)
	vect[int(x)] = 1
	return vect

def getData(d, train=False):
	data = list()
	f = open(d, 'r')
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
