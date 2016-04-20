
"""
Bubble Sort
Because who cares about efficiency?!
"""

def f(x):
	return x

def sort(l, nets, function=f):
	length = len(l)
	for i in range(length):
		for j in range(length - i - 1):
			if function(nets[j]) > function(nets[j+1]):
				temp = l[j]
				tempnet = nets[j]
				l[j] = l[j+1]
				nets[j] = nets[j+1]
				l[j+1] = temp
				nets[j+1] = tempnets

	return l

if __name__ == '__main__':
	from random import random
	from math import floor
	l = list()
	for i in range(5):
		l.append(floor(random() * 10))
	print(sort(l))
