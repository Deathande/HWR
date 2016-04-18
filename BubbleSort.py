
"""
Bubble Sort
Because who cares about efficiency?
"""

def f(x):
	return x

def sort(l, function=f):
	length = len(l)
	for i in range(length):
		for j in range(length - i - 1):
			if function(l[j]) < function(l[j+1]):
				temp = l[j]
				l[j] = l[j+1]
				l[j+1] = temp

	return l

if __name__ == '__main__':
	from random import random
	from math import floor
	l = list()
	for i in range(5):
		l.append(floor(random() * 10))
	print(sort(l))
