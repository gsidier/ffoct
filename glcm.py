import numpy
from numpy import log, sqrt, abs

def glcm(img, di, dj):
	w, h = img.shape
	
	counts = {}
	
	for i in xrange(h):
		i2 = i + di
		if i2 >= h or i2 < 0:
			continue
		for j in xrange(w):
			j2 = j + dj
			if j2 >= w or j2 < 0:
				continue
			
			a = img[i, j]
			b = img[i2, j2]
			counts[(a, b)] = counts.get((a, b), 0) + 1.
	
	n = float(sum(counts.values()))
	for (k, v) in counts.iteritems():
		counts[k] /= n
	return counts
	
def energy(P):
	return sum(p * p for p in P.values())

def entropy(P):
	return - sum(p * log(p) for p in P.values())
	
def contrast(P):
	return sum((i - j) * (i - j) * p for ((i, j), p) in P.iteritems())

def absval(P):
	return sum(abs(i - j) * p for ((i, j), p) in P.iteritems())

def homogeneity(P):
	return sum(p / (1 + abs(i - j)) for ((i, j), p) in P.iteritems())

def invdiff(P):
	return sum(p / (1 + (i - j) * (i - j)) for ((i, j), p) in P.iteritems())

def correlation(P):
	mui = sum(i * p for ((i, j), p) in P.iteritems())
	muj = sum(j * p for ((i, j), p) in P.iteritems())
	sigmai = sqrt(sum(i * i * p for ((i, j), p) in P.iteritems()) - mui * mui)
	sigmaj = sqrt(sum(j * j * p for ((i, j), p) in P.iteritems()) - muj * muj)
	return sum((i - mui) * (j - muj) * p / (sigmai * sigmaj) for ((i, j), p) in P.iteritems())

