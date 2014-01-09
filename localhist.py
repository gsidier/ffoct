#!/usr/bin/env python
from sampset import SampleSet, Sample
import config

import numpy
import pylab
from math import ceil, sqrt
from time import time
from collections import Counter

from skimage.feature import local_binary_pattern

def cum_counts(feat, bins = None):
	"""
	Given an N1 x N2 x ... Nn array of integer features F which take values 0 to M - 1, 
	return  an (N1+1) x (N2+1) x ... (Nn+1) x M array C such that 
		
		C[i1, ..., in, a] = #{ (j1, .. jn), forall k, jk < ik, F[j1, .. jn] == a }
	
	"""
	bins = bins or numpy.max(feat) + 1
	counts = numpy.zeros((feat.shape + (bins,)))
	for b in xrange(bins):
		I = numpy.where(feat == b)
		I = I + ([b] * len(I[0]),)
		counts[I] = 1
	for axis in xrange(len(feat.shape)):
		counts = numpy.cumsum(counts, axis = axis)
	C = numpy.zeros([n + 1 for n in numpy.shape(counts)[:-1]] + [numpy.shape(counts)[-1]], counts.dtype)
	C[1:, 1:, :] = counts
	return C

def local_histogram(counts, x1, x2):
	i1, j1 = x1
	i2, j2 = x2
	return counts[i2, j2, :] + counts[i1, j1, :] - counts[i1, j2, :] - counts[i2, j1, :]
