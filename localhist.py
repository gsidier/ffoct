#!/usr/bin/env python
from sampset import SampleSet, Sample
import config
from ffoct import reduce_image, Timer

import numpy
import pylab
from math import ceil, sqrt
from time import time
from collections import Counter
import gc

from skimage.feature import local_binary_pattern

def cum_counts(feat, bins = None):
	"""
	Given an N1 x N2 x ... Nn array of integer features F which take values 0 to M - 1, 
	return  an (N1+1) x (N2+1) x ... (Nn+1) x M array C such that 
		
		C[i1, ..., in, a] = #{ (j1, .. jn), forall k, jk < ik, F[j1, .. jn] == a }
	
	"""
	bins = bins or numpy.max(feat) + 1
	counts = numpy.zeros(((bins,) + feat.shape), dtype = numpy.int32)
	print feat.shape, bins, numpy.product(feat.shape) * bins
	with Timer("count"):
		for b in xrange(bins):
			counts[b] = feat == b
			gc.collect()
	with Timer("cumsum"):
		for axis in xrange(len(feat.shape)):
			numpy.cumsum(counts, axis = axis + 1, out = counts)
	with Timer("copy"):
		C = numpy.zeros([numpy.shape(counts)[0]] + [n + 1 for n in numpy.shape(counts)[1:]], 
			dtype = numpy.float32)
		C[:, 1:, 1:] = counts
		del counts
		gc.collect()
	return C

def local_histogram(counts, x1, x2):
	i1, j1 = x1
	i2, j2 = x2
	hist = counts[:, i2, j2] + counts[:, i1, j1] - counts[:, i1, j2] - counts[:, i2, j1]
	hist /= hist.sum()
	return hist

def split(counts, thresh, maxsize, minsize):
	w, h = counts.shape
	w -= 1
	h -= 1
	def chi2(h1, h2):
		return (.5 * (h1 - h2)**2 / (h1 + h2)).sum()
	
	def rec(x1, y1, x2, y2):
		l = max((x2 - x1) / 2, (y2 - y1) / 2)
		assert x2 - x1 >= l # todo handle this case
		assert y2 - y1 >= l # todo handle this case
		regions = tuple((x, y, min(x + l, x2 - 1), min(y + l, y2 - 1)) 
			for y in xrange(0, h - 1, maxsize) 
			for x in xrange(0, w - 1, maxsize))
		h0, h1, h2, h3 = list(local_histogram(counts, (x1_, y1_), (x2_, y2_)) for (x1_, y1_, x2_, y2_) in regions)
		d01 = chi2(h0, h1)
		d02 = chi2(h0, h2)
		d03 = chi2(h0, h3)
		d12 = chi2(h1, h2)
		d13 = chi2(h1, h3)
		d23 = chi2(h2, h3)
		dmin = min(d01, d02, d03, d12, d13, d23)
		dmax = max(d01, d02, d03, d12, d13, d23)
		if dmax / dmin < thresh:
			return (x1, y1, x2, y2)
		else:
			reg1 = tuple(rec(x1_, y1_, x2_, y2_) for (x1_, y1_, x2_, y2_) in regions)
	
	return rec(0, 0, w, h)

if __name__ == '__main__':
	
	import os, sys 
	
	restart = None
	if sys.argv[1:]:
		restart, = sys.argv[1:]
	
	steps = dict([ (lbl, i) for (i, lbl) in enumerate([
		'LOAD_IMG',
	])])
	
	def require(step):
		return not restart or steps[restart] <= steps[step]
	
	if require('LOAD_IMG'):
		with Timer("Load images ..."):
			# 
			paths = os.listdir(config.master_dir)
			sampleset = SampleSet(config)
			masters = []
			patches = []
			idx = [ 0 ]
			#
			for path in paths:
				try:
					master = sampleset.generate(os.path.join(config.master_dir, path))
				except IOError: # not an image?
					print >> sys.stderr, "Cannot load '%s' - skipping" % path
					continue
				master = reduce_image(master)
				masters.append(master.image)
				if not os.path.exists(master.path):
					master.save()
				masters = masters[::-1]
	
	if require('CALC_FEATURES'):
		with Timer("Compute the features ..."):
			features = [ ]
			for master in masters:
				feat = local_binary_pattern(master, 8, 1)
				feat = feat.astype(int)
				features.append(feat)

	if require('FEATURE_HIST'):
		with Timer("Compute the local feature histograms ..."):
			cdfs = [ ]
			for (master, feat) in zip(masters, features):
				cdf = cum_counts(feat)
				cdfs.append(cdf)

