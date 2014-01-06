#!/usr/bin/env python
from sampset import SampleSet, Sample
import config

import numpy
import pylab
from math import ceil, sqrt
from time import time
from collections import Counter

from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn import cluster
from scipy import sparse
from scipy.sparse import csr_matrix

RESIZE_FACT = 1. / 20.
SAMP_WIDTH = 7
SAMP_HEIGHT = 7

N_COMP = 400

N_ATOMS = 2

class Timer(object):
	def __init__(self, msg):
		self.msg = msg
	def __enter__(self):
		self.t0 = time()
		print self.msg, 
		sys.stdout.flush()
	def __exit__(self, *args):
		print "completed in %f sec" % (time() - self.t0)

def reduce(master):
	samples = master.samples
	w, h = master.image.size
	res = samples.generate(master.path, 
		('to8bit',), 
		('resize', int(round(w * RESIZE_FACT)), int(round(h * RESIZE_FACT)), 'ANTIALIAS'))
	return res

def display(fit):
	n = min(400, len(fit.components_))
	rows = int(ceil(sqrt(n)))
	cols = int(ceil(n / rows))
	for i, comp in enumerate(fit.components_[:n]):
		pylab.subplot(rows, cols, i + 1)
		pylab.imshow(comp.reshape(SAMP_WIDTH, SAMP_HEIGHT), 
			cmap = pylab.cm.gray, interpolation = 'nearest')
		pylab.xticks(())
		pylab.yticks(())

def imshow(*args, **kwargs):
	if 'cmap' not in kwargs:
		kwargs['cmap'] = pylab.cm.gray
	if 'interpolation' not in kwargs:
		kwargs['interpolation'] = 'nearest'
	pylab.imshow(*args, **kwargs)

if __name__ == '__main__':
	import os, sys 
	
	paths = os.listdir(config.master_dir)
	
	masters = []
	patches = []
	sampleset = SampleSet(config)
	
	idx = [ 0 ]
	
	for path in paths:
		try:
			master = sampleset.generate(os.path.join(config.master_dir, path))
		except IOError: # not an image?
			print >> sys.stderr, "Cannot load '%s' - skipping" % path
			continue
		master = reduce(master)
		if not os.path.exists(master.path):
			master.save()
		masters.append(master.image)
		w, h = master.image.size
		im_patches = extract_patches_2d(numpy.array(master.image), (SAMP_WIDTH, SAMP_HEIGHT))
		"""
		for x in xrange(0, w, SAMP_WIDTH):
			for y in xrange(0, h, SAMP_HEIGHT):
				patch = master.image.crop((x, y, x + SAMP_WIDTH, y + SAMP_HEIGHT))
				patches.append(patch)
		"""
		patches.extend(im_patches)
		idx.append(len(patches))
	
	with Timer("Normalizing data ..."):
		data = list( numpy.array(patch, numpy.float32).flatten() for patch in patches )
		data = numpy.array(data)
		data /= 256.
		mean = numpy.mean(data, axis = 0)
		data -= mean
		std = numpy.std(data, axis = 0)
		data /= std
	with Timer("Fitting model ..."):
		cols = ceil(sqrt(N_COMP))
		rows = ceil(N_COMP / float(cols))
		model = MiniBatchDictionaryLearning(n_components = N_COMP, alpha = 1)
		fit = model.fit(data)
	with Timer("Display components ..."): 
		pylab.ion()
		pylab.show()
		display(fit)
	with Timer("Compute projection ..."):
		model.set_params(transform_algorithm = 'omp', transform_n_nonzero_coefs = N_ATOMS)
		# the intention is simply this:
		#	code = model.transform(data)
		# but we chunk it up and store it in a sparse matrix for efficiency
		code = []
		CHUNK = 1000
		for i in xrange(0, len(data), CHUNK):
			data_i = data[i:i + CHUNK]
			code_i = model.transform(data_i)
			code_i = csr_matrix(code_i)
			code.append(code_i)
		code = sparse.vstack(code)
		proj = code.dot(fit.components_)
	with Timer("Reconstruct images ..."):
		approxs = [ ]
		errs = [ ]
		for (master, i1, i2) in zip(masters, idx[:-1], idx[1:]):
			im_patches = proj[i1:i2]
			im_patches = im_patches.reshape(len(im_patches), SAMP_WIDTH, SAMP_HEIGHT)
			approx = reconstruct_from_patches_2d(im_patches, master.size[::-1])
			approx *= std.reshape(1, len(std))
			approx += mean.reshape(1, len(mean))
			approxs.append(approx)
			errs.append(approx - master)
	with Timer("Build distrib ..."):
		nz = code.nonzero()
		x = numpy.zeros((max(nz[0]) + 1, 2), dtype = int)
		x[:] = -1
		for i, j in zip(nz[0], nz[1]):
			if x[i, 0] == -1:
				x[i, 0] = j
			else:
				x[i, 1] = j
		for i, (a. b) in enumerate(x):
			x[i, :] = min(a, b), max(a, b)
		counts = Counter(map(tuple, x))
		c = numpy.array(counts.values(), dtype = float)
		p = c / sum(c)
		entropy = - numpy.sum(p * numpy.log2(p))
	print "entropy of non-zeros: %f ( = log2 %d for %d patches)" % (entropy, int(2 ** entropy), len(x))
	cond_distrib = numpy.zeros(N_COMP)
	for (i, j) in x:
		if i > 0 and j > 0:
			cond_distrib[i, j] += 1
			cond_distrib[j, i] += 1
	cond_distrib /= numpy.sum(cond_distrib, axis = 1).reshape((N_COMP, 1))

