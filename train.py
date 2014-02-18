#!/usr/bin/env python
from sampset import SampleSet, Sample
import config
from samples import DataSet
from util import Timer

import os, sys
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

if __name__ == '__main__':

	from optparse import OptionParser
	import pickle
	
	# 1. load image data
	# 2. load pixel features (pixel -> texton)
	# 3. generate samples
	# 4. generate sample features: sample -> histogram of pixel features
	# 5. train classifier on subset of samples
	# 6. save classifier
	
	USAGE = "%prog [<options>] <trainingdata> <textons>"
	
	optp = OptionParser(usage = USAGE)
	optp.add_option('-r', '--restart')
	
	PATCH_WIDTH = 40
	PATCH_HEIGHT = 40
	
	opts, args = optp.parse_args()
	
	try:
		TRAININGDATA_PATH, TEXTON_PATH, = args
	except:
		optp.print_usage()
		sys.exit(1)
	
	steps = dict([ (lbl, i) for (i, lbl) in enumerate([
		'LOAD_DATA',
		'CALC_FEATURES',
	])])
	restart = opts.restart
	
	# 1. read patches test set
	if (not restart) or steps[restart] <= steps['LOAD_DATA']:
		with Timer("Load data ... "):
			
			samples = SampleSet(config)
			
			trainingdata_module = __import__(TRAININGDATA_PATH, fromlist = [True])
			trainingdata = trainingdata_module.TRAINING_DATA
			training_data = DataSet(trainingdata, samples)
	
	# 2. load pixel features
	if (not restart) or steps[restart] <= steps['LOAD_FEATURES']:
		with Timer("Load features ... "):
			with file(TEXTON_PATH, 'r') as texton_file:
				centroids, labels = pickle.load(texton_file)
				
