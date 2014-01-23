#!/usr/bin/env python
from sampset import SampleSet, Sample
import config
from samples import DataSet

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
	
	# 1. read patches test set
	# 2. compute features
	# 3. train classifier
	
	USAGE = "%prog [<options>] <trainingdata>"
	
	optp = OptionParser(usage = USAGE)
	
	PATCH_WIDTH = 10
	PATCH_HEIGHT = 10
	
	opts, args = optp.parse_args()
	
	try:
		TRAININGDATA_PATH, = args
	except:
		optp.print_usage()
		sys.exit(1)
	
	# 1. read patches test set
	
	samples = SampleSet(config)
	
	trainingdata_module = __import__(TRAININGDATA_PATH, fromlist = [True])
	trainingdata = trainingdata_module.TRAINING_DATA
	training_data = DataSet(trainingdata, samples)
	
	# 2. compute features
	
	FeatureClass = config.FeatureClass
	features = FeatureClass()
	
	features.calc(training_data)
	
