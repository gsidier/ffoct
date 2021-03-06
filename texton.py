#/usr/bin/env python

import numpy
from util import Timer

class Texton(object):
	
	def __init__(self, filt, params, cluster):
		"""
		filt: a skimage-style filter function
		params: [ { param: value } ]
		cluster: a function features -> cluster_centers, labels
			where features: [ W x H x Ndim ] float arrays
			      cluster_centers: Nclust x Ndim matrix
			      labels: [ W x H ] int matrices
		"""
		self.filter = filt
		self.params = params
		self.cluster = cluster
	
	def _calc_features(self, dataset):
		self.dataset = dataset
		n = len(self.params)
		self.features = [ ]
		self.masters = [ dataset.samples.generate(master_path) for master_path in dataset.data ]
		for master in self.masters:
			w, h = master.image.size
			with Timer("calc features .. "):
				f = numpy.array(list(self.filter(master.image, ** params) for params in self.params))
				f = f.transpose((1, 2, 0))
				self.features.append(f)
	
	def _cluster(self):
		with Timer("cluster ... "):
			self.textons = self.cluster(self.features)
	
	def calc(self, dataset):
		self._calc_features(dataset)
		self._cluster()
	
if __name__ == '__main__':
	
	import os, sys
	from optparse import OptionParser
	import pickle
	
	from sampset import SampleSet
	import config
	from samples import DataSet
	
	# 1. read patches test set
	# 2. compute features
	# 3. train classifier
	
	USAGE = "%prog [<options>] <trainingdata> <texton output file>"
	
	optp = OptionParser(usage = USAGE)
	optp.add_option('-r', '--restart')
	optp.add_option('-f', '--features-file')
	
	opts, args = optp.parse_args()
	
	try:
		TRAININGDATA_PATH, TEXTON_OUT_PATH, = args
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
		
	# 2. compute features
	if (not restart) or steps[restart] <= steps['CALC_FEATURES']:
		with Timer("Calc features ... "):
			textons = Texton(
				config.texton.filter_bank.function, 
				config.texton.filter_bank.params,
				config.texton.quantize)
			
			textons._calc_features(training_data)
		
		if opts.features_file is not None:
			with Timer("Write features ... "):
				with file(opts.features_file, 'w') as features_out: 
					pickle.dump(textons.features, features_out)
			
		with Timer("Cluster textons ... "):
			textons._cluster()
		
		with Timer("Write textons ... "):
			with file(TEXTON_OUT_PATH, 'w') as textons_out:
				pickle.dump(textons.textons, textons_out)

