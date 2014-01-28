import os, sys
from texton import Texton

root = os.path.abspath(os.path.dirname(__file__))

cache_dir = os.path.join(root, 'cache')

master_dir = os.path.join(root, 'samples', 'master')
derived_dir = os.path.join(root, 'samples', 'derived')

import numpy
from numpy import arange
from skimage.filter import gabor_filter
from sklearn.cluster import KMeans

class gabor:
	from math import pi
	from numpy import linspace

	freqs = [ 
		(.1, 1.), 
#		(.2, .3), 
#		(.2, .4), 
		(.2, .5), 
		(.3, .3), 
#		(.3, .4), 
#		(.3, .5), 
		(.5, .5), 
	]
	
	NUM_ANGLES = 8
	
	thetas = arange(float(NUM_ANGLES)) / NUM_ANGLES * pi # real part of Gabor depends on theta mod pi

	params = [ 
		{ 'frequency': f, 'theta': theta, 'bandwidth': b } 
		for theta in thetas 
		for (f, b) in freqs
	]
	function = staticmethod(lambda *args, ** kwargs: gabor_filter(* args, ** kwargs)[0])
	
class texton:
	filter_bank = gabor
	clusters = 32
	cluster = KMeans(n_clusters = clusters)
	
	@staticmethod
	def kmeans_quantize(features):
		# features: [ image_features ]
		#     image_features: w x h x ndim 
		#
		# X = Nsamp x Ndim array from features
		_, _, ndim = numpy.shape(features[0])
		X = features
		X = [ f.reshape(f.shape[0] * f.shape[1], ndim) for f in X ]
		X = numpy.vstack(X)
		# fit centroids
		kmeans = KMeans(n_clusters = clusters)
		kmeans.fit(X)
		# reshape data
		labels = kmeans.labels_
		I = [0] + cumsum([ f.shape[0] * f.shape[1] for f in features ])
		labels = [ labels[i1:i2] for (i1, i2) in zip(I[:-1], I[1:]) ]
		labels = [ lbl.reshape(f.shape[0], f.shape[1]) for (lbl, f) in zip(labels, features) ]
		centroids = kmeans.cluster_centers_
		return centroids, labels
	
	@staticmethod
	def max_response_quantize(features):
		_, _, ndim = numpy.shape(features[0])
		labels = [ numpy.argmax(f, 2) for f in features ]
		centroids = numpy.eye(ndim)
		return centroids, labels
	
	quantize = max_response_quantize
	
def FeatureClass():
	
	return Texton(
		texton.filter_bank.function, 
		texton.filter_bank.params, 
		texton.quantize)


