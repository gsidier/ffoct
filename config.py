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
		(.2, .3), 
		(.2, .4), 
		(.2, .5), 
		(.3, .3), 
		(.3, .4), 
		(.3, .5), 
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

def FeatureClass():
	
	return Texton(
		texton.filter_bank.function, 
		texton.filter_bank.params, 
		texton.cluster)


