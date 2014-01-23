import os, sys
from features import Texton

root = os.path.abspath(os.path.dirname(__file__))

master_dir = os.path.join(root, 'samples', 'master')
derived_dir = os.path.join(root, 'samples', 'derived')

from skimage.filter import gabor_filter
from sklearn.cluster import KMeans

def FeatureClass():
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
	thetas = linspace(0, 1, 8) * pi # real part of Gabor depends on theta mod pi
	
	filters = [ 
		{ 'frequency': f, 'theta': theta, 'bandwidth': b } for theta in thetas for (f, b) in freqs
	]
	
	cluster = KMeans(n_clusters = 32)
	
	return Texton(gabor_filter, filters, cluster)

