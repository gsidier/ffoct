#/usr/bin/env python

class Feature(object):
	
	def calc(self, dataset):
		raise NotImplementedError

class Texton(object):
	
	def __init__(self, filt, params, cluster):
		"""
		filt: a skimage-style filter function
		params: [ { param: value } ]
		cluster: a sklearn-style cluster object
		"""
		self.filter = filt
		self.params = params
		self.cluster = cluster
	
	def calc(self, dataset):
		n = len(self.params)
		X = [ ]
		for master_path, masks in dataset.data.iteritems():
			master = dataset.samples.generate(master_path)
			w, h = master.image.size
			features = list(self.filter(master.image, ** params) for params in self.params)
			features = numpy.array(features)
			features = features.reshape(n, w * h)
			features = features.T
			X.append(features)
		X = numpy.vstack(X)
		self.cluster.fit(X)
		
