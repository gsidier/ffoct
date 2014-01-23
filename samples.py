#!/usr/bin/env python
from sampset import SampleSet, Sample
import config
from util import loadmask

import numpy
from PIL import Image

def find_samples(im, mask, w, h, dw = None, dh = None):
	if dw is None:
		dw = w
	if dh is None:
		dh = h
	W, H = im.shape[:2]
	found = [ ]
	for i in xrange(0, W - w + 1, dw):
		for j in xrange(0, H - h + 1, dh):
			if numpy.sum(mask[i:i+h, j:j+w]) == w * h:
				yield({'y': i, 'h': h, 'x': j, 'w': w})

class DataSet(object):
	
	def __init__(self, data, samples):
		"""
		data: a dict {
				master_path: { 
					label: mask_path
				}
			}
		samples: a SampleSet instance
		"""
		self.data = data
		self.samples = samples
	
	def iterpatches(self, label, w, h, dw = None, dh = None):
		for master_path, masks in self.data.iteritems():
			if label in mask_path:
				master = self.samples.generate(master_path)
				mask = self.samples.generate(mask_path)
				for patch in find_samples(master, mask, w, h, dw, dh):
					yield patch
	
	def labels(self):
		return set(label for d in self.data.values() for label in d.keys())

if __name__ == '__main__':
	
	import os, sys, traceback
	
	USAGE = "%(prog)s <master> <mask> <width> <height>" % {
		'prog': os.path.basename(sys.argv[0]) 
	}
	
	try:
		master_path, mask_path, w, h = sys.argv[1:]
		w = int(w)
		h = int(h)
	except:
		traceback.print_exc()
		print USAGE
		sys.exit(1)
	
	master = Image.open(master_path)
	
	mask = loadmask(mask_path)
	
	for samp in find_samples(master, mask, w, h):
		print samp

