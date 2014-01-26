#/usr/bin/env python

import numpy
from util import Timer

class Feature(object):
	
	def calc(self, dataset):
		raise NotImplementedError

