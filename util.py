import cv
from pylab import imread, imshow
from PIL import Image
import numpy
from copy import copy
from numpy import median
from scipy.stats.mstats import mquantiles
from time import time
import os, sys

class Timer(object):
	def __init__(self, msg):
		self.msg = msg
	def __enter__(self):
		self.t0 = time()
		print self.msg, 
		sys.stdout.flush()
	def __exit__(self, *args):
		print "completed in %f sec" % (time() - self.t0)

def cvim2array(im):
	return numpy.array(im)[:,:,::-1]

def cv2grey(im):
	return numpy.array(im)[:,:,0].astype(float) / 256.

def cvimshow(im):
	imshow(cvim2array(im))

def loadmask(path):
	im = Image.open(path)
	im = im.convert(mode = 'L')
	m = numpy.array(im, dtype = bool)
	return m

def loadgrey16(path):
	im = Image.open(path)
	im = im.convert('I')
	m = numpy.array(im, dtype = numpy.uint16)
	return m

def samp16(path, x, y, w, h):
	im = Image.open(path)
	samp = im.crop((x, y, x + w, y + h))
	samp = samp.convert('I')
	m = numpy.array(samp, dtype = numpy.uint16)
	return m

def sampflt(path, x, y, w, h):
	m = samp16(path, x, y, w, h)
	m = m.astype(float)
	m /= (1 << 16)
	return m

def loadgrey8(path):
	m = loadgrey16(path)
	m /= 256
	m = m.astype(numpy.uint8)
	return m

def samp8(path, x, y, w, h):
	im = Image.open(path)
	samp = im.crop((x, y, x + w, y + h))
	samp = samp.convert('I')
	m = numpy.array(samp, dtype = numpy.uint8)
	return m

def distrib(X, q):
	q = numpy.array(q)
	x = mquantiles(X, q, 0, 1)
	dx = x[1:] - x[:-1]
	i, = numpy.where(abs(dx) > 1e-10)
	i = [0] + list(i + 1)
	qi = q[i]
	dq = qi[1:] - qi[:-1]
	xi = x[i]
	dx = xi[1:] - xi[:-1]
	m = numpy.array([median(X[(X >= a) & (X < b)])
		for (a, b) in zip(xi[:-1], xi[1:])])
	p = dq / dx
	return (m, p)

def histogram(values, bins):
	x = numpy.array(values)
	if any(x[1:] < x[:-1]):
		x = copy(x)
		x.sort()
	
	xmin = x[0]
	xmax = x[-1]
	n = len(x)
	# discretize in index space
	idx = linspace(0, n, bins + 1)
	i = numpy.floor(idx)
	xi = x[i]
	xmin = xi[:-1]
	xmax = xi[1:]
	xmid = (xmin + xmax) * .5
	
	c = i[1:] - i[:-1]
	c = c[1:]
	
	a = xmid[:-1]
	b = xmid[1:]
	return (xmid, c / b - a / n)

