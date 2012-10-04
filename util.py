import cv
from pylab import imread, imshow
from PIL import Image
import numpy

def cvim2array(im):
	return numpy.array(im)[:,:,::-1]

def cv2grey(im):
	return numpy.array(im)[:,:,0].astype(float) / 256.

def cvimshow(im):
	imshow(cvim2array(im))

def loadmask(path):
	im = Image.open(path)
	im = im.convert(mode = 'L')
	s = im.tostring()
	w, h = im.size
	if len(s) != w * h:
		raise TypeError
	a = numpy.fromstring(s, bool)
	m = a.reshape((h, w))
	return m

def loadgrey16(path):
	im = Image.open(path)
	s = im.tostring()
	w, h = im.size
	if len(s) != w * h * 2:
		raise TypeError
	a = numpy.fromstring(s, numpy.uint16)
	m = a.reshape((h, w))
	return m
