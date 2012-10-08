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
	m = numpy.array(im, dtype = bool)
	return m

def loadgrey16(path):
	im = Image.open(path)
	im = im.convert('I')
	m = numpy.array(im, dtype = numpy.uint16)
	return m

def loadgrey8(path):
	m = loadgrey16(path)
	m /= 256
	m = m.astype(numpy.uint8)
	return m

