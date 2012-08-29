import cv
from pylab import imshow
import numpy

def cvim2array(im):
	return numpy.array(im)[:,:,::-1]

def cvimshow(im):
	imshow(cvim2array(im))


