import cv
from pylab import imshow
import numpy

def cvim2array(im):
	return numpy.array(im)[:,:,::-1]

def cv2grey(im):
	return numpy.array(im)[:,:,0].astype(float) / 256.

def cvimshow(im):
	imshow(cvim2array(im))


