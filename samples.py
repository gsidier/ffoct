FILE1 = 'img/11H00940A-10x7mm-0um.tif'
FILE2 = 'img/11H01173A-10x10mm-25.998801um.tif'

SAMPLES = [
	(FILE1, (1000, 1500), (1000, 1500), ('fibrous',)),
	(FILE1, (1000, 1500), (1500, 2000), ('fibrous',)),
	(FILE1, (1000, 1500), (2000, 2500), ('fibrous',)),
	(FILE1, (1000, 1500), (2500, 3000), ('fibrous',)),
	(FILE1, (1000, 1500), (3000, 3500), ('fibrous',)),
	(FILE1, (1000, 1500), (3000, 3500), ('fibrous',)),


	(FILE1, (1000, 1500), (3500, 4000), ('tissue', 'adipocytes')),
	(FILE1, (1000, 1500), (4000, 4500), ('tissue', 'adipocytes')),
	
	
	(FILE2, (1500, 2000), (1000, 1500), ('tissue',)),
	(FILE2, (1500, 2000), (1500, 2000), ('tissue',)),
	(FILE2, (1500, 2000), (2000, 2500), ('tissue',)),
	(FILE2, (1500, 2000), (2500, 3000), ('tissue',)),
]


import cv
from util import cvim2array, cv2grey
import numpy

imgs = {
#	FILE1: cv2grey(cv.LoadImageM(FILE1)),
#	FILE2: cv2grey(cv.LoadImageM(FILE2)),
}

def find_samples(im, mask, w, h):
	W, H = im.shape[:2]
	found = [ ]
	for i in xrange(0, W, w):
		for j in xrange(0, H, h):
			if numpy.sum(mask[i:i+h, j:j+w]) == w * h:
				found.append({'y': i, 'h': h, 'x': j, 'w': w})
		
	return found

