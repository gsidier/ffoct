#!/usr/bin/env python
from sampset import SampleSet, Sample
import config

RESIZE_FACT = 1. / 10.
SAMP_WIDTH = 10
SAMP_HEIGHT = 10

def reduce(master):
	w, h = master.image.size
	res = master
	res = res.to8bit()
	res = res.resize(round(w * RESIZE_FACT), round(h * RESIZE_FACT), 'ANTIALIAS')
	return res

if __name__ == '__main__':
	import os, sys 
	
	masters = os.listdir(config.master_dir)
	
	samples = []
	sampleset = SampleSet(config)
	
	for path in masters:
		try:
			master = sampleset.generate(os.path.join(config.master_dir, path))
		except IOError: # not an image?
			print >> sys.stderr, "Cannot load '%s' - skipping" % path
			continue
		master = reduce(master)
		master.save()
		w, h = master.image.size
		for i in xrange(0, w, SAMP_WIDTH):
			for j in xrange(0, h, SAMP_HEIGHT):
				samp = master.crop(i, j, SAMP_WIDTH, SAMP_HEIGHT)
				samp.save()
				samples.append(samp)
