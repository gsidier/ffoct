#!/usr/bin/env python

"""
The purpose of this module is to effectively and safely handle transformations on a set of master sample images.

Given a set of master samples that live under a master/ dir, and will never be modified,
we create derived samples under derived/ which acts as a cache. If we request a derived sample
that already exists in the derived cache in the sense that an image obtained from the same 
sequence of transformations is saved there, then the saved copy is simply read.

"""

import os

from PIL import Image

class SampleSetConfig:
	
	master_dir = './master'
	derived_dir = './derived'
	
	def __init__(self, **kwargs):
		for k, v in kwargs.items():
			setattr(self, k, v)

class SampleSet(object):
	
	def __init__(self, config):
		self.config = config
	
	def derive_name(self, name, tag):
		parts = name.split('.')
		ident = parts[0]
		ext = parts[-1]
		tags = parts[1:-1]
		tags.append(tag)
		res = '.'.join([ident] + tags + [ext])
		return res
	
	def derive_path(self, path, tag):
		name = os.path.basename(path)
		name = self.derive_name(name, tag)
		path = os.path.join(self.config.derived_dir, name)
		return path
	
	def generate(self, base, *ops):
		# base is a cached version for a subset of the transformations.
		# i0 is the start index of the ops that need to be applied to it.
		
		# fix dir if necessary:
		dirname = os.path.dirname(base)
		if dirname == '':
			if '.' in base:
				base = os.path.join(self.config.derived_dir, base)
			else:
				base = os.path.join(self.config.master_dir, base)
			
		path = base
		
		i0 = 0
		for i, op_args in enumerate(ops):
			tag = '-'.join(map(str, op_args))
			path = self.derive_path(path, tag)
			if os.path.exists(path):
				base = path
				i0 = i + 1
		# base now contains the most helpful cached version.
		print "using base image '%s'" % base
		img = Image.open(base)
		samp = Sample(self, img, base)
		for op in ops[i0:]:
			samp = samp.transf(op[0], op[1:])
		return samp

class Sample(object):
	
	def __init__(self, samples, image, path):
		self.samples = samples
		self.image = image
		self.path = path
	
	def transf(self, op, args):
		tag = '-'.join(map(str, [op] + list(args)))
		path = self.samples.derive_path(self.path, tag)
		if os.path.exists(path):
			im = Image.open(path)
		else:
			func = getattr(self, '_' + op)
			im = func(*args)
		return Sample(self.samples, im, path)
	
	def _resize(self, w, h, filt):
		filtnum = getattr(Image, filt)
		return self.image.resize((w, h), filtnum)
		
	def resize(self, w, h, filt):
		return self.transf('resize', (int(w), int(h), filt))
	
	def _crop(self, x, y, w, h):
		return self.image.crop((x, y, x + w, y + h))
	
	def crop(self, x, y, w, h):
		return self.transf('crop', (int(x), int(y), int(w), int(h)))
	
	def _to8bit(self):
		return self.image.convert('I').point(lambda i: i * (1./256.)).convert('L')
	
	def to8bit(self):
		return self.transf('to8bit', ())
	
	def save(self, format = None, **options):
		if format is None:
			format = self.path.split('.')[-1]
		ext = '.' + format.lower()
		format = Image.EXTENSION[ext]
		dirname = os.path.dirname(self.path)
		name = os.path.basename(self.path)
		parts = name.split('.')
		parts[-1] = ext[1:]
		name = '.'.join(parts)
		path = os.path.join(dirname, name)
		self.image.save(path, format, **options)

