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
from skimage.filter import denoise_tv_bregman
from skimage.filter import gabor_filter 
import numpy

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
		base, ext = os.path.splitext(name)
		parts = base.split('_')
		ident = parts[0]
		tags = parts[1:]
		tags.append(tag)
		res = '_'.join([ident] + tags) + ext
		return res
	
	def derive_path(self, path, tag):
		name = os.path.basename(path)
		name = self.derive_name(name, tag)
		path = os.path.join(self.config.derived_dir, name)
		return path
	
	def generate(self, base, *ops):
		"""
		base: path of base
		ops: list of (op, arg1, arg2, ...)
		"""
		# base is a cached version for a subset of the transformations.
		# i0 is the start index of the ops that need to be applied to it.
		
		# fix dir if necessary:
		dirname = os.path.dirname(base)
		if dirname == '':
			if '_' in base:
				base = os.path.join(self.config.derived_dir, base)
			else:
				base = os.path.join(self.config.master_dir, base)
			
		path = base
		
		i0 = 0
		for i, op_args in enumerate(ops):
			tag = ','.join(map(str, op_args))
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
	
	def transf(self, op, args = ()):
		tag = ','.join(map(str, [op] + list(args)))
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
		return self.transf('to8bit')
	
	def _denoise(self, weight):
		return denoise_tv_bregman(numpy.array(self.image), weight)
	
	def denoise(self, weight):
		return self.transf('denoise', (weight,))
	
	def _contrast(self, pow):
		im = numpy.array(self.image)
		im = im / float(numpy.max(im))
		return im ** pow
	
	def contrast(self, pow):
		return self.transf('contrast', (pow,))
	
	def _gabor(self, *params):
		real, imag = gabor_filter(numpy.array(self.image), *params)
		return real + 1j * imag
	
	def gabor(self, *params):
		return self.transf('gabor', tuple(params))
	
	def _real(self):
		return numpy.real(self.image)
	
	def real(self):
		return self.transf('real')
	
	def _imag(self):
		return numpy.imag(self.image)
	
	def imag(self):
		return self.transf('imag')
	
	def _abs(self):
		return numpy.abs(self.image)
	
	def _renorm(self):
		return (self.image - numpy.mean(self.image)) / numpy.std(self.image)
	
	def renorm(self):
		return self.transf('renorm')
	
	def abs(self):
		return self.transf('abs')
	
	def save(self, format = None, **options):
		if hasattr(self.image, 'save'):
			image = self.image
		else:
			image = Image.fromarray(self.image)
		if format is None:
			_, format = os.path.splitext(self.path)
		ext = format.lower()
		format = Image.EXTENSION[ext]
		dirname = os.path.dirname(self.path)
		name = os.path.basename(self.path)
		base, _ = os.path.splitext(name)
		parts = base.split('_')
		name = '_'.join(parts) + ext
		path = os.path.join(dirname, name)
		image.save(path, format, **options)

