#!/usr/bin/env python
from sampset import SampleSet, Sample
import config
from ffoct import reduce_image, Timer

import numpy
import matplotlib.colors
import pylab
from math import ceil, sqrt
from time import time
from collections import Counter
import gc
import pdb
from blist import sortedlist

from skimage.feature import local_binary_pattern

RESIZE_FACT = 1. / 10.

SPLIT_THRESH = 1.2
SPLIT_MAXSIZE = 256
SPLIT_MINSIZE = 20

MERGE_GRACE = .1
MERGE_THRESH = 1.3

def cum_counts(feat, bins = None):
	"""
	Given an N1 x N2 x ... Nn array of integer features F which take values 0 to M - 1, 
	return  an (N1+1) x (N2+1) x ... (Nn+1) x M array C such that 
		
		C[i1, ..., in, a] = #{ (j1, .. jn), forall k, jk < ik, F[j1, .. jn] == a }
	
	"""
	bins = bins or numpy.max(feat) + 1
	counts = numpy.zeros(((bins,) + feat.shape), dtype = numpy.int32)
	print feat.shape, bins, numpy.product(feat.shape) * bins
	with Timer("count"):
		for b in xrange(bins):
			counts[b] = feat == b
			gc.collect()
	with Timer("cumsum"):
		for axis in xrange(len(feat.shape)):
			numpy.cumsum(counts, axis = axis + 1, out = counts)
	with Timer("copy"):
		C = numpy.zeros([numpy.shape(counts)[0]] + [n + 1 for n in numpy.shape(counts)[1:]], 
			dtype = numpy.float32)
		C[:, 1:, 1:] = counts
		del counts
		gc.collect()
	return C

def normalize_hist(hist):
	norm = hist.sum()
	return hist / norm if norm != 0 else hist

def local_histogram(counts, x1, x2, normalize = True):
	i1, j1 = x1
	i2, j2 = x2
	hist = counts[:, i2, j2] + counts[:, i1, j1] - counts[:, i1, j2] - counts[:, i2, j1]
	if normalize:
		return normalize_hist(hist)
	else:
		return hist

def chi2(h1, h2):
	if numpy.all(h1 == 0) and numpy.all(h2 == 0):
		return 0
	I = (h1 != 0) | (h2 != 0)
	return (.5 * (h1 - h2)[I]**2 / (h1 + h2)[I]).sum()
	
def split(counts, thresh, maxsize, minsize):
	bins, w, h = counts.shape
	w -= 1
	h -= 1
	def rec(x1, y1, x2, y2):
		if min(x2 - x1, y2 - y1) <= minsize:
			return (x1, y1, x2, y2)
		l = max((x2 - x1) / 2, (y2 - y1) / 2)
		assert x2 - x1 >= l # todo handle this case
		assert y2 - y1 >= l # todo handle this case
		regions = tuple((x, y, min(x + l, x2), min(y + l, y2))
			for y in [ y1, y1 + l ]
			for x in [ x1, x1 + l ])
		h0, h1, h2, h3 = list(local_histogram(counts, (x1_, y1_), (x2_, y2_)) for (x1_, y1_, x2_, y2_) in regions)
		d01 = chi2(h0, h1)
		d02 = chi2(h0, h2)
		d03 = chi2(h0, h3)
		d12 = chi2(h1, h2)
		d13 = chi2(h1, h3)
		d23 = chi2(h2, h3)
		dmin = min(d01, d02, d03, d12, d13, d23)
		dmax = max(d01, d02, d03, d12, d13, d23)
		if dmax / dmin < thresh:
			return (x1, y1, x2, y2)
		else:
			reg1 = tuple(rec(x1_, y1_, x2_, y2_) for (x1_, y1_, x2_, y2_) in regions)
			return reg1
	
	regions = tuple((x, y, min(x + maxsize, w), min(y + maxsize, h))
		for y in xrange(0, h - maxsize + 1, maxsize)
		for x in xrange(0, w - maxsize + 1, maxsize))
	
	return tuple(rec(x1, y1, x2, y2) for (x1, y1, x2, y2) in regions)

def depth_first_iter(regions):
	if type(regions[0]) == int:
		yield regions
	else:
		for r in regions:
			for res in depth_first_iter(r):
				yield res
	
def breadth_first_iter(regions):
	cousins = [ regions ]
	while cousins:
		nephews = [ ]
		for r in cousins:
			if type(r[0]) == int:
				yield r
			else:
				for child in r:
					nephews.append(child)
		cousins = nephews

def plot_blocks(regions, **kwargs):
	for x1, y1, x2, y2 in breadth_first_iter(regions):
		if max(x2 - x1, y2 - y1) > SPLIT_MINSIZE:
			pylab.vlines([y1, y2], x1, x2, **kwargs)
			pylab.hlines([x1, x2], y1, y2, **kwargs)

def draw_regions(out, regions):
	for i, (blocks, _, _) in regions.iteritems():
		for (x1, y1, x2, y2) in blocks:
			out[x1:x2, y1:y2] = i
	return out

if __name__ == '__main__':
	
	import os, sys 
	
	restart = None
	if sys.argv[1:]:
		restart, = sys.argv[1:]
	
	run = False
	
	def require(step):
		global run
		run = run or not restart or step == restart
		#not restart or steps[restart] <= steps[step]
		return run
	
	if require('LOAD_IMG'):
		with Timer("Load images ..."):
			# 
			paths = os.listdir(config.master_dir)
			sampleset = SampleSet(config)
			masters = []
			patches = []
			idx = [ 0 ]
			#
			for path in paths:
				try:
					master = sampleset.generate(os.path.join(config.master_dir, path))
				except IOError: # not an image?
					print >> sys.stderr, "Cannot load '%s' - skipping" % path
					continue
				master = reduce_image(master, RESIZE_FACT)
				if not os.path.exists(master.path):
					master.save()
				master = numpy.array(master.image)
				masters.append(master)
			palette = numpy.random.rand(4000, 3)
			discrete_cmap = matplotlib.colors.ListedColormap(palette, name = 'discrete')
			pylab.register_cmap(name = 'discrete', cmap = discrete_cmap)
	
	if require('CALC_FEATURES'):
		with Timer("Compute the features ..."):
			features = [ ]
			for master in masters:
				feat = local_binary_pattern(master, 8, 1)
				feat = feat.astype(int)
				features.append(feat)

	if require('FEATURE_HIST'):
		with Timer("Compute the local feature histograms ..."):
			cdfs = [ ]
			for (master, feat) in zip(masters, features):
				cdf = cum_counts(feat)
				cdfs.append(cdf)
	
	if require('SPLIT'):
		with Timer("Split areas ... "):
			trees = [ ]
			for (master, cdf) in zip(masters, cdfs):
				tree = split(cdf, SPLIT_THRESH, maxsize = SPLIT_MAXSIZE, minsize = SPLIT_MINSIZE)
				trees.append(tree)
	
	if require('MERGE'):
		with Timer("Merge ... "):
			def key(i, j):
				return min(i, j), max(i, j)
			textures = [ ]
			for (master, cdf, tree) in zip(masters, cdfs, trees):
				# texture[x, y] = index of connected region with same texture
				texture = numpy.zeros(master.shape, dtype = numpy.int32)
				w, h = texture.shape
				textures.append(texture)
				#edges = { }  # dict (i, j) -> ((x1, y2), (x2, y2))
				edges = set( )
				regions = { } # dict of i -> ([block1, block2, ...], hist, area)
				for (i, (x1, y1, x2, y2)) in enumerate(depth_first_iter(tree)):
					texture[x1:x2, y1:y2] = i + 1
					blocks = [ (x1, y1, x2, y2) ]
					hist = local_histogram(cdf, (x1, y1), (x2, y2), normalize = False)
					area = (x2 - x1) * (y2 - y1)
					regions[i + 1] = (blocks, hist, area)
				for (x1, y1, x2, y2) in depth_first_iter(tree):
					i = texture[(x1 + x2) / 2, (y1 + y2) / 2]
					for x in xrange(x1 + SPLIT_MINSIZE / 2, x2, SPLIT_MINSIZE):
						y = y1 - 1
						if y >= 0:
							j = texture[x, y]
							assert i != j
							if j > 0:
								edges.add(key(i, j))
						y = y2
						if y < h:
							j = texture[x, y]
							assert i != j
							if j > 0:
								edges.add(key(i, j))
					for y in xrange(y1 + SPLIT_MINSIZE / 2, y2, SPLIT_MINSIZE):
						x = x1 - 1
						if x >= 0:
							j = texture[x, y]
							assert i != j
							if j > 0:
								edges.add(key(i, j))
						x = x2
						if x < w:
							j = texture[x, y]
							assert i != j
							if j > 0:
								edges.add(key(i, j))
				MImax = 0.
				N = len(edges)
				def MI(i, j):
					blocks1, hist1, area1 = regions[i]
					blocks2, hist2, area2 = regions[j]
					area = min(area1, area2)
					hist1 = normalize_hist(hist1)
					hist2 = normalize_hist(hist2)
					G = chi2(hist1, hist2)
					mi = area * G
					return mi
				
				MIcache = dict([ ((i, j), MI(i, j)) for (i, j) in edges ])
				MIlist = sortedlist(edges, key = lambda (i, j): MIcache[(i, j)])
				node2edges = dict([ (i, set()) for i in xrange(1 + sum(1 for _ in depth_first_iter(tree))) ])
				for (i, j) in edges:
					node2edges[i].add((i, j))
					node2edges[j].add((i, j))
				
				for n in xrange(len(edges)):
					i0, j0 = MIlist.pop(0)
					MIcur = MIcache[(i0, j0)]
					
					# stop?
					if float(n) / N > MERGE_GRACE and MIcur / MImax > MERGE_THRESH:
						break
					
					# merge
					blocks1, hist1, area1 = regions[i0]
					blocks2, hist2, area2 = regions[j0]
					blocks = blocks1 + blocks2
					hist = hist1 + hist2
					area = area1 + area2
					regions[i0] = blocks, hist, area
					del regions[j0]
					
					# relabel nodes j0 -> i0
					remove = node2edges[i0].union(node2edges[j0])
					recalc = set([ key(i0 if i == j0 else i, i0 if j == j0 else j) for (i, j) in remove ])
					recalc.discard((i0, i0))
					
					for (i, j) in remove:
						MIlist.discard((i, j))
					
					for (i, j) in remove:
						del MIcache[(i, j)]
					for (i, j) in recalc:
						MIcache[(i, j)] = MI(i, j)
						
					for (i, j) in remove:
						node2edges[i].discard((i, j))
						node2edges[j].discard((i, j))
					
					# recompute all i0's edges
					for (i, j) in recalc:
						MIlist.add((i, j))
						node2edges[i].add((i, j))
						node2edges[j].add((i, j))
					
					print len(regions), MIcur, MImax, MIcur / MImax
					MImax = max(MIcur, MImax)
				draw_regions(texture, regions)

