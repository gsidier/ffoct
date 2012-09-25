#!/usr/bin/env python
import cherrypy
import json

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import re

from ffoct.util import loadgrey16, loadmask

RE_SIZE = r'([0-9]+)x([0-9]+)mm'
RE_DEPTH = r'([0-9.]+)um'

REGEXES = {
	RE_SIZE: ('width_mm', 'height_mm'),
	RE_DEPTH: ('depth_um',)
}

THUMBNAIL_SZ = 100

class SampleServer(object):
	
	def __init__(self, imgdir, workdir):
		self.imgdir = imgdir
		self.masters = { } # { id: (path, props) }
		self.masks = { } # { id: { label: path } }
		
	def find_imgs(self):
		filenames = os.listdir(self.imgdir)
		
		for fname in filenames:
			ext = fname.split('.')[-1]
			if ext.lower() != 'tif':
				continue
			noext = fname[:- (len(ext) + 1)]
			parts = noext.split('-')
			id = parts[0]
			if parts[1] == 'MASK':
				label = parts[2]
				if id not in self.masks:
					self.masks[id] = { }
				masks = self.masks[id]
				masks[label] = fname
			else:
				props = { }
				for part in parts[1:]:
					for (regex, keys) in REGEXES.iteritems():
						m = re.match(regex + '$', part)
						if m:
							data = zip(keys, m.groups())
							props.update(data)
							break
					self.masters[id] = (fname, props)
	
	def _master_thumbnail_path(self, id):
		return os.path.join(self.workdir, '%s-thumb.png' % id)
	
	def master_thumbnail(self, id):
		thumb_path = self._master_thumbnail_path(id)
		path, props = self.masters[id]
		if not os.path.exists(thumb_path) or os.path.getmtime(thumb_path) < path:
			try:
				os.path.remove(thumb_path)
			except:
				pass
			im = Image.open(path)
			w, h = im.size
			fact = float(min(w, h)) / float(THUMBNAIL_SZ)
			w2 = int(w / fact)
			h2 = int(h / fact)
			im.thumbnail((w2, h2))
			im.save(thumb_path)
		return thumb_path

class WebFFOCT:
	
	def __init__(self, samples):
		self.samples = samples
	
	def get_masters(self, **kwargs):
		cherrypy.response.headers['Content-type'] = 'application/json'
		res = self.samples.masters.keys()
		res_json = json.dumps(res)
		return res_json
		
	def get_thumbnail(self, id, **kwargs):
		
		
