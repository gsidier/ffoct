#!/usr/bin/env python
import cherrypy
import json

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import re

from ffoct.util import loadgrey8, loadgrey16, loadmask
from ffoct.samples import find_samples
from PIL import Image

RE_SIZE = r'([0-9]+)x([0-9]+)mm'
RE_DEPTH = r'([0-9.]+)um'

REGEXES = {
	RE_SIZE: ('width_mm', 'height_mm'),
	RE_DEPTH: ('depth_um',)
}

THUMBNAIL_SZ = 100
LORES_SZ = 500
LORES_QUAL = 85

SAMPLE_SZ = 200
SAMP_THUMB_SZ = 100

def need_update(output_path, *deps):
	if not os.path.exists(output_path) or any(os.path.getmtime(output_path) < os.path.getmtime(path) for path in deps):
		try:
			os.remove(output_path)
		except OSError:
			pass
		return True
	return False

class SampleServer(object):
	
	def __init__(self, imgdir, workdir):
		self.imgdir = imgdir
		self.workdir = workdir
		self.masters = { } # { id: (fname, props) }
		self.masks = { } # { id: { label: path } }
		self.find_imgs()
		
	def find_imgs(self):
		filenames = os.listdir(self.imgdir)
		
		for fname in filenames:
			ext = fname.split('.')[-1]
			if ext.lower() not in ('tif', 'png', 'jpg'):
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
	
	def _master_path(self, id):
		fname, props = self.masters[id]
		master_path = os.path.join(self.imgdir, fname)
		return master_path
	
	def _master_thumbnail_path(self, id):
		return os.path.join(self.workdir, '%s-thumb.png' % id)
	
	def master_thumbnail(self, id):
		thumb_path = self._master_thumbnail_path(id)
		master_path = self._master_path(id)
		if not os.path.exists(thumb_path) or os.path.getmtime(thumb_path) < os.path.getmtime(master_path):
			try:
				os.remove(thumb_path)
			except OSError:
				pass
			im = Image.open(master_path)
			w, h = im.size
			fact = float(max(w, h)) / float(THUMBNAIL_SZ)
			w2 = int(w / fact)
			h2 = int(h / fact)
			im.thumbnail((w2, h2))
			im = im.convert(mode = 'I') # 8-bit
			im.save(thumb_path)
		return thumb_path
	
	def _master_lores_path(self, id):
		return os.path.join(self.workdir, '%s-lores.jpg' % id)

	def master_lores(self, id):
		lores_path = self._master_lores_path(id)
		master_path = self._master_path(id)
		if not os.path.exists(lores_path) or os.path.getmtime(lores_path) < os.path.getmtime(master_path):
			try:
				os.remove(lores_path)
			except OSError:
				pass
			m = loadgrey8(master_path)
			im = Image.fromarray(m, mode = 'L')
			w, h = im.size
			fact = float(max(w, h)) / float(LORES_SZ)
			w2 = int(w / fact)
			h2 = int(h / fact)
			im.thumbnail((w2, h2))
			im.save(lores_path, quality = LORES_QUAL)
		return lores_path
	
	def _sample_thumbnail_path(self, id, x, w, y, h):
		return os.path.join(self.workdir, 'sample.%s.%d-%d.%dx%d.thumb.png' % (id, x, y, w, h))
	
	def sample_thumbnail(self, id, x, y, w, h):
		thumb_path = self._sample_thumbnail_path(id, x, y, w, h)
		master_path = self._master_path(id)
		if need_update(thumb_path, master_path):
			im = Image.open(master_path)
			smp = im.crop((x, y, x + w, y + h))
			smp.thumbnail((SAMP_THUMB_SZ, SAMP_THUMB_SZ))
			smp = smp.convert(mode = 'I') # 8-bit
			smp.save(thumb_path)
		return thumb_path

	def get_master(self, id):
		master_path = self._master_path(id)
		master = loadgrey16(master_path)
		return master
	
	def get_mask(self, id, label):
		mask_path = os.path.join(self.imgdir, self.masks[id][label])
		mask = loadmask(mask_path)
		return mask
	
	def _samples(self, id):
		im = self.get_master(id)
		samples = { }
		for label in self.masks[id]:
			mask = self.get_mask(id, label)
			samps = find_samples(im, mask, SAMPLE_SZ, SAMPLE_SZ)
			samples[label] = samps
		return samples
	
	def samples(self, id):
		samples_path = os.path.join(self.workdir, 'samples-%s-%sx%s.json' % (id, SAMPLE_SZ, SAMPLE_SZ))
		master_path = self._master_path(id)
		if need_update(samples_path, master_path):
			data = self._samples(id)
			with file(samples_path, 'w') as f:
				json.dump(data, f)
		with file(samples_path, 'r') as f:
			data = json.load(f)
		return data
	
class StatsServer(object):
	
	def __init__(self, samples):
		self.samples = samples

	def filter_samples(self, masters = None, labels = None, filter = None):
		if masters is None: 
			masters = self.samples.masters.keys()
		
		samples = [ (label, master, samp) 
			for master in masters 
			for (label, lst) in self.samples.samples(master).items()
			if (labels is not None) and label in labels
			for sample in lst 
			if filter and filter(master, sample)
		]
		return samples
	
	def calc_stats(self, stat, samples):
		res = list( (label, master, samp, stat(master, samp)) 
			for (label, master, samp) in samples )
	
	def histogram(self, stats):
		pass
	
class WebFFOCT:
	
	def __init__(self, samples):
		self.samples = samples
	
	def setup_routes(self, routes):
		routes.connect(
			name = 'masters',
			route = '/masters/',
			controller = self, 
			action = 'get_masters',
			conditions = dict(method = ['GET'])
		)
		routes.connect(
			name = 'master_thumbnail',
			route = '/masters/{id}/thumbnail',
			controller = self,
			action = 'get_thumbnail',
			conditions = dict(method = ['GET'])
		)
		routes.connect(
			name = 'master_lores',
			route = '/masters/{id}/lores',
			controller = self,
			action = 'get_lores',
			conditions = dict(method = ['GET'])
		)
		routes.connect(
			name = 'samples',
			route = '/masters/{id}/samples',
			controller = self,
			action = 'get_samples',
			conditions = dict(method = ['GET'])
		)
		routes.connect(
			name = 'sample',
			route = '/masters/{id}/sample/thumbnail',
			controller = self,
			action = 'get_sample_thumbnail',
			conditions = dict(method = ['GET'])
		)
	
	def get_masters(self, **kwargs):
		cherrypy.response.headers['Content-type'] = 'application/json'
		res = list({'id': id, 'props': props} for (id, (path, props)) in self.samples.masters.iteritems() )
		res_json = json.dumps(res)
		return res_json
	
	def get_thumbnail(self, id, **kwargs):
		cherrypy.response.headers['Content-type'] = 'image/png'
		path = self.samples.master_thumbnail(id)
		f = file(path, 'r')
		return f.read()
	
	def get_lores(self, id, **kwargs):
		cherrypy.response.headers['Content-type'] = 'image/jpeg'
		path = self.samples.master_lores(id)
		f = file(path, 'r')
		return f.read()
	
	def get_samples(self, id, **kwargs):
		cherrypy.response.headers['Content-type'] = 'application/json'
		res = self.samples.samples(id)
		res_json = json.dumps(res)
		return res_json
	
	def get_sample_thumbnail(self, id, x, y, w, h, **kwargs):
		cherrypy.response.headers['Content-type'] = 'image/png'
		x, y, w, h = map(int, (x, y, w, h))
		path = self.samples.sample_thumbnail(id, x=x, y=y, w=w, h=h)
		f = file(path, 'r')
		return f.read()

if __name__ == '__main__':
	import os, sys
	
	imgdir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'img'))
	workdir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tmp'))
	samplesrv = SampleServer(imgdir, workdir)
	
	api = WebFFOCT(samplesrv)
	
	routes = cherrypy.dispatch.RoutesDispatcher()
	routes.mapper.explicit = False
	api.setup_routes(routes)
	
	HTDOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), 'www'))
	
	global_config = {
		'server.socket_host': '0.0.0.0',
		'server.socket_port': 8080,
	}
	
	root_config = {
		'/': {
			'tools.staticdir.on': True,
			'tools.staticdir.dir': HTDOCS,
			'tools.staticdir.index': 'index.html',
		}
	}
	cherrypy.tree.mount(None, '/', config = root_config)
	
	API_ROOT = '/api'
	api_config = {
		'/': {
			'request.dispatch': routes,
			'tools.sessions.on': True
		}
	}
	cherrypy.tree.mount(None, API_ROOT, config = api_config)
	
	cherrypy.engine.start()
	cherrypy.engine.block()

