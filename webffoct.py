#!/usr/bin/env python
import cherrypy
import json

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import re

from ffoct.util import loadgrey16, loadmask
from PIL import Image

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
		self.workdir = workdir
		self.masters = { } # { id: (fname, props) }
		self.masks = { } # { id: { label: path } }
		self.find_imgs()
		
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
		fname, props = self.masters[id]
		path = os.path.join(self.imgdir, fname)
		if not os.path.exists(thumb_path) or os.path.getmtime(thumb_path) < os.path.getmtime(path):
			try:
				os.remove(thumb_path)
			except OSError:
				pass
			im = Image.open(path)
			w, h = im.size
			fact = float(max(w, h)) / float(THUMBNAIL_SZ)
			w2 = int(w / fact)
			h2 = int(h / fact)
			im.thumbnail((w2, h2))
			im = im.convert(mode = 'I') # 8-bit
			im.save(thumb_path)
		return thumb_path

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
	
	def get_masters(self, **kwargs):
		cherrypy.response.headers['Content-type'] = 'application/json'
		#res = self.samples.masters.items()
		res = list({'id': id, 'props': props} for (id, (path, props)) in self.samples.masters.iteritems() )
		res_json = json.dumps(res)
		return res_json
	
	def get_thumbnail(self, id, **kwargs):
		cherrypy.response.headers['Content-type'] = 'image/png'
		path = self.samples.master_thumbnail(id)
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

