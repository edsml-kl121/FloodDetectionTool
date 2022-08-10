import sys
import os

# We're working in the ./docs directory, but need the package root in the path
# This command appends the directory one level up, in a cross-platform way.
#sys.path.insert(0, os.path.abspath(os.sep.join((os.curdir, '..'))))
sys.path.append(os.path.abspath('../flood_tool'))


project = 'Flood Tool'
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon', 
              'sphinx.ext.mathjax',
              'rinoh.frontend.sphinx']
pdf_documents = [(u'mpm_la.pdf', u'mpm_la pdf doc',  u'lc2216'), ]
source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = ['_build']
autoclass_content = "both"

html_theme = 'sphinxdoc'


