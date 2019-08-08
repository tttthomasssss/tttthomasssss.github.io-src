#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Thomas Kober'
SITENAME = 'NLP & Me'
SITEURL = ''

TIMEZONE = 'Europe/London'

DEFAULT_LANG = 'en'

# Basic Settings
DISPLAY_PAGES_ON_MENU = True
DISPLAY_CATEGORIES_ON_MENU = True
HIDE_SIDEBAR = True
DELETE_OUTPUT_DIRECTORY = False

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Paths
PATH = 'content'
ARTICLE_PATHS = ['blog']
PAGE_PATHS = ['pages']

# Theme
THEME = 'pelican-bootstrap3'

BOOTSTRAP_THEME = 'readable'
PYGMENTS_STYLE = 'monokai'

# Blogroll
#LINKS = (('Pelican', 'http://getpelican.com/'),
#         ('Python.org', 'http://python.org/'),
#         ('Jinja2', 'http://jinja.pocoo.org/'),
#         ('You can modify those links in your config file', '#'),)

# Social widget
#SOCIAL = (('You can add links in your config file', '#'),
#          ('Another social link', '#'),)

SLUGIFY_SOURCE = 'title'

DEFAULT_PAGINATION = 10

PLUGIN_PATHS = [
    '/Users/thomas/DevSandbox/InfiniteSandbox/tag-lab/pelican_plugins/pelican-plugins'
]

PLUGINS = [
    #'bootstrapify',
    'i18n_subsites',
    'liquid_tags.include_code',
    'liquid_tags.notebook',
]

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True

STATIC_PATHS = ['images', 'pdfs', 'notebooks', 'bibtex', 'datasets', 'other']

PAGE_ORDER_BY = 'sortorder'

'''
MENUITEMS = (
  ('Home', '/'),
  ('About', '/about/'),
  ('Contact', '/contact/'),
  ('Blog', '/blog/'),
)
'''
# TODO: http://habeebq.github.io/archives.html

# Jinja settings
JINJA_ENVIRONMENT = {'extensions': ['jinja2.ext.i18n']}