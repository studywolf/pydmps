#!/usr/bin/env python
import imp
import io
import sys
import os

try:
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import find_packages, setup


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

setup(
  name = 'pydmps',
  packages = ['pydmps'],
  version = '0.2',
  description = 'Python implementation of Dynamic Movement Primitives (DMPs)',
  author = 'Travis DeWolf',
  author_email = 'travis.dewolf@gmail.com',
  url = 'https://github.com/studywolf/pydmps',
  download_url = 'https://github.com/studywolf/pydmps/tarball/0.2',
  keywords = ['dmps', 'dynamic movement primitives', 'trajectory generation', 'control'],
  classifiers = [],
)
