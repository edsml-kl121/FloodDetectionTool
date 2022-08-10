#!/usr/bin/env python

from setuptools import setup, Extension


import os
from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()



setup(name='Flood Tool',
      version='1.0',
      description='Flood Risk Analysis Tool',
      author='EDMSL project team',
      packages=['flood_tool'], 
      install_requires=required
      )
