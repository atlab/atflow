#!/usr/bin/env python
from setuptools import setup, find_packages
import sys
from os import path

if sys.version_info < (3,4):
    sys.exit('atflow is only supported on Python 3.4 or higher')

here = path.abspath(path.dirname(__file__))

long_description = "A collection of TensorFlow libraries for Andreas Tolias lab"

# read in version number
with open(path.join(here, 'atflow', 'version.py')) as f:
    exec(f.read())


setup(
    name='atflow',
    version=__version__,
    description="TensorFlow library for Andreas Tolias lab",
    long_description=long_description,
    author='Edgar Y. Walker',
    author_email='edgar.walker@gmail.com',
    license="GNU LGPL",
    url='https://github.com/atlab/atflow',
    keywords='tensorflow',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=['numpy'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 :: Only',
    ],
)
