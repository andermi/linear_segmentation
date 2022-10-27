#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='linear_segmentation',
    version='0.1',
    description='Automatic sparse sampling of 1-D array into linear segments minimizing error',
    long_description=read('README.md'),
    author='Michael Anderson',
    author_email='manders9@jhu.edu',
    url='http://github.com/andermi/linear_segmentation',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib', 'tqdm'],
    python_requires='>=3',
    # scripts=['linear_segmentation/examples/linear_segmentation_with_slider.py']
)