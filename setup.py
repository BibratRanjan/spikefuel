"""Setup script for the spikefuel package.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from setuptools import setup

classifiers = """
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Natural Language :: English
Operating System :: OS Independent
Programming Language :: Python :: 2.7
Topic :: Utilities
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Neuromorphic Engineering
Topic :: Scientific/Engineering :: Artificial Intelligence
Topic :: Software Development :: Libraries :: Python Modules
License :: OSI Approved :: MIT License
"""

try:
    from spikefuel import __about__
    about = __about__.__dict__
except ImportError:
    about = dict()
    exec(open("spikefuel/__about__.py").read(), about)

setup(
    name='spikefuel',
    version=about['__version__'],

    author=about['__author__'],
    author_email=about['__author_email__'],

    url=about['__url__'],

    packages=['spikefuel'],
    scripts=[],

    classifiers=list(filter(None, classifiers.split('\n'))),
    description='Toolkit for converting visual recognition benchmarks \
                 to spiking neuromorphic datasets.',
    long_description=open('README.md').read()
)
