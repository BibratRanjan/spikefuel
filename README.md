# SpikeFuel

[![Build Status](https://travis-ci.org/duguyue100/spikefuel.svg?branch=master)](https://travis-ci.org/duguyue100/spikefuel)

SpikeFuel is a toolkit for converting popular visual benchmarks to
spiking neuromorphic datasets.

The design principle of this package is to eliminate human intervention during
the experiment as much as possible. In such way, human experimenter just needs
to setup a proper environment and lets the pipeline run!

The general objectives are:

+ Precise control of record logging with Python. :checkered_flag:
+ User interface for showing video or images in routine. :checkered_flag:
+ Experiment configuration system (with JSON style). :checkered_flag:
+ Post signal analysis and selection tools. :checkered_flag:

## Requirements

The scientific Python distribution - [Anaconda](https://anaconda.org/) will
provide most dependencies. I recommend this distribution if you don't
want to mess with system's Python.

3rd party packages
+ `numpy` (included in Anaconda)
+ `sacred` (install by `pip install sacred`)
+ `subprocess32` (install by `pip install subprocess32`)

3rd party packages that are installed with Anaconda
+ `ffmpeg` (follow the [link](https://anaconda.org/soft-matter/ffmpeg))
+ `pyav` (follow the [link](https://anaconda.org/soft-matter/pyav))
+ `opencv` (follow the [link](https://anaconda.org/menpo/opencv))

## Installation

You can install the bleeding edge version from `pip`:

```
pip install git+git://github.com/duguyue100/spikefuel.git
```

## Contacts

Yuhuang Hu  
Email: duguyue100@gmail.com
