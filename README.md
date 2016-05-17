# SpikeFuel

[![Build Status](https://travis-ci.org/duguyue100/spikefuel.svg?branch=master)](https://travis-ci.org/duguyue100/spikefuel)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](http://doge.mit-license.org)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/cfd5f9a9dfd747379c92236d5986c90c)](https://www.codacy.com/app/duguyue100/spikefuel?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=duguyue100/spikefuel&amp;utm_campaign=Badge_Grade)

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
provide most dependencies. I strongly recommend this distribution if you don't
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

## jAER

This project is designed to use along with [jAER](http://jaerproject.org) for
accessing DVS camera, perform the recordings, etc. Please find the installation
instructions of jAER from [here](https://sourceforge.net/p/jaer/wiki/jAER%20Installation/).
For additional materials in setup this package, you can have a look at some development logs:

+ [On configuring jAER](https://github.com/duguyue100/spikefuel/wiki/Old-README-in-Development#on-configuring-jaer)
+ [On Remote Control of jAER using Terminal](https://github.com/duguyue100/spikefuel/wiki/Old-README-in-Development#on-remote-control-of-jaer-using-terminal)
+ [On Remote Control of jAER using Python](https://github.com/duguyue100/spikefuel/wiki/Old-README-in-Development#on-remote-control-of-jaer-using-python)

_A detailed guide will be presented as a markdown document in this project in future._

This project is developed with [DAViS240C](http://inilabs.com/products/dynamic-and-active-pixel-vision-sensor/davis-specifications/).
Therefore most of the functions are specialized for this DVS camera.
Supports for more DVS camera will be added in near future.
Please use with caution.

## Contacts

Yuhuang Hu  
Email: duguyue100@gmail.com
