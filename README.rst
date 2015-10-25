planfftw
========

planfftw provides Julia-inspired function planners for fft and a number of fft-
based operations. The initial objective was to provide a clean and highly
optimized interface to the excellent PyFFTW package, but falls back on 
scipy.fftpack or numpy.fft should that not be available.

Requirements
------------
Python 3.4 or higher OR Python 2.7 or higher
numpy 1.8.2

Earlier versions might work but have not been tested

Highly recommended
-----------
PyFFTW

Installation
------------

    pip install planfftw

or::

    easy_install planfftw
