planfftw
========

planfftw provides Julia-inspired function planners for fft and a number of fft-
based operations. The initial objective was to provide a clean and highly
optimized interface to the excellent [PyFFTW](https://github.com/pyFFTW/pyFFTW)
package. However, PyFFTW is not a strict requirement and planfftw falls back on
scipy.fftpack or numpy.fft, should it not be available.

Issues, requests and questions can be raised at the Github 
[issues](https://github.com/sjpet/planfftw/issues)-page.

The documentation, albeit somewhat crude, is available on 
[Read the Docs](https://planfftw.readthedocs.io/en/latest/).

Requirements
------------
Python 3.5 or higher
numpy 1.16.1 or higher

Earlier versions may very well work but are not tested.

Highly recommended
-----------
PyFFTW

Installation
------------
Install with::

    pip install planfftw

or::

    easy_install planfftw
