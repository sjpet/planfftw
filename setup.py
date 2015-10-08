#!usr/bin/env python
from distutils.core import setup

import planfftw

setup(name="planfftw",
      packages=["planfftw"],
      version=planfftw.__version__,
      description="Julia-inspired function planners around pyfftw",
      author="Stefan Peterson",
      author_email="stefan.j.peterson@gmail.com",
      url="https://github.com/sjpet/planfftw",
      download_url="https://github.com/sjpet/planfftw/tarball/0.1",
      requires=['numpy'],
      install_requires=["pyfftw>=0.9.2",
                        "numpy>=1.8.2"],
      keywords=["pyfftw", "fft", "fftw", "plan", "planner"],
      classifiers = [])

