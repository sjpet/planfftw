# -*- coding: utf-8 -*-
"""
======================================================================
Function planners for FFT and FFT-based computations (:mod:`planfftw`)
======================================================================

If available, the planned functions use `pyfftw` as a backend for great good.
If not, automatic fallback to `scipy.fftpack` and `numpy` is provided.

FFT
===

.. autosummary::
    :toctree: generated/

    fft         -- One-dimensional FFT
    fft_pair    -- One-dimensional FFT and IFFT
    rfft        -- One-dimensional, real-valued FFT
    rfft_pair   -- One-dimensional, real-valued FFT and IFFT
    fftn        -- N-dimensional FFT
    fftn_pair   -- N-dimensional FFT and IFFT
    rfftn       -- N-dimensional, real-valued FFT
    rfftn_pair  -- N-dimensional, real-valued FFT and IFFT
    ifft        -- One-dimensional IFFT
    ifft_pair   -- One-dimensional IFFT and FFT
    irfft       -- One-dimensional, real-valued IFFT
    irfft_pair  -- One-dimensional, real-valued IFFT and FFT
    ifftn       -- N-dimensional IFFT
    ifftn_pair  -- N-dimensional IFFT and FFT
    irfftn      -- N-dimensional, real-valued IFFT
    irfftn_pair -- N-dimensional, real-valued IFFT and FFT

Filtering
=========

.. autosummary::
    :toctree: generated/

    firfilter               -- FIR filtering
    firfilter_const_filter  -- FIR filtering, constant filter
    firfilter_const_signal  -- FIR filtering, constant signal
    firfilter2              -- FIR filtering, advanced use cases

Convolution
===========

.. autosummary::
    :toctree: generated/

    convolve            -- Convolution
    convolve_const_x    -- Convolution, constant first array
    convolve_const_y    -- Convolution, constant second array
    convolve2           -- Convolution, advanced use cases

Correlation
===========

.. autosummary::
    :toctree: generated/

    correlate           -- Correlation
    correlate_const_x   -- Correlation, constant first array
    correlate_const_y   -- Correlation, constant second array
    correlate2          -- Correlation, advanced uses cases
"""

from .planners import (fft,
                       fft_pair,
                       rfft,
                       rfft_pair,
                       fftn,
                       fftn_pair,
                       rfftn,
                       rfftn_pair,
                       ifft,
                       ifft_pair,
                       irfft,
                       irfft_pair,
                       ifftn,
                       ifftn_pair,
                       irfftn,
                       irfftn_pair,
                       firfilter,
                       firfilter2,
                       firfilter_const_filter,
                       firfilter_const_signal,
                       correlate,
                       correlate2,
                       correlate_const_x,
                       correlate_const_y,
                       convolve,
                       convolve2,
                       convolve_const_x,
                       convolve_const_y)

# clean up namespace
del planners

__version__ = '0.3'
