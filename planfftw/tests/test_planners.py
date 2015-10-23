#!usr/bin/env python
import numpy as np

from planfftw import (firfilter, correlate, convolve)
# noinspection PyProtectedMember
from planfftw._utils import pad_array

from numpy.testing import assert_array_almost_equal


class TestFirFilter:
    b = np.array([0, 1, -2])
    x = np.array([1.2, 0.4, -0.1, 1.2, 1.1, 0.9, -0.7, -0.3, 0.1, -0.2, 0.3])

    xfb = np.array([0., 1.2, -2., -0.9, 1.4, -1.3, -1.3, -2.5, 1.1, 0.7, -0.4])

    def test_simple_firfilter(self):
        my_firfilter = firfilter(self.b, self.x)
        assert_array_almost_equal(my_firfilter(self.b, self.x), self.xfb)


class TestCorrelate:
    x = np.array([1.2, 0.3, -0.2, 0.7, 0.3, 0.1, 0.1])
    y = np.array([1.1, 0.7, 0.1, -0.2, -0.6])

    rxy = np.array([-0.72, -0.42, 0.18, 0.49, 1.19, 0.14, 0.22, 0.97, 0.41,
                    0.18, 0.11])

    def test_simple_correlate(self):
        my_correlate_1 = correlate(self.x, self.y)
        my_correlate_2 = correlate(self.x, self.y, 'same')
        my_correlate_3 = correlate(self.x, self.y, 'valid')
        assert_array_almost_equal(my_correlate_1(self.x, self.y),
                                  self.rxy)
        assert_array_almost_equal(my_correlate_2(self.x, self.y),
                                  self.rxy[2:9])
        assert_array_almost_equal(my_correlate_3(self.x, self.y),
                                  self.rxy[4:7])

class TestConvolve:
    x = np.array([1.2, 0.3, -0.2, 0.7, 0.3, 0.1, 0.1])
    y = np.array([1.1, 0.7, 0.1, -0.2, -0.6])

    cxy = np.array([ 1.32, 1.17, 0.11, 0.42, 0.02, 0.25, 0.19, -0.4, -0.19,
                     -0.08, -0.06])


    def test_simple_convolve(self):
        my_convolve_1 = convolve(self.x, self.y)
        my_convolve_2 = convolve(self.x, self.y, 'same')
        my_convolve_3 = convolve(self.x, self.y, 'valid')
        assert_array_almost_equal(my_convolve_1(self.x, self.y),
                                  self.cxy)
        assert_array_almost_equal(my_convolve_2(self.x, self.y),
                                  self.cxy[2:9])
        assert_array_almost_equal(my_convolve_3(self.x, self.y),
                                  self.cxy[4:7])