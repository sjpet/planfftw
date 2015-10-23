#!usr/bin/env python
import numpy as np

from planfftw import (firfilter, correlate, convolve)
# noinspection PyProtectedMember
from planfftw._utils import pad_array

from numpy.testing import assert_array_almost_equal


class TestFirFilter:
    b = np.array([0, 1, -2])
    x = np.array([1.2, 0.4, -0.1, 1.2, 1.1, 0.9, -0.7, -0.3, 0.1, -0.2, 0.3])
    x2 = np.array([[1.2, 0.4, -0.1, 1.2, 1.1, 0.9,
                    -0.7, -0.3, 0.1, -0.2, 0.3],
                   [0.7, 1.1, -0.2, -0.3, 0.1, 0.2,
                    -0.2, -0.7, 1.2, 1.0, 0.8]])
    x3 = np.array([[[1.2, 0.4, -0.1, 1.2, 1.1, 0.9,
                     -0.7, -0.3, 0.1, -0.2, 0.3],
                    [0.7, 1.1, -0.2, -0.3, 0.1, 0.2,
                     -0.2, -0.7, 1.2, 1.0, 0.8]],
                   [[0.7, 1.1, -0.2, -0.3, 0.1, 0.2,
                     -0.2, -0.7, 1.2, 1.0, 0.8],
                    [1.2, 0.4, -0.1, 1.2, 1.1, 0.9,
                     -0.7, -0.3, 0.1, -0.2, 0.3]]])

    xfb = np.array([0., 1.2, -2., -0.9, 1.4, -1.3, -1.3, -2.5, 1.1, 0.7, -0.4])
    x2fb = np.array([[0., 1.2, -2., -0.9, 1.4, -1.3,
                      -1.3, -2.5, 1.1, 0.7, -0.4],
                     [0., 0.7, -0.3, -2.4, 0.1, 0.7,
                      0., -0.6, -0.3, 2.6, -1.4]])

    def test_firfilter_1d_input(self):
        my_firfilter = firfilter(self.b, self.x)
        assert_array_almost_equal(my_firfilter(self.b, self.x), self.xfb)

    def test_firfilter_1d_input_x_constant(self):
        my_firfilter = firfilter(self.b, self.x, constant_signal=True)
        assert_array_almost_equal(my_firfilter(self.b), self.xfb)

    def test_firfilter_1d_input_b_constant(self):
        my_firfilter = firfilter(self.b, self.x, constant_filter=True)
        assert_array_almost_equal(my_firfilter(self.x), self.xfb)

    def test_firfilter_2d_input(self):
        my_firfilter = firfilter(self.b, self.x2)
        assert_array_almost_equal(my_firfilter(self.b, self.x2), self.x2fb)

    def test_firfilter_2d_input_x_constant(self):
        my_firfilter = firfilter(self.b, self.x2, constant_signal=True)
        assert_array_almost_equal(my_firfilter(self.b), self.x2fb)

    def test_firfilter_2d_input_b_constant(self):
        my_firfilter = firfilter(self.b, self.x2, constant_filter=True)
        assert_array_almost_equal(my_firfilter(self.x2), self.x2fb)

    def test_firfilter_1d_segments(self):
        my_firfilter = firfilter(self.b, self.x2, x_segmented=True)
        assert_array_almost_equal(my_firfilter(self.b, 0), self.x2fb[0])
        assert_array_almost_equal(my_firfilter(self.b, 1), self.x2fb[1])

    def test_firfilter_2d_segments(self):
        my_firfilter = firfilter(self.b, self.x3, x_segmented=True)
        assert_array_almost_equal(my_firfilter(self.b, 0), self.x2fb)
        assert_array_almost_equal(my_firfilter(self.b, 1),
                                  self.x2fb[::-1, :])


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

    def test_convolve_1d(self):
        my_convolve_1 = convolve(self.x, self.y)
        my_convolve_2 = convolve(self.x, self.y, 'same')
        my_convolve_3 = convolve(self.x, self.y, 'valid')
        assert_array_almost_equal(my_convolve_1(self.x, self.y),
                                  self.cxy)
        assert_array_almost_equal(my_convolve_2(self.x, self.y),
                                  self.cxy[2:9])
        assert_array_almost_equal(my_convolve_3(self.x, self.y),
                                  self.cxy[4:7])

    def test_convolve_1d_x_constant(self):
        my_convolve_1 = convolve(self.x, self.y, constant_x=True)
        my_convolve_2 = convolve(self.x, self.y, 'same', constant_x=True)
        my_convolve_3 = convolve(self.x, self.y, 'valid', constant_x=True)
        assert_array_almost_equal(my_convolve_1(self.y),
                                  self.cxy)
        assert_array_almost_equal(my_convolve_2(self.y),
                                  self.cxy[2:9])
        assert_array_almost_equal(my_convolve_3(self.y),
                                  self.cxy[4:7])

    def test_convolve_1d_y_constant(self):
        my_convolve_1 = convolve(self.x, self.y, constant_y=True)
        my_convolve_2 = convolve(self.x, self.y, 'same', constant_y=True)
        my_convolve_3 = convolve(self.x, self.y, 'valid', constant_y=True)
        assert_array_almost_equal(my_convolve_1(self.x),
                                  self.cxy)
        assert_array_almost_equal(my_convolve_2(self.x),
                                  self.cxy[2:9])
        assert_array_almost_equal(my_convolve_3(self.x),
                                  self.cxy[4:7])

    # single axes in two N-D arrays
    # multiple axes in N-D arrays
    # single axis in N-D with 1-D
    # 1-D with single axis in N-D
    # 1-D segments with 1-D sequence
    # N-D segments with N-D sequence
    # single axis in N-D segments with 1-D sequence
