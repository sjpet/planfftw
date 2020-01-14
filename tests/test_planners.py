#!usr/bin/env python
import numpy as np
import pytest

from planfftw import (firfilter,
                      firfilter_const_filter,
                      firfilter_const_signal,
                      correlate,
                      convolve,
                      convolve_const_x,
                      convolve_const_y)

from numpy.testing import assert_array_almost_equal


class TestFirFilter:
    b = np.array([0, 1, -2])
    x = np.array([1.2, 0.4, -0.1, 1.2, 1.1, 0.9, -0.7, -0.3, 0.1, -0.2, 0.3])
    x2 = np.array([[1.2, 0.4, -0.1, 1.2, 1.1, 0.9,
                    -0.7, -0.3, 0.1, -0.2, 0.3],
                   [0.7, 1.1, -0.2, -0.3, 0.1, 0.2,
                    -0.2, -0.7, 1.2, 1.0, 0.8]])
    x3 = np.array([[[1.2, 0.7],
                    [0.4, 1.1],
                    [-0.1, -0.2],
                    [1.2, -0.3],
                    [1.1, 0.1],
                    [0.9, 0.2],
                    [-0.7, -0.2],
                    [-0.3, -0.7],
                    [0.1, 1.2],
                    [-0.2, 1.0],
                    [0.3, 0.8]],
                   [[0.7, 1.2],
                    [1.1, 0.4],
                    [-0.2, -0.1],
                    [-0.3, 1.2],
                    [0.1, 1.1],
                    [0.2, 0.9],
                    [-0.2, -0.7],
                    [-0.7, -0.3],
                    [1.2, 0.1],
                    [1.0, -0.2],
                    [0.3, 0.8]]])

    xfb = np.array([0., 1.2, -2., -0.9, 1.4, -1.3, -1.3, -2.5, 1.1, 0.7, -0.4])
    x2fb = np.array([[0., 1.2, -2., -0.9, 1.4, -1.3,
                      -1.3, -2.5, 1.1, 0.7, -0.4],
                     [0., 0.7, -0.3, -2.4, 0.1, 0.7,
                      0., -0.6, -0.3, 2.6, -1.4]])
    x3fb = np.array([[[0., 0.],
                      [1.2, 0.7],
                      [-2., -0.3],
                      [-0.9, -2.4],
                      [1.4, 0.1],
                      [-1.3, 0.7],
                      [-1.3, 0.],
                      [-2.5, -0.6],
                      [1.1, -0.3],
                      [0.7, 2.6],
                      [-0.4, -1.4]],
                     [[0., 0.],
                      [0.7, 1.2],
                      [-0.3, -2.],
                      [-2.4, -0.9],
                      [0.1, 1.4],
                      [0.7, -1.3],
                      [0., -1.3],
                      [-0.6, -2.5],
                      [-0.3, 1.1],
                      [2.6, 0.7],
                      [-1.4, -0.4]]])

    def test_firfilter_1d_input(self):
        my_firfilter = firfilter(self.b, self.x)
        assert_array_almost_equal(my_firfilter(self.b, self.x), self.xfb)

    def test_firfilter_1d_input_filter_length(self):
        my_firfilter = firfilter(len(self.b), self.x)
        assert_array_almost_equal(my_firfilter(self.b, self.x), self.xfb)

    def test_firfilter_1d_input_signal_length(self):
        my_firfilter = firfilter(self.b, len(self.x))
        assert_array_almost_equal(my_firfilter(self.b, self.x), self.xfb)

    def test_firfilter_1d_input_filter_length_error(self):
        my_firfilter = firfilter(len(self.b) + 2, self.x)
        with pytest.raises(ValueError):
          assert_array_almost_equal(my_firfilter(self.b, self.x), self.xfb)

    def test_firfilter_1d_input_signal_length_error(self):
        my_firfilter = firfilter(self.b, len(self.x) + 2)
        with pytest.raises(ValueError):
            assert_array_almost_equal(my_firfilter(self.b, self.x), self.xfb)

    def test_firfilter_1d_input_x_constant(self):
        my_firfilter = firfilter_const_signal(self.b, self.x)
        assert_array_almost_equal(my_firfilter(self.b), self.xfb)

    def test_firfilter_1d_input_b_constant(self):
        my_firfilter = firfilter_const_filter(self.b, self.x)
        assert_array_almost_equal(my_firfilter(self.x), self.xfb)

    def test_firfilter_2d_input_default_axis(self):
        my_firfilter = firfilter(self.b, self.x2)
        assert_array_almost_equal(my_firfilter(self.b, self.x2), self.x2fb)
        assert_array_almost_equal(my_firfilter(self.b, self.x2), self.x2fb)

    def test_firfilter_2d_input_alternative_axis(self):
        my_firfilter = firfilter(self.b, self.x2.T, axis=0)
        assert_array_almost_equal(my_firfilter(self.b, self.x2.T), self.x2fb.T)

    def test_firfilter_2d_input_x_constant(self):
        my_firfilter = firfilter_const_signal(self.b, self.x2)
        assert_array_almost_equal(my_firfilter(self.b), self.x2fb)
        assert_array_almost_equal(my_firfilter(self.b), self.x2fb)

    def test_firfilter_2d_input_b_constant(self):
        my_firfilter = firfilter_const_filter(self.b, self.x2)
        assert_array_almost_equal(my_firfilter(self.x2), self.x2fb)
        assert_array_almost_equal(my_firfilter(self.x2), self.x2fb)

    def test_firfilter_3d_input_alternative_axis(self):
        my_firfilter = firfilter(self.b, self.x3, axis=1)
        assert_array_almost_equal(my_firfilter(self.b, self.x3), self.x3fb)


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

    x2 = np.array([[1.2, 0.3, -0.2, 0.7, 0.3, 0.1, 0.1],
                   [0.8, -0.1, -0.2, -0.1, 0.2, 0.3, 0.1]])
    y2 = np.array([[1.1, 0.7, 0.1, -0.2, -0.6],
                   [0.6, 0.4, 0.3, 0.2, -0.1]])

    x3 = np.array([[0.1, 0.5, -0.2, 0.4, 0.9],
                   [-0.4, -0.1, 0.2, 0.2, 0.1],
                   [0.5, 0.3, -0.2, 0.4, 0.6],
                   [0.1, 0.1, 0.2, 0.1, 0.1],
                   [0.3, -0.2, 0.1, -0.2, 0.3]])
    y3 = np.array([[0.7, 0.8, 0.1, -0.1],
                   [0.2, -0.2, 0.2, -0.3],
                   [-0.1, -0.7, 0.9, 1.2]])

    cxy = np.array([1.32, 1.17, 0.11, 0.42, 0.02, 0.25, 0.19, -0.4, -0.19,
                    -0.08, -0.06])

    cx2y = np.array([[1.32, 1.17, 0.11, 0.42, 0.02, 0.25, 0.19, -0.4, -0.19,
                      -0.08, -0.06],
                     [0.88, 0.45, -0.21, -0.42, -0.33, 0.56, 0.48, 0.12, -0.17,
                      -0.2, -0.06]])

    cxy2 = np.array([[1.32, 1.17, 0.11, 0.42, 0.02, 0.25, 0.19, -0.4, -0.19,
                      -0.08, -0.06],
                     [0.72, 0.66, 0.36, 0.67, 0.34, 0.32, 0.35, 0.06, 0.02,
                      0.01, -0.01]])

    cx2y2 = np.array([[1.32, 1.17, 0.11, 0.42, 0.02, 0.25,
                       0.19, -0.4, -0.19, -0.08, -0.06],
                      [0.48, 0.26, 0.08, -0.01, -0.08, 0.2,
                       0.24, 0.18, 0.07, -0.01, -0.01]])

    cx3y3 = np.array([[0.07, 0.43, 0.27, 0.16, 0.88, 0.78, 0.05, -0.09],
                      [-0.26, -0.31, -0.1, 0.52, 0.17, 0.04, 0.05, -0.28],
                      [0.26, 0.55, -0.11, 0.87, 0.79, -0.01, 1.27, 0.99],
                      [0.21, 0.4, -0.08, -0.47, -0.02, 0.44, 0.33, -0.07],
                      [0.18, -0.28, 0.24, 0.83, -0.07, -0.15, 1.06, 0.66],
                      [0.05, -0.18, 0.12, -0.13, 0.4, 0.13, 0.33, 0.03],
                      [-0.03, -0.19, 0.4, 0.13, -0.04, -0.27, 0.03, 0.36]])

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
        my_convolve_1 = convolve_const_x(self.x, self.y)
        my_convolve_2 = convolve_const_x(self.x, self.y, 'same')
        my_convolve_3 = convolve_const_x(self.x, self.y, 'valid')
        assert_array_almost_equal(my_convolve_1(self.y),
                                  self.cxy)
        assert_array_almost_equal(my_convolve_2(self.y),
                                  self.cxy[2:9])
        assert_array_almost_equal(my_convolve_3(self.y),
                                  self.cxy[4:7])

    def test_convolve_1d_y_constant(self):
        my_convolve_1 = convolve_const_y(self.x, self.y)
        my_convolve_2 = convolve_const_y(self.x, self.y, 'same')
        my_convolve_3 = convolve_const_y(self.x, self.y, 'valid')
        assert_array_almost_equal(my_convolve_1(self.x),
                                  self.cxy)
        assert_array_almost_equal(my_convolve_2(self.x),
                                  self.cxy[2:9])
        assert_array_almost_equal(my_convolve_3(self.x),
                                  self.cxy[4:7])

    def test_convolve_single_2d(self):
        my_convolve_1 = convolve(self.x2, self.y2, axis=-1)
        my_convolve_2 = convolve(self.x2, self.y2, 'same', axis=-1)
        my_convolve_3 = convolve(self.x2, self.y2, 'valid', axis=-1)
        assert_array_almost_equal(my_convolve_1(self.x2, self.y2),
                                  self.cx2y2)
        assert_array_almost_equal(my_convolve_2(self.x2, self.y2),
                                  self.cx2y2[:, 2:9])
        assert_array_almost_equal(my_convolve_3(self.x2, self.y2),
                                  self.cx2y2[:, 4:7])

    def test_convolve_single_2d_x_constant(self):
        my_convolve_1 = convolve_const_x(self.x2, self.y2, axis=-1)
        my_convolve_2 = convolve_const_x(self.x2, self.y2, 'same', axis=-1)
        my_convolve_3 = convolve_const_x(self.x2, self.y2, 'valid', axis=-1)
        assert_array_almost_equal(my_convolve_1(self.y2), self.cx2y2)
        assert_array_almost_equal(my_convolve_2(self.y2), self.cx2y2[:, 2:9])
        assert_array_almost_equal(my_convolve_3(self.y2), self.cx2y2[:, 4:7])

    def test_convolve_single_2d_y_constant(self):
        my_convolve_1 = convolve_const_y(self.x2, self.y2, axis=-1)
        my_convolve_2 = convolve_const_y(self.x2, self.y2, 'same', axis=-1)
        my_convolve_3 = convolve_const_y(self.x2, self.y2, 'valid', axis=-1)
        assert_array_almost_equal(my_convolve_1(self.x2), self.cx2y2)
        assert_array_almost_equal(my_convolve_2(self.x2), self.cx2y2[:, 2:9])
        assert_array_almost_equal(my_convolve_3(self.x2), self.cx2y2[:, 4:7])

    def test_convolve_multiple_2d(self):
        my_convolve_1 = convolve(self.x3, self.y3)
        my_convolve_2 = convolve(self.x3, self.y3, 'same')
        my_convolve_3 = convolve(self.x3, self.y3, 'valid')
        assert_array_almost_equal(my_convolve_1(self.x3, self.y3), self.cx3y3)
        assert_array_almost_equal(my_convolve_2(self.x3, self.y3),
                                  self.cx3y3[1:6, 1:6])
        assert_array_almost_equal(my_convolve_3(self.x3, self.y3),
                                  self.cx3y3[2:5, 3:5])

    def test_convolve_multiple_2d_x_constant(self):
        my_convolve_1 = convolve_const_x(self.x3, self.y3)
        my_convolve_2 = convolve_const_x(self.x3, self.y3, 'same',)
        my_convolve_3 = convolve_const_x(self.x3, self.y3, 'valid')
        assert_array_almost_equal(my_convolve_1(self.y3), self.cx3y3)
        assert_array_almost_equal(my_convolve_2(self.y3), self.cx3y3[1:6, 1:6])
        assert_array_almost_equal(my_convolve_3(self.y3), self.cx3y3[2:5, 3:5])

    def test_convolve_multiple_2d_y_constant(self):
        my_convolve_1 = convolve_const_y(self.x3, self.y3)
        my_convolve_2 = convolve_const_y(self.x3, self.y3, 'same')
        my_convolve_3 = convolve_const_y(self.x3, self.y3, 'valid')
        assert_array_almost_equal(my_convolve_1(self.x3), self.cx3y3)
        assert_array_almost_equal(my_convolve_2(self.x3), self.cx3y3[1:6, 1:6])
        assert_array_almost_equal(my_convolve_3(self.x3), self.cx3y3[2:5, 3:5])

    def test_convolve_single_2d_1d(self):
        my_convolve_1 = convolve(self.x2, self.y)
        assert_array_almost_equal(my_convolve_1(self.x2, self.y), self.cx2y)

    def test_convolve_1d_single_2d(self):
        my_convolve_1 = convolve(self.x, self.y2)
        assert_array_almost_equal(my_convolve_1(self.x, self.y2), self.cxy2)
