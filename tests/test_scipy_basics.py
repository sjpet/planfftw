#!usr/bin/env python
import numpy as np

from planfftw._scipy_basics import (
    fft, fft_pair, rfft, rfft_pair, fftn, fftn_pair, rfftn, rfftn_pair,
    ifft, ifft_pair, irfft, irfft_pair, ifftn, ifftn_pair, irfftn, irfftn_pair)
# noinspection PyProtectedMember
from planfftw._utils import pad_array

from numpy.testing import assert_array_almost_equal


class TestFFT:
    # Test arrays
    x = np.array([0.22611457, 0.59476972, -0.50287775, 1.49945174,
                  -0.22935349, -0.54408391, 0.12970366, 0.26661302,
                  0.4242425, 0.13780873, -0.01282802, -0.02088417,
                  -0.97447331, -0.4746792])

    x2 = np.array([[-0.30102051, -2.43414466, 0.97140896, 0.21493837,
                    1.63241865],
                   [1.26844792, 0.64110062, 0.03199621, 1.25124704,
                    -0.14500464],
                   [-0.41356482, 0.48176488, 1.32648019, -0.94751586,
                    -1.72374966],
                   [-0.63770105, 1.69398695, -0.22669155, 0.87382129,
                    0.34183466]])

    def test_fft_1d(self):
        my_fft = fft(self.x)
        assert_array_almost_equal(my_fft(self.x), np.fft.fft(self.x))

    def test_fft_nfft(self):
        my_fft = fft(self.x, nfft=32)
        assert_array_almost_equal(my_fft(self.x), np.fft.fft(self.x, n=32))

    def test_fft_nfft_small(self):
        my_fft = fft(self.x, nfft=7)
        assert_array_almost_equal(my_fft(self.x), np.fft.fft(self.x, n=7))

    def test_fft_2d_default(self):
        my_fft = fft(self.x2)
        assert_array_almost_equal(my_fft(self.x2), np.fft.fft(self.x2))

    def test_fft_2d_axis(self):
        my_fft = fft(self.x2, axis=0)
        assert_array_almost_equal(my_fft(self.x2), np.fft.fft(self.x2, axis=0))

    def test_fft_2d_nfft(self):     # No additional coverage for this, keep?
        my_fft = fft(self.x2, nfft=32)
        assert_array_almost_equal(my_fft(self.x2), np.fft.fft(self.x2, n=32))

    def test_fft_pair(self):
        my_fft, my_ifft = fft_pair(self.x, nfft=32)
        x_fft = my_fft(self.x)
        assert_array_almost_equal(my_ifft(x_fft), pad_array(self.x, (32,)))

    def test_fft_pair_cropped(self):
        my_fft, my_ifft = fft_pair(self.x, nfft=32, crop_ifft=True)
        x_fft = my_fft(self.x)
        assert_array_almost_equal(my_ifft(x_fft), self.x)


class TestRFFT:
    # Test arrays
    x = np.array([0.22611457, 0.59476972, -0.50287775, 1.49945174,
                  -0.22935349, -0.54408391, 0.12970366, 0.26661302,
                  0.4242425, 0.13780873, -0.01282802, -0.02088417,
                  -0.97447331, -0.4746792])

    x2 = np.array([[-0.30102051, -2.43414466, 0.97140896, 0.21493837,
                    1.63241865],
                   [1.26844792, 0.64110062, 0.03199621, 1.25124704,
                    -0.14500464],
                   [-0.41356482, 0.48176488, 1.32648019, -0.94751586,
                    -1.72374966],
                   [-0.63770105, 1.69398695, -0.22669155, 0.87382129,
                    0.34183466]])

    def test_fft_1d(self):
        my_rfft = rfft(self.x)
        assert_array_almost_equal(my_rfft(self.x), np.fft.rfft(self.x))

    def test_fft_nfft(self):
        my_rfft = rfft(self.x, nfft=32)
        assert_array_almost_equal(my_rfft(self.x), np.fft.rfft(self.x, n=32))

    def test_fft_2d_default(self):
        my_rfft = rfft(self.x2)
        assert_array_almost_equal(my_rfft(self.x2), np.fft.rfft(self.x2))

    def test_fft_2d_axis(self):
        my_rfft = rfft(self.x2, axis=0)
        assert_array_almost_equal(my_rfft(self.x2),
                                  np.fft.rfft(self.x2, axis=0))

    def test_fft_2d_nfft(self):
        my_rfft = rfft(self.x2, nfft=32)
        assert_array_almost_equal(my_rfft(self.x2), np.fft.rfft(self.x2, n=32))

    def test_fft_pair(self):
        my_rfft, my_irfft = rfft_pair(self.x, nfft=32)
        x_fft = my_rfft(self.x)
        assert_array_almost_equal(my_irfft(x_fft), pad_array(self.x, (32,)))

    def test_fft_pair_cropped(self):
        my_rfft, my_irfft = rfft_pair(self.x, nfft=32, crop_ifft=True)
        x_fft = my_rfft(self.x)
        assert_array_almost_equal(my_irfft(x_fft), self.x)


class TestFFTN:
    # Test arrays
    x2 = np.array([[-0.30102051, -2.43414466, 0.97140896, 0.21493837,
                    1.63241865],
                   [1.26844792, 0.64110062, 0.03199621, 1.25124704,
                    -0.14500464],
                   [-0.41356482, 0.48176488, 1.32648019, -0.94751586,
                    -1.72374966],
                   [-0.63770105, 1.69398695, -0.22669155, 0.87382129,
                    0.34183466]])

    x3 = np.array([[[-0.1923832, 0.03131112, -0.8529837, -0.75595146,
                     0.56668188, 0.96283906, -0.61629368, 1.42770147,
                     -0.46705707, 0.50140555, 1.15011872, -1.32424782],
                    [1.57015182, -1.20920907, 0.44095057, 0.27611523,
                     1.28197664, 0.43592991, 0.09304247, -0.64853008,
                     0.2627155, -1.32809712, 0.30022486,  1.54736705],
                    [1.30578254, -0.25293877, -0.46174017, -0.99456879,
                     1.62086393, 0.55958295, 0.79784967, -0.51324102,
                     1.66667799, -0.38689949, 0.33981117, -0.38695571]],
                   [[0.31054434, -0.01583677, 0.4003227, -0.7961732,
                     0.44648656, -1.05080412, 0.80119623, 0.89568846,
                     0.34777738, 1.24777642, -0.27926125, 1.33109184],
                    [-0.95919969, -1.91818637, -0.29423917, 1.16631166,
                     -0.99130333, 1.73340527, -0.19555054, 1.10684304,
                     1.14000992,  0.04287957, -1.10419989, -0.82061708],
                    [1.96729441, 0.35547018, -0.04883976, -0.57180173,
                     0.50852922, -0.03722728, 0.83683813, 1.19639478,
                     -0.61560698, -0.20976715, -1.79322839, 0.65665904]],
                   [[-0.93146716, -0.89127933, -0.10022056, -2.06647064,
                     0.05214158, -1.31893485, 0.26438599, -0.44371855,
                     0.82696167, 1.20113261, 0.68363415, 1.70728545],
                    [0.80653463, -0.28183967, -1.10717413, -0.68224127,
                     0.86282679, 0.21007209, -1.44285538,  0.17650003,
                     -0.57186417, -0.7667556, 1.97725472, 1.53209004],
                    [-0.99411145, 0.61434073, 0.20331582, 0.95688453,
                     0.95240977, 0.12827774, -0.3977226, 0.78752786,
                     0.91100169, 0.46833489, 0.07616078, -1.75934211]],
                   [[-0.86855304, -0.71715145, 1.69391834, -0.45096056,
                     2.15298798, -0.63319347, -0.63367794, -0.41107065,
                     -0.36684979, 0.38494642, 1.66722248,  0.12049035],
                    [-0.15588067, -1.65856522,  0.998639, -0.06812809,
                     -1.12948693, -0.00638995, -0.54972101,  0.50255547,
                     -0.79749639, 0.68597091, 0.85827738, -0.00805226],
                    [-0.82823101, 1.08138927, 2.42707403, -1.13963788,
                     -1.15061122, -0.25227279, 1.59960476, 0.8196863,
                     0.37633217, 0.90736547, -0.55975426, 0.89341611]]])

    def test_fftn_2d(self):
        my_fftn = fftn(self.x2)
        assert_array_almost_equal(my_fftn(self.x2), np.fft.fftn(self.x2))

    def test_fftn_3d_larger_shape(self):
        test_shape = (4, 6, 14)
        my_fftn = fftn(self.x3, shape=test_shape)
        assert_array_almost_equal(my_fftn(self.x3), np.fft.fftn(self.x3,
                                                                s=test_shape))

    def test_fftn_3d_smaller_shape(self):
        test_shape = (3, 3, 6)
        my_fftn = fftn(self.x3, shape=test_shape)
        assert_array_almost_equal(my_fftn(self.x3), np.fft.fftn(self.x3,
                                                                s=test_shape))

    def test_fftn_3d_mixed_and_short_shape(self):
        test_shape = (2, 16)
        my_fftn = fftn(self.x3, shape=test_shape)
        assert_array_almost_equal(my_fftn(self.x3), np.fft.fftn(self.x3,
                                                                s=test_shape))

    def test_fftn_3d_axes(self):
        test_axes = (0, -1)
        my_fftn = fftn(self.x3, axes=test_axes)
        assert_array_almost_equal(my_fftn(self.x3), np.fft.fftn(self.x3,
                                                                axes=test_axes))

    def test_fftn_pair(self):
        test_axes = (0, -1)
        test_shape = (4, 16)
        my_fftn, my_ifftn = fftn_pair(self.x3,
                                      shape=test_shape,
                                      axes=test_axes)
        x_fft = my_fftn(self.x3)
        assert_array_almost_equal(my_ifftn(x_fft),
                                  pad_array(self.x3, (4, 3, 16)))

    def test_fftn_pair_cropped(self):
        test_axes = (0, -1)
        test_shape = (4, 16)
        my_fftn, my_ifftn = fftn_pair(self.x3,
                                      shape=test_shape,
                                      axes=test_axes,
                                      crop_ifft=True)
        x_fft = my_fftn(self.x3)
        assert_array_almost_equal(my_ifftn(x_fft), self.x3)


class TestRFFTN:
    # Test arrays
    x2 = np.array([[-0.30102051, -2.43414466, 0.97140896, 0.21493837,
                    1.63241865],
                   [1.26844792, 0.64110062, 0.03199621, 1.25124704,
                    -0.14500464],
                   [-0.41356482, 0.48176488, 1.32648019, -0.94751586,
                    -1.72374966],
                   [-0.63770105, 1.69398695, -0.22669155, 0.87382129,
                    0.34183466]])

    x3 = np.array([[[-0.1923832, 0.03131112, -0.8529837, -0.75595146,
                     0.56668188, 0.96283906, -0.61629368, 1.42770147,
                     -0.46705707, 0.50140555, 1.15011872, -1.32424782],
                    [1.57015182, -1.20920907, 0.44095057, 0.27611523,
                     1.28197664, 0.43592991, 0.09304247, -0.64853008,
                     0.2627155, -1.32809712, 0.30022486,  1.54736705],
                    [1.30578254, -0.25293877, -0.46174017, -0.99456879,
                     1.62086393, 0.55958295, 0.79784967, -0.51324102,
                     1.66667799, -0.38689949, 0.33981117, -0.38695571]],
                   [[0.31054434, -0.01583677, 0.4003227, -0.7961732,
                     0.44648656, -1.05080412, 0.80119623, 0.89568846,
                     0.34777738, 1.24777642, -0.27926125, 1.33109184],
                    [-0.95919969, -1.91818637, -0.29423917, 1.16631166,
                     -0.99130333, 1.73340527, -0.19555054, 1.10684304,
                     1.14000992,  0.04287957, -1.10419989, -0.82061708],
                    [1.96729441, 0.35547018, -0.04883976, -0.57180173,
                     0.50852922, -0.03722728, 0.83683813, 1.19639478,
                     -0.61560698, -0.20976715, -1.79322839, 0.65665904]],
                   [[-0.93146716, -0.89127933, -0.10022056, -2.06647064,
                     0.05214158, -1.31893485, 0.26438599, -0.44371855,
                     0.82696167, 1.20113261, 0.68363415, 1.70728545],
                    [0.80653463, -0.28183967, -1.10717413, -0.68224127,
                     0.86282679, 0.21007209, -1.44285538,  0.17650003,
                     -0.57186417, -0.7667556, 1.97725472, 1.53209004],
                    [-0.99411145, 0.61434073, 0.20331582, 0.95688453,
                     0.95240977, 0.12827774, -0.3977226, 0.78752786,
                     0.91100169, 0.46833489, 0.07616078, -1.75934211]],
                   [[-0.86855304, -0.71715145, 1.69391834, -0.45096056,
                     2.15298798, -0.63319347, -0.63367794, -0.41107065,
                     -0.36684979, 0.38494642, 1.66722248,  0.12049035],
                    [-0.15588067, -1.65856522,  0.998639, -0.06812809,
                     -1.12948693, -0.00638995, -0.54972101,  0.50255547,
                     -0.79749639, 0.68597091, 0.85827738, -0.00805226],
                    [-0.82823101, 1.08138927, 2.42707403, -1.13963788,
                     -1.15061122, -0.25227279, 1.59960476, 0.8196863,
                     0.37633217, 0.90736547, -0.55975426, 0.89341611]]])

    def test_rfftn_2d(self):
        my_rfftn = rfftn(self.x2)
        assert_array_almost_equal(my_rfftn(self.x2), np.fft.rfftn(self.x2))

    def test_rfftn_3d_larger_shape(self):
        test_shape = (4, 6, 14)
        my_rfftn = rfftn(self.x3, shape=test_shape)
        assert_array_almost_equal(my_rfftn(self.x3), np.fft.rfftn(self.x3,
                                                                  s=test_shape))

    def test_rfftn_3d_smaller_shape(self):
        test_shape = (3, 3, 6)
        my_rfftn = rfftn(self.x3, shape=test_shape)
        assert_array_almost_equal(my_rfftn(self.x3), np.fft.rfftn(self.x3,
                                                                  s=test_shape))

    def test_rfftn_3d_mixed_and_short_shape(self):
        test_shape = (2, 16)
        my_rfftn = rfftn(self.x3, shape=test_shape)
        assert_array_almost_equal(my_rfftn(self.x3), np.fft.rfftn(self.x3,
                                                                  s=test_shape))

    def test_rfftn_3d_axes(self):
        test_axes = (0, -1)
        my_rfftn = rfftn(self.x3, axes=test_axes)
        assert_array_almost_equal(my_rfftn(self.x3),
                                  np.fft.rfftn(self.x3, axes=test_axes))

    def test_rfftn_pair(self):
        test_axes = (0, -1)
        test_shape = (4, 16)
        my_rfftn, my_irfftn = rfftn_pair(self.x3,
                                         shape=test_shape,
                                         axes=test_axes)
        x_fft = my_rfftn(self.x3)
        assert_array_almost_equal(my_irfftn(x_fft),
                                  pad_array(self.x3, (4, 3, 16)))

    def test_rfftn_pair_cropped(self):
        test_axes = (0, -1)
        test_shape = (4, 16)
        my_rfftn, my_irfftn = rfftn_pair(self.x3,
                                         shape=test_shape,
                                         axes=test_axes,
                                         crop_ifft=True)
        x_fft = my_rfftn(self.x3)
        assert_array_almost_equal(my_irfftn(x_fft), self.x3)


class TestIFFT:
    x_fft = np.array([0.24168006-0.24528664j, -0.49631226+0.71002236j,
                      -1.29084773-0.04870652j, -0.00571143-0.37720967j,
                      -1.44095505+0.21357669j, 1.49478589-0.95366282j,
                      0.90648661-0.88762462j, -0.53105531+0.72219144j,
                      0.04839991-1.63086726j,  0.19211264-1.10039169j,
                      0.62725381-1.10119838j, -2.61622115-0.63605559j])

    x_fft_2 = np.array([[1.69809800-0.05301582j, -0.92526638+0.3591387j,
                         -0.63561789+0.4974533j, 0.92796619+0.90261579j,
                         1.45776989+0.69573542j, -0.43802622+0.21699345j,
                         -0.81690054-0.63575929j, 0.26419249-0.14937098j,
                         0.36107871-0.30456336j, -0.94954403-0.20955856j,
                         0.55329032+0.8345774j, -0.64080383-1.6870927j],
                        [-1.06416853+0.29846147j, 0.79009204+0.32151942j,
                         -0.58874680+0.24897117j, 0.23134225-0.72163099j,
                         -0.68317960-1.13724498j, -0.26538413-0.60702582j,
                         -0.51769258-1.59573457j, -1.11594964+0.56722895j,
                         1.84674370+0.48927262j, 0.82430047-0.1388641j,
                         -0.08531134+0.71194645j, -0.85954264-0.62644226j]])

    def test_ifft_1d(self):
        my_ifft = ifft(self.x_fft)
        assert_array_almost_equal(my_ifft(self.x_fft), np.fft.ifft(self.x_fft))

    def test_ifft_1d_nfft(self):
        my_ifft = ifft(self.x_fft, nfft=16)
        assert_array_almost_equal(my_ifft(self.x_fft),
                                  np.fft.ifft(self.x_fft, n=16))

    def test_ifft_2d_default(self):
        my_ifft = ifft(self.x_fft_2)
        assert_array_almost_equal(my_ifft(self.x_fft_2),
                                  np.fft.ifft(self.x_fft_2))

    def test_ifft_2d_axis(self):
        my_ifft = ifft(self.x_fft_2, axis=0)
        assert_array_almost_equal(my_ifft(self.x_fft_2),
                                  np.fft.ifft(self.x_fft_2, axis=0))

    def test_ifft_2d_nfft(self):    # No additional coverage for this, keep?
        my_ifft = ifft(self.x_fft_2, nfft=16)
        assert_array_almost_equal(my_ifft(self.x_fft_2),
                                  np.fft.ifft(self.x_fft_2, n=16))

    def test_ifft_pair(self):
        my_ifft, my_fft = ifft_pair(self.x_fft)
        my_x = my_ifft(self.x_fft)
        assert_array_almost_equal(my_fft(my_x), self.x_fft)

    def test_ifft_pair_cropped(self):
        my_ifft, my_fft = ifft_pair(self.x_fft,
                                    nfft=32,
                                    crop_fft=True)
        my_x = my_ifft(self.x_fft)
        assert_array_almost_equal(my_fft(my_x), self.x_fft)


class TestIRFFT:
    x_fft = np.array([-0.65715380+0.j, -4.08505860-5.99199174j,
                      3.75294043-1.40104244j, -3.82119351-3.68866147j,
                      -1.60547583-0.96120687j,  5.41874806-0.0817256j,
                      5.35323892+5.519293j, -3.36411099+0.87371896j,
                      2.93663827-6.61944528j, 0.50964679+1.61343887j,
                      -0.02985507+0.94478302j, -3.48530695+0.j])

    x_fft_2 = np.array([[1.69809800-0.05301582j, -0.92526638+0.3591387j,
                         -0.63561789+0.4974533j, 0.92796619+0.90261579j,
                         1.45776989+0.69573542j, -0.43802622+0.21699345j,
                         -0.81690054-0.63575929j, 0.26419249-0.14937098j,
                         0.36107871-0.30456336j, -0.94954403-0.20955856j,
                         0.55329032+0.8345774j, -0.64080383-1.6870927j],
                        [-1.06416853+0.29846147j, 0.79009204+0.32151942j,
                         -0.58874680+0.24897117j, 0.23134225-0.72163099j,
                         -0.68317960-1.13724498j, -0.26538413-0.60702582j,
                         -0.51769258-1.59573457j, -1.11594964+0.56722895j,
                         1.84674370+0.48927262j, 0.82430047-0.1388641j,
                         -0.08531134+0.71194645j, -0.85954264-0.62644226j]])

    def test_irfft_1d(self):
        my_irfft = irfft(self.x_fft)
        assert_array_almost_equal(my_irfft(self.x_fft),
                                  np.fft.irfft(self.x_fft))

    def test_irfft_1d_nfft(self):
        my_irfft = irfft(self.x_fft, nfft=16)
        assert_array_almost_equal(my_irfft(self.x_fft),
                                  np.fft.irfft(self.x_fft, n=16))

    def test_irfft_2d_default(self):
        my_irfft = irfft(self.x_fft_2)
        assert_array_almost_equal(my_irfft(self.x_fft_2),
                                  np.fft.irfft(self.x_fft_2))

    def test_irfft_2d_axis(self):
        my_irfft = irfft(self.x_fft_2, axis=0)
        assert_array_almost_equal(my_irfft(self.x_fft_2),
                                  np.fft.irfft(self.x_fft_2, axis=0))

    def test_irfft_2d_nfft(self):    # No additional coverage for this, keep?
        my_irfft = irfft(self.x_fft_2, nfft=16)
        assert_array_almost_equal(my_irfft(self.x_fft_2),
                                  np.fft.irfft(self.x_fft_2, n=16))

    def test_irfft_pair(self):
        my_irfft, my_rfft = irfft_pair(self.x_fft)
        my_x = my_irfft(self.x_fft)
        assert_array_almost_equal(my_rfft(my_x), self.x_fft)


class TestIFFTN:
    # Test arrays
    x2_fft = np.array([[3.90005299+0.j, -2.76104255-0.68138194j,
                        0.60141991+0.51297946j, 0.60141991-0.51297946j,
                        -2.76104255+0.68138194j],
                       [1.36018608-1.00253685j, 0.20354034+5.94135865j,
                        0.22565011-1.10500067j, -0.21392501-5.59176544j,
                        -1.01272997-7.77280054j],
                       [-6.28602191+0.j, -2.46393831+0.65877986j,
                        2.24361876+7.43910467j, 2.24361876-7.43910467j,
                        -2.46393831-0.65877986j],
                       [1.36018608+1.00253685j, -1.01272997+7.77280054j,
                        -0.21392501+5.59176544j, 0.22565011+1.10500067j,
                        0.20354034-5.94135865j]])

    x3_fft = np.array([[[0.12475658 + 0.00000000e+00j,
                         -0.04322924 - 8.34514337e-02j,
                         -0.00803911 - 4.00823389e-02j,
                         0.01211255 - 3.22348481e-02j,
                         -0.07061529 - 1.02749659e-02j,
                         0.04097889 - 1.12844137e-01j,
                         0.09870128 + 0.00000000e+00j,
                         0.04097889 + 1.12844137e-01j,
                         -0.07061529 + 1.02749659e-02j,
                         0.01211255 + 3.22348481e-02j,
                         -0.00803911 + 4.00823389e-02j,
                         -0.04322924 + 8.34514337e-02j],
                        [-0.01038341 - 6.22294334e-02j,
                         0.03919708 - 4.26961046e-02j,
                         0.04971317 - 5.91551758e-02j,
                         0.03716097 + 1.28392755e-02j,
                         0.00197185 + 3.81883403e-02j,
                         -0.07645664 + 8.04848046e-03j,
                         0.02481349 - 2.73328359e-02j,
                         -0.00201960 + 1.36099992e-01j,
                         -0.06461842 - 6.30059597e-02j,
                         -0.09721290 - 2.91569579e-02j,
                         -0.15121174 - 1.12452693e-02j,
                         -0.00412296 + 8.59964818e-02j],
                        [-0.01038341 + 6.22294334e-02j,
                         -0.00412296 - 8.59964818e-02j,
                         -0.15121174 + 1.12452693e-02j,
                         -0.09721290 + 2.91569579e-02j,
                         -0.06461842 + 6.30059597e-02j,
                         -0.00201960 - 1.36099992e-01j,
                         0.02481349 + 2.73328359e-02j,
                         -0.07645664 - 8.04848046e-03j,
                         0.00197185 - 3.81883403e-02j,
                         0.03716097 - 1.28392755e-02j,
                         0.04971317 + 5.91551758e-02j,
                         0.03919708 + 4.26961046e-02j]],
                       [[0.03545088 + 3.80874306e-05j,
                         -0.03038947 - 5.35959808e-03j,
                         0.03682702 + 5.15325338e-02j,
                         0.05706433 + 1.08557413e-01j,
                         0.03051096 + 3.63480254e-02j,
                         0.01424887 + 1.34623052e-02j,
                         0.05809326 - 5.91523095e-02j,
                         0.05655769 - 1.92107129e-02j,
                         0.01434035 + 6.27232848e-02j,
                         0.00775200 + 8.72879406e-02j,
                         0.06033527 + 8.34096192e-02j,
                         -0.02390823 - 9.53612739e-02j],
                        [-0.01566024 + 2.34877828e-02j,
                         0.02173232 + 3.28055745e-02j,
                         -0.05994566 - 2.46692498e-02j,
                         0.02853584 - 3.58611264e-02j,
                         0.03295335 + 1.30864786e-01j,
                         0.10094931 - 1.45540820e-02j,
                         -0.04881496 - 4.07956229e-02j,
                         0.05858073 - 3.21025288e-02j,
                         -0.01858250 - 6.98208695e-02j,
                         0.00776778 - 4.84519209e-02j,
                         0.07297952 + 1.37016018e-02j,
                         0.01317275 - 3.02260581e-02j],
                        [0.01036957 + 1.19053781e-02j,
                         0.01489478 + 6.02354214e-02j,
                         -0.03754497 + 9.93804754e-02j,
                         -0.04076314 + 1.78591240e-02j,
                         -0.03034278 + 3.14993895e-02j,
                         0.05319896 + 3.06022830e-02j,
                         -0.08974488 - 2.89923546e-03j,
                         0.02019499 - 7.25981881e-02j,
                         -0.02971482 - 3.13215582e-02j,
                         -0.01341666 - 4.63735830e-02j,
                         -0.03920884 + 1.05143244e-02j,
                         -0.14370240 + 1.73169120e-02j]],
                       [[-0.00821381 + 0.00000000e+00j,
                         0.01680197 - 2.36852026e-02j,
                         -0.00476679 - 1.11602740e-01j,
                         0.10152752 + 3.00274667e-02j,
                         -0.05670188 + 7.78271422e-02j,
                         -0.00877083 + 9.46336734e-02j,
                         0.08691157 + 0.00000000e+00j,
                         -0.00877083 - 9.46336734e-02j,
                         -0.05670188 - 7.78271422e-02j,
                         0.10152752 - 3.00274667e-02j,
                         -0.00476679 + 1.11602740e-01j,
                         0.01680197 + 2.36852026e-02j],
                        [-0.06008399 + 4.41136724e-02j,
                         -0.04840379 + 4.73925735e-02j,
                         0.06179666 + 1.42935800e-01j,
                         -0.00960846 + 1.72447007e-02j,
                         0.01849384 - 4.64287125e-02j,
                         0.08173751 - 4.32898701e-02j,
                         -0.08944396 + 8.24523241e-03j,
                         -0.02981234 - 3.84202961e-02j,
                         0.02143829 + 8.90636559e-03j,
                         -0.07205315 - 2.38779352e-02j,
                         0.00208972 + 4.97465823e-02j,
                         -0.03431943 + 1.45140347e-01j],
                        [-0.06008399 - 4.41136724e-02j,
                         -0.03431943 - 1.45140347e-01j,
                         0.00208972 - 4.97465823e-02j,
                         -0.07205315 + 2.38779352e-02j,
                         0.02143829 - 8.90636559e-03j,
                         -0.02981234 + 3.84202961e-02j,
                         -0.08944396 - 8.24523241e-03j,
                         0.08173751 + 4.32898701e-02j,
                         0.01849384 + 4.64287125e-02j,
                         -0.00960846 - 1.72447007e-02j,
                         0.06179666 - 1.42935800e-01j,
                         -0.04840379 - 4.73925735e-02j]],
                       [[0.03545088 - 3.80874306e-05j,
                         -0.02390823 + 9.53612739e-02j,
                         0.06033527 - 8.34096192e-02j,
                         0.00775200 - 8.72879406e-02j,
                         0.01434035 - 6.27232848e-02j,
                         0.05655769 + 1.92107129e-02j,
                         0.05809326 + 5.91523095e-02j,
                         0.01424887 - 1.34623052e-02j,
                         0.03051096 - 3.63480254e-02j,
                         0.05706433 - 1.08557413e-01j,
                         0.03682702 - 5.15325338e-02j,
                         -0.03038947 + 5.35959808e-03j],
                        [0.01036957 - 1.19053781e-02j,
                         -0.14370240 - 1.73169120e-02j,
                         -0.03920884 - 1.05143244e-02j,
                         -0.01341666 + 4.63735830e-02j,
                         -0.02971482 + 3.13215582e-02j,
                         0.02019499 + 7.25981881e-02j,
                         -0.08974488 + 2.89923546e-03j,
                         0.05319896 - 3.06022830e-02j,
                         -0.03034278 - 3.14993895e-02j,
                         -0.04076314 - 1.78591240e-02j,
                         -0.03754497 - 9.93804754e-02j,
                         0.01489478 - 6.02354214e-02j],
                        [-0.01566024 - 2.34877828e-02j,
                         0.01317275 + 3.02260581e-02j,
                         0.07297952 - 1.37016018e-02j,
                         0.00776778 + 4.84519209e-02j,
                         -0.01858250 + 6.98208695e-02j,
                         0.05858073 + 3.21025288e-02j,
                         -0.04881496 + 4.07956229e-02j,
                         0.10094931 + 1.45540820e-02j,
                         0.03295335 - 1.30864786e-01j,
                         0.02853584 + 3.58611264e-02j,
                         -0.05994566 + 2.46692498e-02j,
                         0.02173232 - 3.28055745e-02j]]])

    def test_ifftn_2d(self):
        my_ifftn = ifftn(self.x2_fft)
        assert_array_almost_equal(my_ifftn(self.x2_fft),
                                  np.fft.ifftn(self.x2_fft))

    def test_ifftn_3d_larger_shape(self):
        test_shape = (4, 6, 14)
        my_ifftn = ifftn(self.x3_fft, shape=test_shape)
        assert_array_almost_equal(my_ifftn(self.x3_fft),
                                  np.fft.ifftn(self.x3_fft, s=test_shape))

    def test_ifftn_3d_smaller_shape(self):
        test_shape = (3, 3, 6)
        my_ifftn = ifftn(self.x3_fft, shape=test_shape)
        assert_array_almost_equal(my_ifftn(self.x3_fft),
                                  np.fft.ifftn(self.x3_fft, s=test_shape))

    def test_ifftn_3d_mixed_and_short_shape(self):
        test_shape = (2, 16)
        my_ifftn = ifftn(self.x3_fft, shape=test_shape)
        assert_array_almost_equal(my_ifftn(self.x3_fft),
                                  np.fft.ifftn(self.x3_fft, s=test_shape))

    def test_ifftn_3d_axes(self):
        test_axes = (0, -1)
        my_ifftn = ifftn(self.x3_fft, axes=test_axes)
        assert_array_almost_equal(my_ifftn(self.x3_fft),
                                  np.fft.ifftn(self.x3_fft, axes=test_axes))

    def test_ifftn_pair(self):
        my_ifftn, my_fftn = ifftn_pair(self.x3_fft)
        my_x = my_ifftn(self.x3_fft)
        assert_array_almost_equal(my_fftn(my_x), self.x3_fft)

    def test_ifftn_pair_cropped(self):
        my_ifftn, my_fftn = ifftn_pair(self.x3_fft,
                                       shape=(8, 4, 32),
                                       crop_fft=True)
        my_x = my_ifftn(self.x3_fft)
        assert_array_almost_equal(my_fftn(my_x), self.x3_fft)


class TestIRFFTN:
    # Test arrays
    x2_fft = np.array([[3.90005299+0.j, -2.76104255-0.68138194j,
                        0.60141991+0.51297946j, 0.60141991-0.51297946j,
                        -2.76104255+0.68138194j],
                       [1.36018608-1.00253685j, 0.20354034+5.94135865j,
                        0.22565011-1.10500067j, -0.21392501-5.59176544j,
                        -1.01272997-7.77280054j],
                       [-6.28602191+0.j, -2.46393831+0.65877986j,
                        2.24361876+7.43910467j, 2.24361876-7.43910467j,
                        -2.46393831-0.65877986j],
                       [1.36018608+1.00253685j, -1.01272997+7.77280054j,
                        -0.21392501+5.59176544j, 0.22565011+1.10500067j,
                        0.20354034-5.94135865j]])

    x3_fft = np.array([[[17.96494735 + 0.00000000e+00j,
                         -6.22501032 + 1.20170065e+01j,
                         -1.15763117 + 5.77185680e+00j,
                         1.74420677 + 4.64181812e+00j,
                         -10.16860182 + 1.47959509e+00j,
                         5.90095981 + 1.62495557e+01j,
                         14.21298435 + 0.00000000e+00j],
                        [-1.49521094 + 8.96103841e+00j,
                         5.64437986 + 6.14823906e+00j,
                         7.15869585 + 8.51834531e+00j,
                         5.35117938 - 1.84885568e+00j,
                         0.28394614 - 5.49912100e+00j,
                         -11.00975648 - 1.15898119e+00j,
                         3.57314292 + 3.93592836e+00j],
                        [-1.49521094 - 8.96103841e+00j,
                         -0.59370612 + 1.23834934e+01j,
                         -21.77449098 - 1.61931877e+00j,
                         -13.99865720 - 4.19860194e+00j,
                         -9.30505199 - 9.07285819e+00j,
                         -0.29082264 + 1.95983988e+01j,
                         3.57314292 - 3.93592836e+00j]],
                       [[5.10492686 - 5.48459000e-03j,
                         -4.37608350 + 7.71782123e-01j,
                         5.30309095 - 7.42068487e+00j,
                         8.21726302 - 1.56322675e+01j,
                         4.39357802 - 5.23411566e+00j,
                         2.05183781 - 1.93857194e+00j,
                         8.36542874 + 8.51793257e+00j],
                        [-2.25507387 - 3.38224073e+00j,
                         3.12945408 - 4.72400273e+00j,
                         -8.63217464 + 3.55237198e+00j,
                         4.10916161 + 5.16400221e+00j,
                         4.74528216 - 1.88445292e+01j,
                         14.53670034 + 2.09578781e+00j,
                         -7.02935365 + 5.87456970e+00j],
                        [1.49321854 - 1.71437444e+00j,
                         2.14484795 - 8.67390068e+00j,
                         -5.40647534 - 1.43107885e+01j,
                         -5.86989156 - 2.57171386e+00j,
                         -4.36936059 - 4.53591209e+00j,
                         7.66065026 - 4.40672875e+00j,
                         -12.92326294 + 4.17489906e-01j]],
                       [[-1.18278927 + 0.00000000e+00j,
                         2.41948435 + 3.41066917e+00j,
                         -0.68641833 + 1.60707945e+01j,
                         14.61996259 - 4.32395520e+00j,
                         -8.16507095 - 1.12071085e+01j,
                         -1.26299894 - 1.36272490e+01j,
                         12.51526589 + 0.00000000e+00j],
                        [-8.65209441 - 6.35236882e+00j,
                         -6.97014607 - 6.82453059e+00j,
                         8.89871948 - 2.05827553e+01j,
                         -1.38361766 - 2.48323690e+00j,
                         2.66311278 + 6.68573460e+00j,
                         11.77020179 + 6.23374130e+00j,
                         -12.87993001 - 1.18731347e+00j],
                        [-8.65209441 + 6.35236882e+00j,
                         -4.94199729 + 2.09002099e+01j,
                         0.30092012 + 7.16350786e+00j,
                         -10.37565320 - 3.43842266e+00j,
                         3.08711424 + 1.28251664e+00j,
                         -4.29297669 - 5.53252263e+00j,
                         -12.87993001 + 1.18731347e+00j]],
                       [[5.10492686 + 5.48459000e-03j,
                         -3.44278445 - 1.37320234e+01j,
                         8.68827888 + 1.20109852e+01j,
                         1.11628810 + 1.25694635e+01j,
                         2.06501009 + 9.03215302e+00j,
                         8.14430716 - 2.76634266e+00j,
                         8.36542874 - 8.51793257e+00j],
                        [1.49321854 + 1.71437444e+00j,
                         -20.69314621 + 2.49363533e+00j,
                         -5.64607299 + 1.51406271e+00j,
                         -1.93199942 - 6.67779595e+00j,
                         -4.27893368 - 4.51030438e+00j,
                         2.90807789 - 1.04541391e+01j,
                         -12.92326294 - 4.17489906e-01j],
                        [-2.25507387 + 3.38224073e+00j,
                         1.89687648 - 4.35255237e+00j,
                         10.50905052 + 1.97303067e+00j,
                         1.11856081 - 6.97707662e+00j,
                         -2.67587984 - 1.00542052e+01j,
                         8.43562497 - 4.62276415e+00j,
                         -7.02935365 - 5.87456970e+00j]]])

    def test_irfftn_2d(self):
        my_irfftn = irfftn(self.x2_fft)
        assert_array_almost_equal(my_irfftn(self.x2_fft),
                                  np.fft.irfftn(self.x2_fft))

    def test_irfftn_3d_larger_shape(self):
        test_shape = (4, 6, 32)
        my_irfftn = irfftn(self.x3_fft, shape=test_shape)
        assert_array_almost_equal(my_irfftn(self.x3_fft),
                                  np.fft.irfftn(self.x3_fft, s=test_shape))

    def test_irfftn_3d_smaller_shape(self):
        test_shape = (3, 3, 12)
        my_irfftn = irfftn(self.x3_fft, shape=test_shape)
        assert_array_almost_equal(my_irfftn(self.x3_fft),
                                  np.fft.irfftn(self.x3_fft, s=test_shape))

    def test_irfftn_3d_mixed_and_short_shape(self):
        test_shape = (6, 12)
        my_irfftn = irfftn(self.x3_fft, shape=test_shape)
        assert_array_almost_equal(my_irfftn(self.x3_fft),
                                  np.fft.irfftn(self.x3_fft, s=test_shape))

    def test_irfftn_3d_axes(self):
        test_axes = (0, -1)
        my_irfftn = irfftn(self.x3_fft, axes=test_axes)
        assert_array_almost_equal(my_irfftn(self.x3_fft),
                                  np.fft.irfftn(self.x3_fft, axes=test_axes))

    def test_irfftn_pair(self):
        my_irfftn, my_rfftn = irfftn_pair(self.x3_fft)
        my_x = my_irfftn(self.x3_fft)
        assert_array_almost_equal(my_rfftn(my_x), self.x3_fft)
