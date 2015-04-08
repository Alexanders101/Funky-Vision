__author__ = 'alex'

import numpy as np
from numpy import max as npmax
from numpy import ravel as np_ravel
from numpy import bincount as np_bincount
from numpy import sum as np_sum
from numpy import zeros_like
from numpy import uint8
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.stats import threshold
from copy import copy
from itertools import izip
from cv2 import medianBlur as cv2_medianBlur


def __find_peak_intervals_two(data):
    indices = np.array(range(len(data)))
    tags = data > 0
    fst = indices[tags & ~ np.roll(tags, 1)]
    lst = indices[tags & ~ np.roll(tags, -1)]
    ranges = [(i, j) for i, j in izip(fst, lst)]

    return ranges


def __find_peak_intervals(data):
    isntzero = np.concatenate(([0], np.greater(data, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(isntzero))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)

    return ranges


def __get_local_maximums(data, intervals):
    return np.array([npmax(data[i:j]) for i, j in intervals])


def __thinout_peaks(data, intervals, lmax, factor=10):
    lmax /= factor
    for (i, j), k in izip(intervals, lmax):
        data[i:j] = [(m if m > k else 0) for m in data[i:j]]
    return data


def __remove_small_peaks(data, intervals, lmax, factor=10):
    small = (lmax < (npmax(data) / factor))
    for is_small, (min, max) in izip(small, intervals):
        if is_small:
            data[min:max] = 0
    return data


def __remove_thin_peaks(data, intervals, min=15):
    thin = (intervals[:, 1] - intervals[:, 0] - 1) < min
    for is_thin, (min, max) in izip(thin, intervals):
        if is_thin:
            data[min:max] = 0
    return data


def __flat_binarize(data):
    out = data > 0
    return out.astype(np.int)


def __get_bins(frame, length=256):
    return np_bincount(np_ravel(frame), minlength=length)


def __pipeline(data):
    data[:11] = 0
    data[data < (npmax(data) / 10)] = 0
    data = medfilt(data, 3)
    intervals = __find_peak_intervals(data)
    local_max = __get_local_maximums(data, intervals)
    data = __remove_small_peaks(data, intervals, local_max, 10)
    data = __thinout_peaks(data, intervals, local_max, 4)
    intervals = __find_peak_intervals(data)
    data = __remove_thin_peaks(data, intervals, 20)
    data = __flat_binarize(data)

    return data


def __get_depths(data, indeces):
    end = __pipeline(data).astype(bool)
    return indeces[end]


def __get_depths_range(data):
    end = __pipeline(data)
    intervals = __find_peak_intervals(end)
    return intervals


def isolate_depths(frame, expand=1.0, length=256, blur=None):
    """Isolate depths of a greyscale depth image

    frame: frame to analyse
    expand: amount to expand the range of the isolation
    length: possible values of pixels
    blur: amount to blur, None for none
    """
    if blur is not None:
        frame = cv2_medianBlur(frame, blur, frame)
    expand -= (expand - 1) / 2
    count = __get_bins(length)
    ranges = __get_depths_range(count).astype(float)
    ranges[:, 1] *= expand
    ranges[:, 0] /= expand
    ranges = ranges.astype(uint8)
    iso = [threshold(frame, start, stop - 1) for start, stop in ranges]
    if len(iso) == 0:
        return zeros_like(frame)
    return np_sum(iso, 0).astype(uint8)


def compare(frame):
    """Compare effects of depth isolation in histogram form"""
    small = copy(frame)
    done = __pipeline(frame)
    done = done.astype(float) / npmax(done)
    small[:5] = 0
    small = (small.astype(float) / npmax(small))
    plt.plot(done)
    plt.plot(small)
    plt.ylim(0, 1.0)
    plt.show()