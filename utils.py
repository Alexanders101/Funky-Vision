__author__ = 'alex'
import numpy as np
from numpy import max as np_max
from numpy import ravel as np_ravel
from numpy import sort as np_sort
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import Image
from bottleneck import partsort
from time import time



def make_neural_label(data):
    output = []
    numLabels = len(np.unique(data))
    zr = np.zeros_like(np.ndarray((numLabels,), np.int))
    for i in xrange(len(data)):
        zr[data[i]] = 1
        output.append(zr.copy())
        zr[data[i]] = 0
    return output

def plot_density(data, smooth = False):
    min = 0
    max = np_max(data)
    data = np.ravel(data)
    count = np.bincount(data, minlength=max + 1)
    nums = np.arange(min, max+1, 1)
    if smooth:
        nums_new = np.linspace(min, max, 1000)
        count_new = spline(nums, count, nums_new)

    fig = plt.figure()
    if smooth:
        plt.plot(nums_new, count_new)
    else:
        plt.plot(nums, count)
    plt.show()


def n_largest(arr, n):
    fin = np_ravel(arr)
    fin = 255 - fin
    fin = partsort(fin, n)
    fin = 255 - fin
    return fin[:n]
def n_largest_safe(arr, n):
    return np_sort(np_ravel(arr))[-n:]

def n_smallest(arr, n):
    fin = np_ravel(arr)
    fin = partsort(fin, n)
    return fin[:n]
def n_smallest_safe(arr, n):
    return np_sort(np_ravel(arr))[:n]

def make_pretty_raw(frame):
    return (2047 - frame)

def make_mask(frame):
    img = frame / 255
    img = img.astype(bool)
    img = np.invert(img)
    return img

def convert_to_bw(frame):
    fin = frame
    fin[frame != 0] = 255
    return fin

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = ( w, h, 4 )

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.fromstring("RGBA", ( w, h ), buf.tostring())


def time_function(func, trials=3, repetitions=100, *args, **kwargs):
    times = []
    for j in range(repetitions):
        curr = []
        for i in range(trials):
            t0 = time()
            func(*args, **kwargs)
            t1 = time()
            curr.append(t1 - t0)

        times.append((np.mean(curr), np.std(curr)))
    times = np.rot90(times)
    ave = np.mean(times[1])
    std = np.std(times[0])

    print('Average time and standard deviation for function:\t%.2fms , %.2fms' % (ave * 1000, std * 1000))
