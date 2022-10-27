#!/usr/bin/python3
# Copyright (c) 2022 Michael Anderson, Joseph Collins
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from ..linear_segmentation import linear_segmentation

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import scipy.interpolate as spint
from tqdm import tqdm


use_random_walk = False

valstep = 0.0001
valmin = 0.00001
valmax = 0.15

if use_random_walk:
    def random_walk():
        dims = 1
        step_n = 10000
        step_set = [-1, 0, 1]
        origin = np.zeros((1,dims))
        step_shape = (step_n,dims)
        steps = np.random.choice(a=step_set, size=step_shape)
        path = np.concatenate([origin, steps]).cumsum(0)
        return path
    path = random_walk()
    idx = np.arange(path.shape[0])
    data_idx_ = np.linspace(idx[0], idx[-1], 100)
    tck = spint.splrep(idx, path, k=3, s=400000)
    data_ = spint.splev(data_idx_, tck, der=0)
else:
    data_ = np.sin(np.linspace(0, 10, 1000)) + 0.3*np.sin(np.linspace(0, 100, 1000))
    data_[:500] += 0.1*np.sin(np.linspace(0, 1000, 500))
    data_[250:750] += np.linspace(0.5, 0.75, 500)
    data_ *= 2.0
    data_idx_ = np.arange(data_.shape[0])

n_seg = []


# The parametrized function to be plotted
def f(tolerance):
    x, y = linear_segmentation(data_, tolerance)
    # f_interp = spint.interp1d(x, y)
    # x_new = np.linspace(x[0], x[-1], 10000)
    # y_new = f_interp(x_new)
    return x, y


# Define initial parameters
init_tol = valmax / 2.0

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots(2)
x, y = f(init_tol)
line, = ax[1].plot(data_idx_[x], y, lw=2)
ax[1].set_xlabel('Time [s]')

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the tol.
axtol = fig.add_axes([0.25, 0.1, 0.65, 0.03])
tol_slider = Slider(
    ax=axtol,
    label='Tolerance',
    valmin=valmin,
    valmax=valmax,
    valinit=init_tol,
    valstep=valstep
)

for tol in tqdm(np.linspace(valmin,
                            valmax,
                            int(np.ceil((valmax - valmin) / valstep)))):
    _, y= linear_segmentation(data_, tol)
    n_seg.append(len(y) - 1)

n_seg_x = np.linspace(valmin, valmax, len(n_seg))
n_seg_interp = spint.interp1d(n_seg_x, n_seg)
seg_plot = ax[0].plot(n_seg_x, n_seg)
seg_point, = ax[0].plot(init_tol,
                        n_seg_interp(init_tol),
                        marker='o',
                        markersize=10,
                        markerfacecolor="red")


# The function to be called anytime a slider's value changes
def update(val):
    x, y = f(tol_slider.val)
    line.set_xdata(data_idx_[x])
    line.set_ydata(y)
    seg_point.set_xdata(tol_slider.val)
    seg_point.set_ydata(n_seg_interp(tol_slider.val))
    fig.canvas.draw_idle()


# register the update function with each slider
tol_slider.on_changed(update)

if use_random_walk:
    ax[1].plot(path)
ax[1].plot(data_idx_, data_)
plt.show()
