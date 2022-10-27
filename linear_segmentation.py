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

from bisect import bisect_right
import numpy as np


def linear_segment(data, idx):
    x_ = np.array(idx)
    X_ = np.vstack((np.ones(x_.shape), x_)).T
    y_ = data[[idx[0], idx[1] - 1], np.newaxis]
    c, m = np.linalg.solve(X_, y_)
    return lambda x: m[0]*x + c[0]


def linear_segmentation(data, tol=1):
    idx = [0, data.shape[0]]
    norm_data = data / np.linalg.norm(data)
    def find_points_recurse(idx_seg):
        idx_seg_local = idx_seg[:]
        data_seg = norm_data[slice(*idx_seg_local)]
        y_seg_fn = linear_segment(norm_data, idx_seg_local)
        x_seg = np.arange(*idx_seg_local)
        y_seg = np.array([y_seg_fn(x) for x in x_seg])
        err_seg = data_seg - y_seg
        if np.abs(err_seg).max() > tol:
            edx = np.argmax(np.abs(err_seg)) + idx_seg_local[0]
            ins = bisect_right(idx, edx)
            idx.insert(ins, edx)
            fore = [idx_seg_local[0], edx]
            aft = [edx, idx_seg_local[1]]
            find_points_recurse(fore)
            find_points_recurse(aft)
    find_points_recurse(idx)
    idx[-1] -= 1
    return idx, data[idx]