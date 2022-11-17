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
import itertools as it
import numpy as np


def refinement(data, idx):
    """
    Refine linear segments using least squares.
    TODO: when segments are too small, some overlap occurs and lines get wonky
    
    :param data: original data
    :param idx: indices of linear segments in data
    :returns: new tuple of (indices, values) for refined linear segments
    """
    coeffs = []
    for seg_idx in zip(idx[:-1], idx[1:]):
        a, b = seg_idx
        x = np.arange(a, b + 1)
        X = np.vstack((np.ones(x.shape), x)).T
        y = data[a:b + 1, np.newaxis]
        (c0, c1), _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        coeffs.append((c0, c1))

    x_ref = [0.]
    y_ref = [coeffs[0][0]]
    for jdx, (c_a, c_b) in enumerate(zip(coeffs[:-1], coeffs[1:])):
        m_a, b_a = c_a[1], c_a[0]
        m_b, b_b = c_b[1], c_b[0]
        x_ref.append(((b_a - b_b) / (m_b - m_a))[0])
        y_ref.append((m_a*x_ref[-1] + b_a)[0])
    x_ref.append(idx[-1])
    y_ref.append((coeffs[-1][1]*idx[-1] + coeffs[-1][0])[0])
    return np.round(x_ref).astype(int), y_ref


def __refinement_cls(data, idx):
    """
    (WIP) Refine linear segments using constrained least squares.
    
    :param data: original data
    :param idx: indices of linear segments in data
    :returns: new tuple of (indices, values) for refined linear segments
    """
    M = []  # slope matrix
    B = []  # intercept matrix
    L = []  # lambda matrix
    C = []  # value matrix
    for jdx, seg_idx in enumerate(zip(idx[:-1], idx[1:])):
        a, b = seg_idx
        sum_x_pow0_a2b = b - a + 1.
        sum_x_pow1_a2b = 0.5 * (b - a + 1.) * (a + b)
        sum_x_pow2_a2b = (1. / 6.) * (b - a + 1.) * (2.*a**2. + 2.*a*b - a + 2.*b**2. + b)
        sum_y_a2b = np.sum(data[a:b+1])
        sum_xy_a2b = np.sum(np.arange(a, b+1) * np.array(data[a:b+1]))
        if jdx == 0:
            M.append([sum_x_pow2_a2b, sum_x_pow1_a2b, b])
            B.append([sum_x_pow1_a2b, sum_x_pow0_a2b, 1.])
        elif jdx + 1 < len(idx[:-1]):
            M.append([0., 0.] + 3*(jdx - 1)*[0.] + [-a, sum_x_pow2_a2b, sum_x_pow1_a2b, b])
            B.append([0., 0.] + 3*(jdx - 1)*[0.] + [-1., sum_x_pow1_a2b, sum_x_pow0_a2b, 1.])
        else:
            M.append([0., 0.] + 3*(jdx - 1)*[0.] + [-a, sum_x_pow2_a2b, sum_x_pow1_a2b])
            B.append([0., 0.] + 3*(jdx - 1)*[0.] + [-1., sum_x_pow1_a2b, sum_x_pow0_a2b])

        if jdx == 0:
            L.append([b, -1., 0., -b])
        elif jdx + 1 < len(idx[:-2]):
            L.append(3*jdx*[0.] + [b, -1., 0., -b])
        elif jdx + 1 == len(idx[:-2]):
            L.append(3*jdx*[0.] + [b, -1., 0., 0.])
        else:
            L.append(3*jdx*[0.] + [0., 0.])

        if jdx + 1 < len(idx[:-1]):
            C.extend([sum_xy_a2b, sum_y_a2b, 0.])
        else:
            C.extend([sum_xy_a2b, sum_y_a2b])

    M = np.array(list(it.zip_longest(*M, fillvalue=0.)))
    B = np.array(list(it.zip_longest(*B, fillvalue=0.)))
    L = np.array(list(it.zip_longest(*L, fillvalue=0.)))
    MBL = np.hstack((M, B, L))
    C = np.array(C)[:, np.newaxis]

    mbl, _, _, _ = np.linalg.lstsq(MBL, C, rcond=None)
    m = mbl[:(len(idx) - 1)]
    b = mbl[(len(idx) - 1):(2*len(idx) - 1)]
    y_ref = []
    for ldx, (jdx, kdx, m_, b_) in enumerate(zip(idx[:-1], idx[1:], m, b)):
        if ldx == 0:
            y_ref.append((m_*jdx+b_)[0])
            y_ref.append((m_*kdx+b_)[0])
        else:
            y_ref.append((m_*kdx+b_)[0])

    return idx, y_ref


def linear_segment(data, idx):
    x_ = np.array(idx)
    X_ = np.vstack((np.ones(x_.shape), x_)).T
    y_ = data[[idx[0], idx[1] - 1], np.newaxis]
    c, m = np.linalg.solve(X_, y_)
    return lambda x: m[0]*x + c[0]


def linear_segmentation(data, tol=1):
    """
    Find linear segments in data series. Works better with smoothed data. For direct use with
    interpolation. Run `refinement` afterwords for least squares fit of linear segments to data.
    
    :param data: original data
    :param tol: normalized absolute error for splitting line segments
    :returns: new tuple of (indices, values) for linear segments
    """
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


if __name__ == '__main__':
    import scipy.interpolate as spint
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

    x, y = linear_segmentation(data_, 0.03)
    print(x, y)
    if len(x) > 2:
        x_ref, y_ref = refinement(data_, x)
        print(x_ref, y_ref)
