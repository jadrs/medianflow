# -*- coding: utf-8 -*-
""" MedianFlow sandbox

Usage:
  medianflow++.py SOURCE

Options:
  SOURCE    INT: camera, STR: video file
"""
from __future__ import print_function
from __future__ import division

from docopt import docopt

from os.path import abspath, exists

import numpy as np

import cv, cv2

from rect_selector import RectSelector


class MedianFlowTracker(object):
    def __init__(self):
        self.lk_params = dict(winSize  = (11, 11),
                              maxLevel = 3,
                              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.1))

        self._atan2 = np.vectorize(np.math.atan2)

        self._n_samples = 100
        self._min_n_points = 10
        self._fb_max_dist = 1
        self._ds_factor = 0.95
        self._theta_factor = 1.0
        self._concentration = 0.25

    def track(self, target, prev, curr):
        x0, y0, sx, sy, theta = target

        # sample points from an anisotropic 2D Gaussian
        p0 = np.array([np.random.normal(0.0, self._concentration * sx, self._n_samples),
                       np.random.normal(0.0, self._concentration * sy, self._n_samples)])
        ct, st = np.cos(theta), np.sin(theta)
        R = np.array([[ct, st], [-st, ct]])
        p0 = np.transpose(np.dot(R, p0) + np.array([[x0], [y0]]))

        indx = (p0[:, 0] >= 0.0) & (p0[:, 0] < prev.shape[1]) & \
               (p0[:, 1] >= 0.0) & (p0[:, 1] < prev.shape[0])
        if len(indx) < self._min_n_points:
            return None

        p0 = p0[indx, :].astype(np.float32)

        # forward-backward tracking
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev, curr, p0, None, **self.lk_params)
        indx = np.where(st == 1)[0]
        p0 = p0[indx, :]
        p1 = p1[indx, :]
        p0r, st, err = cv2.calcOpticalFlowPyrLK(curr, prev, p1, None, **self.lk_params)
        if err is None:
            return None

        # check forward-backward error and min number of points
        fb_dist = np.abs(p0 - p0r).max(axis=1)
        good = fb_dist < self._fb_max_dist
        err = err[good].flatten()
        if len(err) < self._min_n_points:
            return None

        # keep half of the points
        indx = np.argsort(err)
        half_indx = indx[:len(indx) // 2]
        p0 = (p0[good])[half_indx]
        p1 = (p1[good])[half_indx]

        # estimate median displacement
        dx = np.median(p1[:, 0] - p0[:, 0])
        dy = np.median(p1[:, 1] - p0[:, 1])

        # all pairs in prev and curr
        i, j = np.triu_indices(len(p0), k=1)
        pdiff0 = p0[i] - p0[j]
        pdiff1 = p1[i] - p1[j]

        # estimate change in scale
        p0_dist = np.sum(pdiff0 ** 2, axis=1)
        p1_dist = np.sum(pdiff1 ** 2, axis=1)
        ds = np.sqrt(np.median(p1_dist / (p0_dist + 2**-23)))
        ds = (1.0 - self._ds_factor) + self._ds_factor * ds;

        # estimate rotation
        theta0 = self._atan2(pdiff0[:, 1], pdiff0[:, 0])
        theta1 = self._atan2(pdiff1[:, 1], pdiff1[:, 0])
        dtheta = np.median(theta1 - theta0)

        return (x0 + ds * dx,
                y0 + ds * dy,
                sx * ds,
                sy * ds,
                theta + self._theta_factor * dtheta)


class API(object):
    def __init__(self, win, source):
        self._device = cv2.VideoCapture(source)
        if isinstance(source, str):
            self.paused = True
        else:
            self.paused = False

        self.win = win
        cv2.namedWindow(self.win, 1)
        self.rect_selector = RectSelector(self.win, self.on_rect)
        self._target = None

        self._tracker = MedianFlowTracker()

    def on_rect(self, rect):
        self._target = [0.5 * (rect[0] + rect[2]),
                        0.5 * (rect[1] + rect[3]),
                        0.5 * (rect[2] - rect[0] + 1.0),
                        0.5 * (rect[3] - rect[1] + 1.0),
                        0.0]

    def run(self):
        prev, curr = None, None

        ret, frame = self._device.read()
        if not ret:
            raise IOError('can\'t reade frame')

        while True:
            if not self.rect_selector.dragging and not self.paused:
                ret, grabbed_frame = self._device.read()
                if not ret:
                    break

            frame = grabbed_frame.copy()

            prev, curr = curr, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev is not None and self._target is not None:
                tgt = self._tracker.track(self._target, prev, curr)

                if tgt is not None:
                    self._target = tgt[:]
                    color = (0, 255, 0)
                else:
                    tgt = self._target[:]
                    color = (0, 0, 255)

                center = (int(tgt[0]), int(tgt[1]))
                scale = (int(tgt[2]), int(tgt[3]))
                angle = tgt[4] * 180.0 / np.pi
                cv.Ellipse(cv.fromarray(frame), center, scale, angle, 0., 360., color, 2)

            self.rect_selector.draw(frame)

            cv2.imshow(self.win, frame)

            ch = cv2.waitKey(1)
            if ch == 27 or ch in (ord('q'), ord('Q')):
                break
            elif ch in (ord('p'), ord('P')):
                self.paused = not self.paused


if __name__ == "__main__":
    args = docopt(__doc__)

    try:
        source = int(args['SOURCE'])
    except:
        source = abspath(str(args['SOURCE']))
        if not exists(source):
            raise IOError('file does not exists')

    API("Median Flow Tracker", source).run()
