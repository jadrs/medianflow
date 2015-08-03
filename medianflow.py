# -*- coding: utf-8 -*-
""" MedianFlow sandbox

Usage:
  medianflow.py SOURCE

Options:
  SOURCE    INT: camera, STR: video file
"""
from __future__ import print_function
from __future__ import division

from docopt import docopt

from os.path import abspath, exists

import numpy as np

import cv2

from rect_selector import RectSelector


class MedianFlowTracker(object):
    def __init__(self):
        self.lk_params = dict(winSize  = (11, 11),
                              maxLevel = 3,
                              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.1))

        self._atan2 = np.vectorize(np.math.atan2)

    def track(self, bb, prev, curr):
        self._n_samples = 100
        self._fb_max_dist = 1
        self._ds_factor = 0.95
        self._min_n_points = 10

        # sample points inside the bounding box
        p0 = np.empty((self._n_samples, 2))
        p0[:, 0] = np.random.randint(bb[0], bb[2] + 1, self._n_samples)
        p0[:, 1] = np.random.randint(bb[1], bb[3] + 1, self._n_samples)

        p0 = p0.astype(np.float32)

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

        # keep half of the points
        err = err[good].flatten()
        if len(err) < self._min_n_points:
            return None

        indx = np.argsort(err)
        half_indx = indx[:len(indx) // 2]
        p0 = (p0[good])[half_indx]
        p1 = (p1[good])[half_indx]

        # estimate displacement
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

        # update bounding box
        dx_scale = (ds - 1.0) * 0.5 * (bb[3] - bb[1] + 1)
        dy_scale = (ds - 1.0) * 0.5 * (bb[2] - bb[0] + 1)
        bb_curr = (int(bb[0] + dx - dx_scale + 0.5),
                   int(bb[1] + dy - dy_scale + 0.5),
                   int(bb[2] + dx + dx_scale + 0.5),
                   int(bb[3] + dy + dy_scale + 0.5))

        if bb_curr[0] >= bb_curr[2] or bb_curr[1] >= bb_curr[3]:
            return None

        bb_curr = (min(max(0, bb_curr[0]), curr.shape[1]),
                   min(max(0, bb_curr[1]), curr.shape[0]),
                   min(max(0, bb_curr[2]), curr.shape[1]),
                   min(max(0, bb_curr[3]), curr.shape[0]))

        return bb_curr


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
        self._bounding_box = None

        self._tracker = MedianFlowTracker()

    def on_rect(self, rect):
        self._bounding_box = rect

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

            if prev is not None and self._bounding_box is not None:
                bb = self._tracker.track(self._bounding_box, prev, curr)

                if bb is not None:
                    self._bounding_box = bb
                    cv2.rectangle(frame, self._bounding_box[:2], self._bounding_box[2:], (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, self._bounding_box[:2], self._bounding_box[2:], (0, 0, 255), 2)

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
