__author__ = 'alex'
import cv2
import numpy as np

class KeyPoints:
    def __init__(self, features=150, gauss_blur=3, med_blur=5, history=1, clip_limit=100.0, tile_grid=(8, 8)):
        self.num_features = 150
        self.orb = cv2.ORB_create(features)
        self.bsb = cv2.createBackgroundSubtractorMOG2(history)
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        self.gaussian_blur_factor = gauss_blur
        self.median_blur_factor = med_blur

    def get_num_features(self):
        return self.num_features

    def _get_process_speedup(self):
        return [cv2.cvtColor,
                cv2.COLOR_BGR2GRAY,
                cv2.equalizeHist,
                self.clahe.apply,
                cv2.GaussianBlur,
                self.gaussian_blur_factor,
                cv2.medianBlur,
                self.median_blur_factor]

    def _get_kp_speedup(self):
        return [self.orb.detect,
                self.orb.compute]

    def _get_draw_speedup(self):
        return [cv2.drawKeypoints,
                np.zeros_like]

    def __process_image(self, frm, speedup=None):
        if speedup is None:
            frm = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            # frm = bsb.apply(frm)
            frm = cv2.equalizeHist(frm, frm)
            frm = self.clahe.apply(frm)
            frm = cv2.GaussianBlur(frm, (self.gaussian_blur_factor, self.gaussian_blur_factor), 1)
            frm = cv2.medianBlur(frm, self.median_blur_factor)
        else:
            frm = speedup[0](frm, speedup[1])
            # frm = bsb.apply(frm)
            frm = speedup[2](frm, frm)
            frm = speedup[3](frm)
            frm = speedup[4](frm, (speedup[5], speedup[5]), 1)
            frm = speedup[6](frm, speedup[7])
        return frm

    def get_key_points(self, frame, speedup=None):
        if speedup is None:
            kp = self.orb.detect(frame, None)
            kp, des = self.orb.compute(frame, kp)
        else:
            kp = speedup[0](frame, None)
            kp, des = speedup[1](frame, kp)
        return (kp, des)

    def draw_key_points(self, frame, kp, black_screen=False, speedup=None):
        if speedup is None:
            frame = cv2.drawKeypoints(np.zeros_like(frame) if black_screen else frame, kp, frame)
        else:
            frame = speedup[0](speedup[1](frame) if black_screen else frame, kp, frame)
        return frame



