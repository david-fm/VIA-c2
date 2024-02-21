#!/usr/bin/env python

# ejemplo de selección de ROI

import numpy as np
import cv2 as cv
from random import randint

from umucv.util import putText
from umucv.stream import autoStream
from scipy import signal

from typing import List, Tuple, Dict



def cconv(k, x):
    return signal.convolve2d(x, k, boundary='symm', mode='same')


def bordes(x):
    kx = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    ky = kx.T
    gx = cconv(kx, x)
    gy = cconv(ky, x)
    return np.absolute(gx)+np.absolute(gy)


def rgb2gray(x):
    return cv.cvtColor(x, cv.COLOR_RGB2GRAY)


def gray2float(x):
    return x.astype(float) / 255


def float2gray(x):
    return (255*x).astype(np.uint8)


class ROI:
    def __init__(self, window):
        self.roi = []
        self.DOWN = False
        self.window = window

        def poly():
            x1, y1, x2, y2 = self.roi
            self.box = np.array(
                [[x1-1, y1-1], [x1-1, y2], [x2, y2], [x2, y1-1]])

        def fun(event, x, y, flags, param):
            if not self.DOWN and event == cv.EVENT_LBUTTONDOWN:
                self.roi = [x, y, x+1, y+1]
                poly()
                self.DOWN = True
            elif self.DOWN and event == cv.EVENT_LBUTTONDOWN:
                self.roi[2:] = x+1, y+1
                self.DOWN = False
            elif self.DOWN:
                x1, y1, _, _ = self.roi
                x2 = x+1
                y2 = y+1
                self.roi = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                poly()
        self.fun = fun
        cv.setMouseCallback(window, fun)
    
    def setAsCallback(self):
        """Method to set the class as the mouse callback"""
        cv.setMouseCallback(self.window, self.fun)

class CircularROI:
    def __init__(self, window):
        """Class to handle the circular ROI selection in a window"""

        """List to store the center and radius of the circular ROI"""
        self.rois: List[Tuple[Tuple[int],int]] = []
        """List to store the masks of the circular ROI"""
        self.masks: Dict[Tuple[Tuple[int],int],np.ndarray] = {}
        """List to store the boxes where to apply filters"""
        self.boxes: Dict[Tuple[Tuple[int],int],np.ndarray] = {}
        self.window = window

        
    
        def fun(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                radius = randint(10, 100)
                self.rois.append(((x, y), radius))
        
        self.fun = fun
            
        cv.setMouseCallback(window, fun)
    
    def setAsCallback(self):
        """Method to set the class as the mouse callback"""
        cv.setMouseCallback(self.window, self.fun)
    
    
    def create_masks(self, h,w):
        """Method to fill the masks per each circular ROI"""
        for roi in self.rois:
            if hash(roi[0]) in self.masks:
                continue

            mask = np.zeros((h,w), dtype=np.uint8)
            center, radius = roi
            cv.circle(mask, center, radius, 255, -1)

            self.masks[hash(roi[0])] = mask
    
    def create_boxes(self, frame):
        """Method to fill the boxes where to apply filters"""
        for roi in self.rois:
            
            center, radius = roi
            x, y = center

            self.boxes[hash(roi[0])] = frame[y-radius:y+radius, x-radius:x+radius]
    
    def apply_filter(self, frame, active):
        """Method to apply the filter to the boxes"""
        for roi in self.rois:
            has_roi = hash(roi[0])
            if active == 'b':
                mask = self.masks[has_roi]
                self.boxes[has_roi] = bordes(self.boxes[has_roi])
                x, y = roi[0]
                radius = roi[1]
                mask_section = mask[y-radius:y+radius, x-radius:x+radius]
                frame_section = frame[y-radius:y+radius, x-radius:x+radius]
                frame[y-radius:y+radius, x-radius:x+radius] = np.where(mask_section == 255,self.boxes[has_roi], frame_section )
            elif active == 'l':
                mask = self.masks[has_roi]
                self.boxes[has_roi] = cconv(
                    ([[0, -1,  0], [-1, 4, -1], [0, -1,  0]]), self.boxes[has_roi])
                x, y = roi[0]
                radius = roi[1]
                mask_section = mask[y-radius:y+radius, x-radius:x+radius]
                frame_section = frame[y-radius:y+radius, x-radius:x+radius]
                frame[y-radius:y+radius, x-radius:x+radius] = np.where(mask_section == 255,self.boxes[has_roi], frame_section )


    
    def clean(self):
        """Method to clean the ROI"""
        self.rois = []
        self.masks.clear()
        self.boxes.clear()
    

cv.namedWindow("input")
cv.moveWindow('input', 0, 0)

region = ROI("input")
circular_region = CircularROI("input")
active = ''
mode = 'circular'   # 'quad' or 'circular'
points = []     # list of points to apply masks in circular regions

AVAILABLE_KEYS = [ord('b'), ord('l')]

for key, frame in autoStream():
    h, w = frame.shape[:2]
    frame = rgb2gray(frame)
    frame = gray2float(frame)


    if key == ord('e'):
        mode = 'circular' if mode == 'quad' else 'quad'
        if mode == 'quad':
            region.setAsCallback()
        else:
            circular_region.setAsCallback()
    elif key == ord('c'):
        active = ''
    elif key in AVAILABLE_KEYS:
            active = chr(key)

    if region.roi and mode == 'quad':

        [x1, y1, x2, y2] = region.roi
        if key == ord('x'):
            region.roi = []
        

        cv.rectangle(frame, (x1, y1), (x2, y2),
                     color=(0, 255, 255), thickness=2)

        # putText(frame, f'{x2-x1+1}x{y2-y1+1}', orig=(x1,y1-8))

        match active:
            case 'b':
                frame[y1:y2, x1:x2] = bordes(frame[y1:y2, x1:x2])
            case 'l':
                frame[y1:y2, x1:x2] = cconv(
                    ([[0, -1,  0], [-1, 4, -1], [0, -1,  0]]), frame[y1:y2, x1:x2])

    elif circular_region.rois and mode == 'circular':
        if key == ord('x'):
            circular_region.clean()
        
        circular_region.create_masks(h,w)
        circular_region.create_boxes(frame)
        circular_region.apply_filter(frame, active)
        

    frame = float2gray(frame)

    
    putText(frame, f'{w}x{h}')
    putText(frame, f'Mode: {mode}', orig=(w-120, 40))
    putText(frame, f'Letter: {active}', orig=(w-120, 16))
    cv.imshow('input', frame)
