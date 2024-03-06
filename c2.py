#!/usr/bin/env python

# ejemplo de selección de ROI

import numpy as np
import cv2 as cv
from random import randint

from umucv.util import putText
from umucv.stream import autoStream
from scipy import signal

from typing import List, Tuple, Dict
from polynomials import *


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

        """Dict to store the center and radius of the circular ROI"""
        self.rois: Dict[int,Tuple[Tuple[int],int]] = {}
        """List to store the masks of the circular ROI"""
        self.masks: Dict[Tuple[Tuple[int],int],np.ndarray] = {}
        """List to store the boxes where to apply filters"""
        self.boxes: Dict[Tuple[Tuple[int],int],np.ndarray] = {}
        self.window = window

        
    
        def fun(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                if not hash((x,y)) in self.rois:

                    radius = randint(10, 100)
                    self.rois[hash((x,y))] = ((x,y),radius)
        
        self.fun = fun
            
        self.setAsCallback()
    
    def setAsCallback(self):
        """Method to set the class as the mouse callback"""
        cv.setMouseCallback(self.window, self.fun)
    
    
    def create_masks(self, h,w):
        """Method to fill the masks per each circular ROI"""
        for roi in self.rois:
            roi = self.rois[roi]
            if hash(roi[0]) in self.masks:
                continue

            mask = np.zeros((h,w), dtype=np.uint8)
            center, radius = roi
            cv.circle(mask, center, radius, 255, -1)

            self.masks[hash(roi[0])] = mask
    
    def create_boxes(self, frame):
        """Method to fill the boxes where to apply filters"""
        for roi in self.rois:
            roi = self.rois[roi]
            center, radius = roi
            x, y = center

            self.boxes[hash(roi[0])] = frame[y-radius:y+radius, x-radius:x+radius]
    
    def apply_filter(self, frame, active):
        """Method to apply the filter to the boxes"""
        for roi in self.rois:
            roi = self.rois[roi]
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
            
            elif active == 'g':
                mask = self.masks[has_roi]
                self.boxes[has_roi] = cv.GaussianBlur(self.boxes[has_roi], (15,15), 10)
                x, y = roi[0]
                radius = roi[1]
                mask_section = mask[y-radius:y+radius, x-radius:x+radius]
                frame_section = frame[y-radius:y+radius, x-radius:x+radius]
                frame[y-radius:y+radius, x-radius:x+radius] = np.where(mask_section == 255,self.boxes[has_roi], frame_section )


    
    def clean(self):
        """Method to clean the ROI"""
        self.rois.clear()
        self.masks.clear()
        self.boxes.clear()

        

class PolyROI:
    def __init__(self, window):
        """Class to handle the circular ROI selection in a window"""

        """Dict to store the center and radius of the circular ROI"""
        self.rois: Dict[int,Tuple[Tuple[int],int]] = {}
        """List to store the masks of the circular ROI"""
        self.masks: Dict[Tuple[pt,int],np.ndarray] = {}
        """List to store the boxes where to apply filters"""
        self.boxes: Dict[Tuple[pt,int],np.ndarray] = {}
        self.window = window

        def random_radius(upper_limit: int, lower_limit: int) -> int:
            """Return a random radius"""
            to_return = randint(upper_limit, lower_limit if lower_limit % 2 == 0 else lower_limit - 1) 
            return to_return if to_return % 2 == 0 else to_return + 1

    
        def fun(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                hash_point = hash((x,y))
                if not hash_point in self.rois:
                    radii = [random_radius(10, 100) for _ in range(randint(2,20))]
                    self.rois[hash_point] =  (pt(x, y), radii)
        
        self.fun = fun
            
        cv.setMouseCallback(window, fun)
    
    def setAsCallback(self):
        """Method to set the class as the mouse callback"""
        cv.setMouseCallback(self.window, self.fun)
    
    
    def create_masks(self, h,w):
        """Method to fill the masks per each circular ROI"""
        for roi in self.rois:
            roi = self.rois[roi]
            if hash(roi[0]) in self.masks:
                continue

            center, radii = roi
            points = polynomialModel(center, radii, 15)
            polygon = lPolynomial2nlPolynomial(points)
            mask = maskFromPolygons([np.array(polygon, np.int32)], (h,w))

            self.masks[hash(roi[0])] = mask
    
    def create_boxes(self, frame):
        """Method to fill the boxes where to apply filters"""
        for roi in self.rois:
            roi = self.rois[roi]

            center, radii = roi
            x, y = center

            radius = max(radii)

            self.boxes[hash(roi[0])] = frame[y-radius:y+radius, x-radius:x+radius]
    
    def apply_filter(self, frame, active):
        """Method to apply the filter to the boxes"""
        for roi in self.rois:
            roi = self.rois[roi]
            has_roi = hash(roi[0])
            if active == 'b':
                mask = self.masks[has_roi]
                self.boxes[has_roi] = bordes(self.boxes[has_roi])
                x, y = roi[0]
                radii = roi[1]
                radius = max(radii)
                mask_section = mask[y-radius:y+radius, x-radius:x+radius]
                frame_section = frame[y-radius:y+radius, x-radius:x+radius]
                frame[y-radius:y+radius, x-radius:x+radius] = np.where(mask_section == 255,self.boxes[has_roi], frame_section )

            elif active == 'l':
                mask = self.masks[has_roi]
                self.boxes[has_roi] = cconv(
                    ([[0, -1,  0], [-1, 4, -1], [0, -1,  0]]), self.boxes[has_roi])
                x, y = roi[0]
                radii = roi[1]
                radius = max(radii)
                mask_section = mask[y-radius:y+radius, x-radius:x+radius]
                frame_section = frame[y-radius:y+radius, x-radius:x+radius]
                frame[y-radius:y+radius, x-radius:x+radius] = np.where(mask_section == 255,self.boxes[has_roi], frame_section )

            elif active == 'g':
                mask = self.masks[has_roi]
                self.boxes[has_roi] = cv.GaussianBlur(self.boxes[has_roi], (15,15), 10)
                x, y = roi[0]
                radii = roi[1]
                radius = max(radii)
                mask_section = mask[y-radius:y+radius, x-radius:x+radius]
                frame_section = frame[y-radius:y+radius, x-radius:x+radius]
                frame[y-radius:y+radius, x-radius:x+radius] = np.where(mask_section == 255,self.boxes[has_roi], frame_section )


    
    def clean(self):
        """Method to clean the ROI"""
        self.rois.clear()
        self.masks.clear()
        self.boxes.clear()

cv.namedWindow("input")
cv.moveWindow('input', 0, 0)

region = ROI("input")
poly_region = PolyROI("input")
circular_region = CircularROI("input")

active = ''
mode = 'circular'   # 'quad' or 'circular'
points = []     # list of points to apply masks in circular regions

AVAILABLE_KEYS = [ord('b'), ord('l'), ord('g')]

for key, frame in autoStream():
    h, w = frame.shape[:2]
    frame = rgb2gray(frame)
    frame = gray2float(frame)


    if key == ord('e'):
        if mode == 'quad':
            mode = 'circular'
        elif mode == 'circular':
            mode = 'poly'
        else:
            mode = 'quad'
        
        if mode == 'quad':
            region.setAsCallback()
        elif mode == 'circular':
            circular_region.setAsCallback()
        elif mode == 'poly':
            poly_region.setAsCallback()
    elif key == ord('c'):
        active = ''
    elif key == ord('q'):
        break
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
    
    elif poly_region.rois and mode == 'poly':
        if key == ord('x'):
            poly_region.clean()
        
        poly_region.create_masks(h,w)
        poly_region.create_boxes(frame)
        poly_region.apply_filter(frame, active)
        

    frame = float2gray(frame)

    
    putText(frame, f'{w}x{h}')
    putText(frame, f'Mode: {mode}', orig=(w-120, 40))
    putText(frame, f'Letter: {active}', orig=(w-120, 16))
    cv.imshow('input', frame)
