#!/usr/bin/env python3
from pathlib import Path
import cv2 as cv
import numpy as np
from dataclasses import dataclass

tuner_window = "Static image HSL Tuner"
image_path = Path.cwd()/"test-images"/"close.jpg"
print(image_path)

class Range:
    low: int
    high: int
    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high
        
    @staticmethod
    def uint8() -> "Range":
        return Range(0, 255)

class HLSRange:
    hue: Range
    saturation: Range
    lightness: Range

    def __init__(self):
        self.hue = Range.uint8()
        self.saturation = Range.uint8()
        self.lightness = Range.uint8()

    def high(self) -> list[int]:
        return [self.hue.high, self.lightness.high, self.saturation.high]

    def low(self) -> list[int]:
        return [self.hue.low, self.lightness.low, self.saturation.low]

    def bounds(self) -> (list[int], list[int]):
        return (self.low(), self.high())

def create_uint8_trackbar(name, callback):
    cv.createTrackbar(name, tuner_window, 0, 255, callback)

filter = HLSRange()

cv.namedWindow(tuner_window)
image = cv.imread(image_path.as_posix())

cv.imshow(tuner_window, image)
while cv.waitKey(20) != 27:
    pass