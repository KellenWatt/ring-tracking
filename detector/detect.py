import cv2 as cv
import numpy as np
import math

from detector.match import Match


def circularity(area, perimeter):
    return 4 * math.pi * (area / (perimeter*perimeter))


def filter_image(img, lower, upper):
    mask = cv.inRange(img, lower, upper)
    return cv.bitwise_and(img, img, mask=mask)


def hls_to_gray(img):
    return cv.cvtColor(cv.cvtColor(img, cv.COLOR_HLS2BGR_FULL), cv.COLOR_BGR2GRAY)

def ring(img, lower, upper, noise_filter=0.02, roundness=0.6, converter=hls_to_gray, context=None):
    # assume img given as HLS
    img = filter_image(img, lower, upper)
    # might be able to simplify
    img = converter(img)
    # messing with the threshold changes result - lower is more permissive.
    _res, img = cv.threshold(img, 63, 255, cv.THRESH_BINARY)
    # img is now a binary image
    
    contours, heirarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if heirarchy is None:
        # No match
        return []
    matches = []
    img_size = img.shape[0] * img.shape[1]
    for (c, h) in zip(contours, heirarchy[0]):
        true_perim = cv.arcLength(c, True)
        c = cv.approxPolyDP(c, true_perim * noise_filter, True)
        area = cv.contourArea(c)
        relative_area = area / img_size
        if relative_area < 0.001 or h[3] != -1:
            continue

        circ = circularity(area, cv.arcLength(c, True))
        # pretty round and has a child that has no children
        # The last bit is probably overkill and definitely will result in false negatives, 
        # but it's cheap and guarantees a hollow shape
        complete_match = circ > roundness \
            and relative_area > 0.001 \
            and h[2] != -1 \
            and heirarchy[0][h[2]][2] == -1
        
        match = Match(*cv.boundingRect(c), c, complete_match)
        if context is not None:
            match = match.with_context(context)
        matches.append(match)

    return matches