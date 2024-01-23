from dataclasses import dataclass
from typing import Optional

import cv2 as cv

@dataclass
class Point:
    x: int
    y: int

    def offset(self, origin) -> "Point":
        return Point(origin.x - self.x, origin.y - self.y)

class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.width = w
        self.height = h

    @property
    def top(self) -> int:
        return self.y
    @property
    def left(self) -> int:
        return self.x
    @property
    def bottom(self) -> int:
        return self.y + self.height
    @property
    def right(self) -> int:
        return self.x + self.width
        
    def top_left(self) -> Point:
        return Point(self.left, self.top)
        
    def top_right(self) -> Point:
        return Point(self.right, self.top)
        
    def bottom_left(self) -> Point:
        return Point(self.left, self.bottom)
        
    def bottom_right(self) -> Point:
        return Point(self.right, self.bottom)

    def center(self) -> Point:
        x = (self.left + self.right) // 2
        y = (self.top + self.bottom) // 2
        return Point(x, y)

    def major_axis(self) -> int:
        return max(self.width, self.height)

    def minor_axis(self) -> int:
        return min(self.width, self.height)

class Match:
    full: bool
    box: BoundingBox

    def __init__(self, x: int, y: int, w: int, h: int, contour, full):
        self.full = full
        self.box = BoundingBox(x, y, w, h)
        self.contour = contour

    def is_full(self) -> bool:
        return self.full

    def is_partial(self) -> bool:
        return not self.full

    def with_context(self, context: "CameraContext") -> "ContextualMatch":
        return ContextualMatch(self.box.x, self.box.y, 
                               self.box.width, self.box.height, 
                               self.contour, 
                               self.full, 
                               context)

    def show(self, img, full_match_color=(63,255,0), partial_match_color=(0,255,255), line_weight=3):
        #img is assumed to be in BGR for the defaults, but it doesn't actually matter
        if self.is_full():
            color = full_match_color
        else:
            color = partial_match_color
        b = self.box
        return cv.rectangle(img, (b.left, b.top), (b.right, b.bottom), color, int(line_weight))
        


class CameraContext:
    image_dimensions: tuple[int, int]
    # Camera FOV in degrees
    fov: Optional[float]
    # Object width in pixels at 1 meter
    object_width: Optional[int]

    def __init__(self, width, height, fov = None, object_width = None):
        self.image_dimensions = (width, height)
        self.fov = fov
        self.object_width = object_width

    def center(self) -> Point:
        return Point(self.image_dimensions[0]//2, self.image_dimensions[1]//2)

    @property
    def width(self) -> int:
        return self.image_dimensions[0]

    @property
    def height(self) -> int:
        return self.image_dimensions[1]

class ContextualMatch(Match):
    context: CameraContext
    
    def __init__(self, x, y, w, h, contour, full, context):
        super().__init__(x, y, w, h, contour, full)
        self.context = context

    def relative_box(self) -> BoundingBox:
        origin = self.context.center()
        return BoundingBox(self.box.x - origin.x, self.box.y - origin.y, self.box.width, self.box.height)

    def normal_position(self) -> tuple[float, float]:
        width = self.context.width
        height = self.context.height
        pos = self.box.center()
        x_norm = pos.x / width
        y_norm = pos.y / height
        x_norm = x_norm * 2 - 1
        y_norm = y_norm * 2 - 1
        return (x_norm, -y_norm)
    
    def yaw(self) -> Optional[float]:
        if self.context.fov is None or self.is_partial():
            return None
        return self.normal_position()[0] * self.context.fov

    def pitch(self) -> Optional[float]:
        if self.context.fov is None or self.is_partial():
            return None
        return self.normal_position()[1] * self.context.fov

    def distance(self) -> Optional[float]:
        if self.context.object_width is None or self.is_partial():
            return None

        # This may not be a linear thing. Seems to work beyond a meter, but up close might get dubious
        ratio = self.context.object_width / self.box.major_axis()
        return ratio
    