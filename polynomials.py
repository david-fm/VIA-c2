import numpy as np
from math import cos, sin
import cv2 as cv
from random import randint
import timeit

from typing import List, Dict
from collections import namedtuple

pt = namedtuple("Point", "x y")

def polynomialModel(center: pt, radius: List[int], fractions: int = 0) -> List[pt]:
    """Create a polynomial model from a set of circles moving in a clockwise direction and selecting random point between the circumferences"""
    assert fractions >= 0, "The fractions must be a positive number"
    assert len(radius) > 1, "The radius must be a list with at least two elements"
    assert len(center) == 2, "The center must be a tuple with two elements"

    def pointInCircumference(center: pt, radius: int, angle: int) -> pt:
        """Return the point in the circumference"""
        x = center.x + radius * cos(angle)
        y = center.y + radius * sin(angle)
        return pt(int(x), int(y))
    

    fractions = fractions if fractions > 0 else randint(1, 10)
    angles = np.linspace(0, 2*np.pi, fractions+1)

    points = [pointInCircumference(center, radius[randint(0,len(radius)-1)], a) for a in angles]

    return points

def lPolynomial2nlPolynomial(points: List[pt]) -> List[pt]:
    """Convert a linear polynomial to a non linear polynomial using 
    2nd degree bezier curves"""
    assert len(points) % 2 == 0, "The number of points must be divisible by 2"

    def bezier(a: pt, b: pt, c:pt) -> List[pt]:
        """Create a bezier curve given three control points"""
        points: Dict[pt, bool] = {}

        p1 = a
        p2 = b
        p3 = c

        for t in np.linspace(0,1,11):
            x = p1.x * (1-t)**2 + p2.x * 2*t*(1-t) + p3.x * t**2
            y = p1.y * (1-t)**2 + p2.y * 2*t*(1-t) + p3.y * t**2

            point = pt(int(x), int(y))
            points[point] = True
        return list(points.keys())
    
    tM = np.array([[(1-t)**2, 2*t*(1-t), t**2] for t in np.linspace(0,1,11)])

    def bezier_improved(a: pt, b: pt, c: pt) -> List[pt]:
        """Create a bezier curve given three control points"""

        p1 = a
        p2 = b
        p3 = c

        pM = np.array([[p1.x, p1.y], [p2.x, p2.y], [p3.x, p3.y]])

        result =  (tM @ pM).astype(int)
        resultDict = {pt(x,y): True for x,y in result}

        return resultDict.keys()
    
    

    new_points = []
    for i in range(0,len(points)-2,2):
        timesImproved = timeit.timeit(lambda: bezier_improved(points[i], points[i+1], points[i+2]), number=1000)
        times = timeit.timeit(lambda: bezier(points[i], points[i+1], points[i+2]), number=1000)
        print(f"Improvement: {times/timesImproved}")
        result = bezier_improved(points[i], points[i+1], points[i+2])
        new_points.extend(result)

    i = len(points)-2
    result = bezier_improved(points[i], points[i+1], points[0])
    new_points.extend(result)

    return new_points

def maskFromPolygons(polygon: List[pt], size: tuple):
    """Create a mask from a set of polygons"""
    mask = np.zeros(size, dtype=np.uint8)
    cv.fillPoly(mask, polygon, 255)
    return mask


if __name__ == "__main__":
    box = np.zeros((400,400,3), dtype=np.uint8)

    points: List[pt]  = polynomialModel(pt(200,200), [200,100,50,110,150,130,140], 15)
    nlPoints: List[pt] = lPolynomial2nlPolynomial(points)
    mask = maskFromPolygons([np.array(nlPoints, np.int32)], (400,400))
    #cv.polylines(box, [np.array(points, np.int32)], isClosed=True, color=255, thickness=1)
    cv.polylines(box, [np.array(nlPoints, np.int32)], isClosed=False, color=(255,255,255), thickness=1)

    # Draw the points in the box
    # for i,p in enumerate(nlPoints):
    #     cv.circle(box, (p.x, p.y), 2, (0,255,0), -1)
    #     cv.putText(box, str(i), (p.x, p.y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv.imshow("mask", mask)
    cv.imshow("box", box)
    # waits for user to press any key 
    # (this is necessary to avoid Python kernel form crashing) 
    cv.waitKey(0) 
    
    # closing all open windows 
    cv.destroyAllWindows() 
