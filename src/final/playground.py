import math


def line_circle_intersect(p1, p2, circle_center, radius):
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = circle_center

    dx = x2 - x1
    dy = y2 - y1
    m = dy / dx

