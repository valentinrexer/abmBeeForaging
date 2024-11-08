import math


def circle_line_intersect(p1, p2, circle_center, radius):
    # Extract coordinates
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = circle_center

    # Vector from p1 to p2
    dx = x2 - x1
    dy = y2 - y1

    # Vector from p1 to circle center
    pcx = cx - x1
    pcy = cy - y1

    # Length of line segment squared
    line_length_sq = dx * dx + dy * dy

    # Skip if line segment has zero length
    if line_length_sq == 0:
        # Check if p1 is within circle
        return math.sqrt(pcx * pcx + pcy * pcy) <= radius

    # Project circle center onto line segment
    proj = (pcx * dx + pcy * dy) / line_length_sq

    # Find closest point on line segment to circle center
    if proj < 0:
        closest_x, closest_y = x1, y1
    elif proj > 1:
        closest_x, closest_y = x2, y2
    else:
        closest_x = x1 + proj * dx
        closest_y = y1 + proj * dy

    # Calculate distance from closest point to circle center
    distance = math.sqrt(
        (closest_x - cx) * (closest_x - cx) +
        (closest_y - cy) * (closest_y - cy)
    )

    # Compare with radius
    return distance <= radius


print(circle_line_intersect((20,20), (3,3), (4,5), 0.701))