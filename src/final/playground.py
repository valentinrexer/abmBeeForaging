import mesa
import bee_foraging_model
import random
import math


def generate_random_point(center_x, center_y, target_distance, tolerance=0.1):
    while True:
        # Generate a random angle in radians (between 0 and π/2 for positive coordinates)
        angle = random.uniform(0, math.pi / 2)

        # Add some randomness to the distance within the tolerance range
        min_distance = target_distance * (1 - tolerance)
        max_distance = target_distance * (1 + tolerance)
        actual_distance = random.uniform(min_distance, max_distance)

        # Calculate the point using polar coordinates
        x = center_x + actual_distance * math.cos(angle)
        y = center_y + actual_distance * math.sin(angle)

        # If both coordinates are positive, return the point
        if x >= 0 and y >= 0:
            return round(x, 2), round(y, 2)


def get_surrounding(position, distance):
    min_x, max_x = int(position[0] - 1.5 * distance), int(position[0] + 1.5 * distance)
    min_y, max_y = int(position[1] - 1.5 * distance), int(position[1] + 1.5 * distance)

    surrounding = []

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            if math.sqrt((x - position[0]) ** 2 + (y - position[1]) ** 2) <= distance:
                surrounding.append((x, y))

    return surrounding

x = 2524
y = 2266
d = 10
surr = get_surrounding((x,y), d)

print(surr)

'''
for y in range(100):
    for x in range(100):
        if surr.__contains__((x, y)):
            print("_", end='')

        else:
            print("0", end='')

    print()
'''
