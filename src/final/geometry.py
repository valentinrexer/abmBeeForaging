from __future__ import annotations

import math
import random

class Calc:
    """
    Class for all static methods for calculations and geometry
    """

    @staticmethod
    def generate_random_point(origin_x : int | float,
                              origin_y: int | float,
                              target_distance : int | float,
                              tolerance: float =0.01,
                              max_attempts : int=10000) -> tuple[float, float] | None:
        """
        Generate a random point on the grid with a given distance to a given point (the hive in our model)

        :param origin_x: x coordinate of the given point
        :param origin_y: y coordinate of the given point
        :param target_distance: the distance between the given point and the new point
        :param tolerance: max difference between the desired and actual distance between the given point and the new point
        :param max_attempts: max number of attempts to find a point on the grid that fulfills all criteria (to prevent an infinity loop)
        :return: a random point with approximately the given distance to the origin point
        """
        for _ in range(max_attempts):
            angle = random.uniform(0, 2* math.pi)

            # Add some randomness to the distance within the tolerance range
            min_distance = target_distance * (1 - tolerance)
            max_distance = target_distance * (1 + tolerance)
            actual_distance = random.uniform(min_distance, max_distance)

            # Calculate the point using polar coordinates
            x = origin_x + actual_distance * math.cos(angle)
            y = origin_y + actual_distance * math.sin(angle)

            # If both coordinates are positive, return the point
            if x >= 0 and y >= 0:
                return round(x, 5), round(y, 5)

    @staticmethod
    def get_next_point(current_x : int | float,
                       current_y : int | float,
                       angle : float, distance : float) -> tuple[float, float] | None:
        """
            Returns the next point given a start point, direction(angle) and distance

            Args:
                current_x (int): x coordinate of the current point
                current_y (int): y coordinate of the current point
                angle (float): angle between 0 and 2pi that determines the direction of movement
                distance (float): distance that's to be covered
        """
        x_next = current_x + distance * math.cos(angle)
        y_next = current_y + distance * math.sin(angle)
        return round(x_next, 2), round(y_next, 2)

    @staticmethod
    def get_distance(pos1 : tuple[float, float], pos2 : tuple[float, float]) -> float:
        """
            Use euclidian distance to calculate the distance between two points
        """
        return round(math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2), 2)

    @staticmethod
    def get_angle(starting_point: tuple[float, float], destination_point : tuple[float, float]) -> float:
        """
            the angle in a triangle is calculated by arctan(y/x)

            We use this principle to calculate the angle/direction in which an object would have to
            be moved to reach a given point
        """

        dx = destination_point[0] - starting_point[0]
        dy = destination_point[1] - starting_point[1]

        angle = math.atan2(dy, dx)

        return Calc.normalize_angle(angle)

    @staticmethod
    def random_deviate_angle(current_angle : float,
                             mean : int | float,
                             standard_deviation : int | float,
                             min_value: int | float,
                             max_value: int | float,
                             radians : bool =False) -> float:
        """
        Alters a given angle by a randomly drawn value

        :param current_angle: angle to be deviated
        :param mean: mean value of the normal distribution function
        :param standard_deviation: sigma value of the normal distribution function
        :param min_value: min value of the normal distribution
        :param max_value: max value of the normal distribution
        :param radians: input is in radians
        :return: deviated angle
        """
        deviation = Calc.draw_normal_distributed_value(mean, standard_deviation, min_value, max_value)

        if not radians:
            deviation = math.radians(deviation)

        current_angle += deviation
        current_angle = current_angle % (2 * math.pi)

        if current_angle < 0:
            current_angle += 2 * math.pi

        return current_angle

    @staticmethod
    def random_deviate_angle_equally(angle : float,
                                     min_value : int | float,
                                     max_value : int | float,
                                     radians : bool=False):
        """
        Deviates a given value by a random uniform value in a given interval

        :param angle: initial angle to be deviated
        :param min_value: min deviation
        :param max_value: max deviation
        :param radians: angle provided in radians already
        :return: deviated angle
        """

        deviation = random.uniform(min_value, max_value)

        if radians:
            angle += deviation

        else:
            angle += math.radians(deviation)

        return Calc.normalize_angle(angle)

    @staticmethod
    def normalize_angle(current_angle : float) -> float:
        """
        brings angle into normalized format 0 <= angle <= 2 * pi

        :param current_angle: angle to be normalized
        :return: normalized angle
        """
        angle = current_angle % (2 * math.pi)

        if angle < 0:
            angle += 2 * math.pi

        return angle

    @staticmethod
    def radians_to_degrees(radians : int | float) -> float:
        """
        Converts a value in radians to degrees

        :param radians: angle in radians
        :return: angle in degrees
        """
        return radians * (180 / math.pi)

    @staticmethod
    def circle_line_intersect(p1 : tuple[float, float],
                              p2 : tuple[float, float],
                              circle_center : tuple[float, float],
                              radius : float) -> bool:
        """
        Calculates if a line intersects a circle

        :param p1: first end of the line
        :param p2: second end of the line
        :param circle_center: center of the circle
        :param radius: radius of the circle
        :return: bool if the line intersects the circle
        """
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
        line_length_sq = dx ** 2 + dy ** 2

        # Skip if line segment has zero length
        if line_length_sq == 0:
            # Check if p1 is within circle
            return math.sqrt(pcx ** 2 + pcy ** 2) <= radius

        # Project circle center onto line segment
        proj = (pcx * dx + pcy * dy) / line_length_sq

        # Find the closest point on the line segment to circle center
        if proj < 0:
            closest_x, closest_y = x1, y1

        elif proj > 1:
            closest_x, closest_y = x2, y2

        else:
            closest_x = x1 + proj * dx
            closest_y = y1 + proj * dy

        # Calculate distance from the closest point to circle center
        distance = math.sqrt((closest_x - cx) ** 2 + (closest_y - cy) ** 2)

        # Compare with radius
        return distance <= radius

    @staticmethod
    def draw_normal_distributed_value(mean : int | float,
                                      standard_deviation : int | float,
                                      min_value : int | float,
                                      max_value : int | float) -> float:
        """
        Returns a random value drawn from the normal distribution function
        :param mean: mean of the normal distribution function
        :param standard_deviation: sigma value of the normal distribution function
        :param min_value: min value of the normal distribution function
        :param max_value: max value of the normal distribution function
        :return: normal distributed random value
        """
        while True:
            value = random.normalvariate(mean, standard_deviation)
            if min_value <= value <= max_value:
                return value