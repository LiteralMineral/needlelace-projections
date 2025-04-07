import math
from math import sin, cos, atan2, atan, sqrt
from typing import Tuple, Union
from numpy import number
import numpy as np

# <editor-fold desc="Setting up the coordinate types">
Point3 = Tuple[number, number, number]  # defining how the vector works.
Point2 = Tuple[number, number]  # defining how the vector works
Coordinate = Union[Point2, Point3]


# </editor-fold>


def cartesian_to_polar2(x, y):
    r = sqrt((x ** 2) + (y ** 2))
    theta = atan2(y, x)
    return (r, theta)


# region basic mappings....
def cartesian_to_polar(p: Point2):
    (x, y) = p
    r = sqrt(x ** 2 + y ** 2)
    theta = atan2(y, x)
    return (r, theta)


def polar_to_cartesian(p: Point2):
    (r, theta) = p
    return r * cos(theta), r * sin(theta)


def translate(p: Point2, s: Point2):
    return p[0] + s[0], p[1] + s[1]


def scale(p: Point2, factor):
    p = p[0] * factor, p[1] * factor
    return p


def stereographic_projection(p: Point2, c=(0.0, 0.0)):
    p = p[0] - c[0], p[1] - c[1]
    (r, theta) = cartesian_to_polar(p)
    # project polar to spherical (phi, theta)
    if r == 0:
        return math.pi, theta
    else:
        return (2 * atan(1 / r), theta)  # (2 * arctan(1/r), theta)


# phi is zenith angle, so latitude. theta is azimuth, so longitude
def stereographic_projection_i(p: Point2):
    # assumes range is lat:(0, pi) and theta: (-pi, pi)
    phi, theta = p[0], p[1]
    # project spherical coords back to polar coords.
    if math.tan(phi / 2) == 0.0:  # if there's a divide-by-zero issue
        p = 0.0, 0.0
    else:
        p = 1 / math.tan(phi / 2), theta  # cot(phi/2) == 1 / cot(phi/2)
    # polar to cartesian coordinates
    p = polar_to_cartesian(p)

    return p[0], p[1]


# phi is zenith angle, so latitude. theta is azimuth, so longitude
def sinusoidal_projection(p: Point2, num_partitions):
    (lat, lon) = p

    # calculate the meridian
    steps = num_partitions / 2
    median = (((steps) * lon // np.pi) + 0.5) * np.pi / (steps)

    return median + ((lon - median) * np.sin(lat)), lat


def sinusoidal_projection_i(p: Point2, num_partitions):
    (x, y) = p

    # calculate the meridian
    steps = num_partitions / 2
    meridian = (((steps) * x // np.pi) + 0.5) * np.pi / (steps)  # keep in mind that the median is
    # compared with the x coord, not y. because different coordinate systems.
    # print(meridian)
    # median + ((lon - median) * np.sin(lat)) = x
    # median + ((lon - median) * np.sin(y)) = x
    # ((lon - median) * np.sin(y))          = x - median
    # (lon - median)                        = (x - median) / np.sin(y)
    # lon                                   = ((x - median) / np.sin(y)) + media
    lat = y
    if np.sin(y) == 0.0:
        lon = meridian
    else:
        lon = ((x - meridian) / np.sin(
            y)) + meridian  # TODO: see if there's a divide-by-zero issue here
    # if the original point is outside of the sinusoid, return

    offset = np.pi/num_partitions
    lbound, ubound = meridian - offset,  meridian + offset
    output_is_in_sinusoid = lbound <= lon <= ubound

    return lat, lon, output_is_in_sinusoid

def sinusoidal_projection_i2(p: Point2, meridian=np.pi):
    # assumes the median is pi... since the range is [0,2pi)
    (x, y) = p

    # median + ((lon - median) * np.sin(lat)) = x
    # median + ((lon - median) * np.sin(y)) = x
    # ((lon - median) * np.sin(y))          = x - median
    # (lon - median)                        = (x - median) / np.sin(y)
    # lon                                   = ((x - median) / np.sin(y)) + median
    lat = y
    if np.sin(y) == 0.0:
        # lat = np.pi/2
        lon = meridian
    else:
        lon = ((x - meridian) / np.sin(
            y)) + meridian  # TODO: see if there's a divide-by-zero issue here
    return lat, lon


def is_in_sinusoidal_projection(p: Point2, num_partitions):
    # answers the question 'does the point p lie within the output of a
    # sinusoidal projection performed with num_partitions sections?'

    (lat, lon) = p
    # calculate meridian.
    steps = num_partitions / 2
    median = (((steps) * lon // np.pi) + 0.5) * np.pi / (steps)
    section_width = np.pi / num_partitions
    offset = (section_width * np.sin(lat))
    l, u = median - offset, median + offset  # lower and upper bound for the section at this latitude.
    # print(l, u)
    return l <= lon and lon <= u  # is point p between those bounds?


def calculate_median(lon, num_partitions):
    steps = num_partitions / 2
    # calculate the medians
    median = (((steps) * lon // np.pi) + 0.5) * np.pi / (steps)
    return median


def custom_projection(p: Point2, c: Point2, m):
    # sinusoidal(stereographic(point)).
    # assumes that the points have already been scaled [0.0, 1.0)
    p = (p[0] - c[0]), (p[1] - c[1])  # center the point with respect to origin = (0, 0).
    p = sinusoidal_projection(stereographic_projection(p, c),
                              m)
    return p


def custom_projection_i(p: Point2, m, c=(0.0, 0.0)):
    # stereographic_i(sinusoidal_i(point)).
    p = stereographic_projection_i(sinusoidal_projection(p, m), c)
    return p


# endregion


# class that handles projection in a hopefully less memory-intensive way.
class Projector:
    def __init__(self):
        # for piecewise ranges/conditions
        self.l_bounds = None
        self.u_bounds = None
        self.ranges = None
        self.medians = None

        # the functions the projector can apply
        self.sinusoidal_funcs = None

        # setting up the above values.
        self.num_partitions = 6
        self.set_bounds(self.num_partitions)

        # based on the boundaries, write the functions.
        self.set_funcs()

    # set the lower and upper bounds for piecewise functions
    def set_bounds(self, num_partitions: int):
        self.l_bounds = (np.linspace(0.0, 2 * np.pi,
                                     num=num_partitions,
                                     endpoint=False))
        self.u_bounds = (np.linspace(self.l_bounds[1], 2 * np.pi,
                                     num=num_partitions,
                                     endpoint=True))
        self.ranges = (
            list(zip(self.l_bounds, self.u_bounds)))  # the ranges for making the condlist
        self.medians = [(bound[0] + bound[1]) / 2 for bound in self.ranges]
        pass

    # set the piecewise functions based on the bounds
    def set_funcs(self):
        self.set_interrupted_sinusoidal()
        # self.set_interrupted_elliptical() #TODO
        pass

    # calculates the values.... assuming the bounds have already been calculated
    def interrupted_sinusoidal(self, x):
        lat, lon = x
        conds = ([x[1] <= bounds[1] for bounds in self.ranges])

        return sinusoidal_projection(x, np.select(conds, self.medians))

    # recalculates the functions given the current ranges.
    def set_interrupted_sinusoidal(self):
        self.sinusoidal_funcs = [lambda x: sinusoidal_projection(x,
                                                                 median=(bounds[0] + bounds[1]) / 2)
                                 for bounds in self.ranges]  # save the functions!!
