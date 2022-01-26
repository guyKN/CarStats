import numpy as np

from drawingUtil import Line


def lines_cross(line: Line, x21, y21, x22, y22: np.array):
    """
    Given two line segments, returns true if they cross.
    The second line may also be passed as a np array of coordinates, in which case an array is returned, which is true in every index where the lines cross.
    :param line: should be in ((x1, y1), (x2, y2)) format
    :param x21:
    :param y21:
    :param x22:
    :param y22: the two endpoints of the second line, or np arrays of endpoints of multiple lines.
    :return: True if the lines cross.
    """
    (x11, y11), (x12, y12) = line
    m1 = (y12 - y11) / (x12 - x11)
    m2 = (y22 - y21) / (x22 - x21)
    x_cross = (m1 * x11 - m2 * x21 + y21 - y11) / (m1 - m2)

    return (np.minimum(x11, x12) < x_cross) & \
           (x_cross < np.maximum(x11, x12)) & \
           (np.minimum(x21, x22) < x_cross) & \
           (x_cross < np.maximum(x21, x22))