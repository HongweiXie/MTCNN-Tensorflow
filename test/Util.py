from operator import itemgetter
import numpy as np


'''
Function:
	calculate Intersect of Union
Input:
	rect_1: 1st rectangle
	rect_2: 2nd rectangle
Output:
	IoU
'''


def IoU(rect_1, rect_2):
    x11 = rect_1[0]  # first rectangle top left x
    y11 = rect_1[1]  # first rectangle top left y
    x12 = rect_1[2]  # first rectangle bottom right x
    y12 = rect_1[3]  # first rectangle bottom right y
    x21 = rect_2[0]  # second rectangle top left x
    y21 = rect_2[1]  # second rectangle top left y
    x22 = rect_2[2]  # second rectangle bottom right x
    y22 = rect_2[3]  # second rectangle bottom right y
    x_overlap = max(0, min(x12, x22) - max(x11, x21))
    y_overlap = max(0, min(y12, y22) - max(y11, y21))
    intersection = x_overlap * y_overlap
    union = (x12 - x11) * (y12 - y11) + (x22 - x21) * (y22 - y21) - intersection
    if union == 0:
        return 0
    else:
        return float(intersection) / union