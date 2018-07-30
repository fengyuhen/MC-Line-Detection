#!/usr/bin/env python
#########################################################################
# File Name: lanes.py
# Author: gaoyu
# mail: gaoyu@momenta.ai
# Created Time: 2018-05-22 13:30:24
#########################################################################

import json
import cv2
import numpy as np
from munkres import Munkres

def cvpoint(p):
    return (int(float(p['x']) + 0.5), int(float(p['y']) + 0.5))

def calc_iou(line1, line2, args):
    pic = np.zeros(eval(args["size"]), np.uint8)
    width = int(args["width"])
    
    for i in xrange(len(line1) - 1):
        cv2.line(pic, cvpoint(line1[i]), cvpoint(line1[i + 1]), 1, width)
    line1_size = (pic == 1).sum()
    for i in xrange(len(line2) - 1):
        cv2.line(pic, cvpoint(line2[i]), cvpoint(line2[i + 1]), 2, width)
    line2_size = (pic == 2).sum()
    u = (pic == 1).sum()
    if line2_size + u < 1e-4:
        return 0
    return float(line1_size - u) / (line2_size + u)

def line_matching(pr_lines, gt_lines, args):
    pr_lines = pr_lines["Lanes"]
    gt_lines = gt_lines["Lanes"]
    if not pr_lines or not gt_lines:
        return 0, 0, 0
    cost_mat = [ [0 for j in gt_lines] for i in pr_lines ]

    for i, l1 in enumerate(pr_lines):
        for j, l2 in enumerate(gt_lines):
            iou = calc_iou(l1, l2, args)
            if iou > args["iou"]:
                cost_mat[i][j] = 1.0 - iou #iou lower->cost higher
            else:
                cost_mat[i][j] = 1e10

    correct_cnt = len(filter(lambda v : cost_mat[v[0]][v[1]] < 1e5, Munkres().compute(cost_mat)))
    #print correct_cnt, len(pr_lines), len(gt_lines)
    return correct_cnt, len(pr_lines), len(gt_lines)
