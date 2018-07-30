#!/usr/bin/env python
#########################################################################
# File Name: benchmark.py
# Author: gaoyu
# mail: gaoyu@momenta.ai
# Created Time: 2018-05-22 14:27:37
#########################################################################

import os
import argparse
import json

import lanes

def parser():
    p = argparse.ArgumentParser()
    p.add_argument("config", help = "config file")
    p.add_argument("-P", "--pr-json", help = "predict json path")

    g = p.add_argument_group("Param")
    g.add_argument("-I", "--iou", help = "min iou")
    g.add_argument("-W", "--width", help = "line width use in calc iou")

    g = p.add_argument_group("Dataset")
    g.add_argument("-L", "--list", help = "image list")
    g.add_argument("-G", "--gt-json", help = "groundtruth json path")
    return p

def process_image(sub_name, args):
    pr_json = os.path.join(args["pr_json"], sub_name[:-4] + ".json")
    gt_json = os.path.join(args["gt_json"], sub_name[:-4] + ".json")
    try:
        pr_lines = json.load(open(pr_json, 'r'))
        gt_lines = json.load(open(gt_json, 'r'))
    except:
        msg = "Error loading json: %s %s" % (pr_json, gt_json)
        return -1, msg, msg
    return lanes.line_matching(pr_lines, gt_lines, args)

def process_image_list(args):
    Crt, Pr, Gt = 0, 0, 0
    for line in open(args["list"]):
        crt, pr, gt = process_image(line.strip().split()[0], args)
        if crt == -1:
            return {
                    "code" : 1,
                    "msg" : pr,
            }
        Crt += crt
        Pr += pr
        Gt += gt

    if Crt == 0 or Pr == 0 or Gt == 0:
        return {
                "code" : 1,
                "msg" : "No correct lane",
        }

    Precision = float(Crt) / Pr
    Recall = float(Crt) / Gt
    F1 = 2.0 / (1.0 / Precision + 1.0 / Recall)
   
    return {
            "code" : 0,
            "f1" : F1,
            "precision" : Precision,
            "recall" : Recall,
            }

if __name__ == "__main__":
    import yaml
    user_args = parser().parse_args()
    args = yaml.load(open(user_args.config, 'r'))
    args.update({k : v for k, v in vars(user_args).items() if not v is None})
    output = process_image_list(args)
    for key, value in sorted(output.items()):
        print "%s : %s" % (str(key), str(value))
