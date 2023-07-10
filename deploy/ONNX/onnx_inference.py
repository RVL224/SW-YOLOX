#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
import numpy as np
from loguru import logger
import time

import onnxruntime

from yolox.data.data_augment import preproc as preprocess
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "demo", 
        default="image", 
        help="demo type, eg. image, video and webcam"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default='test_image.png',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.015,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "-n",
        "--nms_thr", 
        type=float,
        default=0.3,  
        help="test nms threshold",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        #default="608,1088",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    return parser

class Predictor(object):
    def __init__(self, args):
        self.args = args
        self.session = onnxruntime.InferenceSession(args.model)
        self.input_shape = tuple(map(int, args.input_shape.split(',')))

    def inference(self, ori_img):
        img_info = {"id": 0}
        height, width = ori_img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = ori_img

        img, ratio = preprocess(ori_img, self.input_shape)
        img_info["ratio"] = ratio
        ort_inputs = {self.session.get_inputs()[0].name: img[None, :, :, :]}
        output = self.session.run(None, ort_inputs)
        predictions = demo_postprocess(output[0], self.input_shape, p6=self.args.with_p6)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=self.args.nms_thr, score_thr=self.args.score_thr)
        return dets
    
def image_demo(predictor, args):
    origin_img = cv2.imread(args.image_path)
    dets = predictor.inference(origin_img)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        origin_img = vis(origin_img, final_boxes, final_scores, conf=args.score_thr)

    mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, args.image_path.split("/")[-1])
    cv2.imwrite(output_path, origin_img)

def imageflow_demo(predictor, args):
    cap = cv2.VideoCapture(args.image_path if args.demo == "video" else args.camid)
    start_time = time.time()
    counter = 0
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    mkdir(args.output_dir)
    if args.demo == "video":
        save_path = os.path.join(args.output_dir, args.image_path.split("/")[-1])
    else:
        save_path = os.path.join(args.output_dir, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    while (True):
        ret, frame = cap.read()
        counter += 1  # 计算帧数
        if (time.time() - start_time) != 0:  # 实时显示帧数
            dets = predictor.inference(frame)
            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                result_frame = vis(frame, final_boxes, final_scores, conf=args.score_thr)
            cv2.putText(result_frame, "FPS {0}".format(float('%.1f' % (counter / (time.time() - start_time)))), (500, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                        3)
            cv2.imshow('frame', result_frame)
            print("FPS: ", counter / (time.time() - start_time))
            counter = 0
            start_time = time.time()
            time.sleep(1 / fps)  # 按原帧率播放
            vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break

if __name__ == '__main__':
    args = make_parser().parse_args()
    predictor = Predictor(args)

    if args.demo == "image":
        image_demo(predictor, args)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, args)