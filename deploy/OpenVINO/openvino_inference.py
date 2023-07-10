#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import logging as log
import os
import sys

import cv2
import numpy as np
import time

from openvino.inference_engine import IECore

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis


def make_parser() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser("openvino inference sample")
    parser.add_argument(
        "demo",
        default="image",
        help="demo type, eg. image, video and webcam"
    )
    parser.add_argument(
        '-m',
        '--model',
        required=True,
        type=str,
        help='Required. Path to an .xml or .onnx file with a trained model.')
    parser.add_argument(
        '-i',
        '--input',
        # required=True,
        type=str,
        help='Required. Path to an image file.')
    parser.add_argument(
        '-o',
        '--output_dir',
        type=str,
        default='demo_output',
        help='Path to your output dir.')
    parser.add_argument(
        '-s',
        '--score_thr',
        type=float,
        default=0.1,
        help="Score threshould to visualize the result.")
    parser.add_argument(
        "-n",
        "--nms_thr",
        type=float,
        default=0.4,
        help="test nms threshold",
    )
    parser.add_argument(
        '-d',
        '--device',
        default='CPU',
        type=str,
        help='Optional. Specify the target device to infer on; CPU, GPU, \
              MYRIAD, HDDL or HETERO: is acceptable. The sample will look \
              for a suitable plugin for device specified. Default value \
              is CPU.')
    parser.add_argument(
        '--labels',
        default=None,
        type=str,
        help='Option:al. Path to a labels mapping file.')
    parser.add_argument(
        '-nt',
        '--number_top',
        default=10,
        type=int,
        help='Optional. Number of top results.')
    parser.add_argument(
        "--camid",
        type=int,
        default=0,
        help="webcam demo camera id")
    return parser

class Predictor(object):
    def __init__(self,net,exec_net,args,input_blob,out_blob):
        self.args = args
        self.net = net
        self.input_blob = input_blob
        self.out_blob = out_blob
        self.exec_net = exec_net

    def inference(self, ori_img):
        
        _, _, h, w = self.net.input_info[self.input_blob].input_data.shape
        image, ratio = preprocess(ori_img, (h, w))
        
        log.info('Starting inference in synchronous mode')
        t0 = time.time()
        res = self.exec_net.infer(inputs={self.input_blob: image})

        res = res[self.out_blob]
        
        predictions = demo_postprocess(res, (h, w), p6=False)[0]
        
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=self.args.nms_thr, score_thr=self.args.score_thr)
        fps = 1/(time.time() - t0)
        print("fps: {:.4f}".format(fps))
        return dets, fps

def imageflow_demo(predictor,args):
    cap = cv2.VideoCapture(args.input if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_all = 0
    cout = 0
    mkdir(args.output_dir)
    if args.demo == "video":
        save_path = os.path.join(args.output_dir, args.input.split("/")[-1])
    else:
        save_path = os.path.join(args.output_dir, "camera.mp4")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    while (True):
        ret_val, frame = cap.read()
        if ret_val:
            dets,fps = predictor.inference(frame)
            cout += 1
            fps_all += fps
            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                result_frame = vis(frame, final_boxes, final_scores, conf=args.score_thr)
            cv2.imshow('frame', result_frame)
            vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
    print(fps_all/cout)
    print(cout)

def image_demo(predictor,args):
    origin_img = cv2.imread(args.input)
    dets = predictor.inference(origin_img)
    if dets is not None:
        final_boxes = dets[:, :4]
        final_scores, final_cls_inds = dets[:, 4], dets[:, 5]
        origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                         conf=args.score_thr, class_names=COCO_CLASSES)

    mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, args.input.split("/")[-1])
    cv2.imwrite(output_path, origin_img)

def main(args):
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    # ---------------------------Step 1. Initialize inference engine core--------------------------------------------------
    log.info('Creating Inference Engine')
    ie = IECore()

    # ---------------------------Step 2. Read a model in OpenVINO Intermediate Representation or ONNX format---------------
    log.info(f'Reading the network: {args.model}')
    # (.xml and .bin files) or (.onnx file)
    net = ie.read_network(model=args.model)

    if len(net.input_info) != 1:
        log.error('Sample supports only single input topologies')
        return -1
    if len(net.outputs) != 1:
        log.error('Sample supports only single output topologies')
        return -1

     # ---------------------------Step 3. Configure input & output----------------------------------------------------------
    log.info('Configuring input and output blobs')
    # Get names of input and output blobs
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    # Set input and output precision manually
    net.input_info[input_blob].precision = 'FP32'
    net.outputs[out_blob].precision = 'FP16'

    # Get a number of classes recognized by a model
    num_of_classes = max(net.outputs[out_blob].shape)

    # ---------------------------Step 4. Loading model to the device-------------------------------------------------------
    log.info('Loading the model to the plugin')
    exec_net = ie.load_network(network=net, device_name=args.device)

    predictor  = Predictor(net,exec_net,args,input_blob,out_blob)

    if args.demo == "image":
        image_demo(predictor,args)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor,args)

if __name__ == '__main__':
    args = make_parser().parse_args()

    sys.exit(main(args))
