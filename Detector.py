# -*- coding: utf-8 -*-
"""detection+projection tasks.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZhZpF_Mt004Qlu8umYf7d54-ZvMpkb16
"""

import numpy as np
import pandas as pd
import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from numpy.linalg import norm
from itertools import compress

class Detector():

  def __init__(self, videofile):
    self.output = None
    self.frames = []
    self.homography = None
    self.objects = None
    self.keypoints = None
    self.gt_coords = None
    self.filter = None
    self.colors = []
    self._gray_frames = []
    self._thresh_frames = []
    self._times = []
    self._true_coord = []

    vid = cv2.VideoCapture(videofile)

    while True:
      _, img = vid.read()
      if img is None:
        print('Completed')
        break

      ##Убогий колхоз, нужно переобучать модель на нормальных данных
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      retval, img_thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
      img_thresh = cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR)
      ##

      self.frames.append(img[:,:,::-1])
      self._thresh_frames.append(img_thresh)
      self._gray_frames.append(gray)
      self._times.append(vid.get(cv2.CAP_PROP_POS_MSEC))

    vid.release()
    cv2.destroyAllWindows()

  def detect(self, model_type='yolov5l', limit_classes=True):
    if model_type not in ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']:
      raise Exception("One of the models must be chosen: 'yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'")
    model = torch.hub.load('ultralytics/yolov5', model_type)

    if limit_classes:
      model.classes = [0]

    self.objects = model(self.frames)

  def find_keypoints(self, config_file, weights_file):
    device='cpu'
    if torch.cuda.is_available():
      device = 'cuda'

    #set up
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights_file
    cfg.MODEL.DEVICE = device
    predictor = DefaultPredictor(cfg)
    #inference
    outputs = [predictor(im) for im in self._thresh_frames]
    keypoints = [x['instances'].pred_keypoints.cpu().numpy().reshape(5,3) if x['instances'].pred_keypoints.cpu().numpy().size > 0 else np.zeros((5,3)) for x in outputs]
    self.keypoints = np.array([[x[:,0].tolist(), x[:,1].tolist(), np.ones(5).tolist()] for x in keypoints]).transpose(0,2,1)

  @staticmethod
  def _get_homography(x,y):
    A = []
    for pt, tpt in zip(x,y):
      A.append([-pt[0], -pt[1], -1, 0, 0, 0, pt[0]*tpt[0], pt[1]*tpt[0], tpt[0]])
      A.append([0, 0, 0, -pt[0], -pt[1], -1, pt[0]*tpt[1], pt[1]*tpt[1], tpt[1]])
    A = np.array(A)

    _,_,v = np.linalg.svd(A)
    h = v[-1]
    H = h.reshape(3,3)
    return H

  def _get_projection(self,x,y,new):
    H = self._get_homography(x,y)
    scale = (H@new.T)[-1].T
    return (new@H.T/scale[:,None])

  def project(self, ground_truth: dict):
    #dict with true coordinates should preserve the order of points,
    #in which they are predicted
    self._true_coord = np.array(list(ground_truth.values()))
    self.homography = [self._get_homography(x, self._true_coord) for x in self.keypoints]

    #meusure unsimilarity beetween homographies
    benchmark = np.median(np.array(self.homography), axis=0)
    h_dist = [norm(H-benchmark) for H in self.homography]

    #eleminate observations with deviations in keypoints detections
    bad_homography_filter = [True if x<0.5 else False for x in h_dist]

    #eleminate observations with no keypoints detected
    no_points_filter = [True if x.sum() != 5 else False for x in self.keypoints]

    #common filter
    self.filter = np.multiply(bad_homography_filter,no_points_filter).tolist()

  def __color_extract(self, func=np.median):
    yolo = list(compress(self.objects.xywh, self.filter))
    for box, frame in zip(yolo, self._gray_frames):
      for obj in box.cpu().numpy():
        x, y, w, h = obj[0], obj[1], obj[2], obj[3]
        x1,y1,x2,y2 = int(x-w/4), int(y-h/3), int(x+w/4), int(y)
        self.colors.append(func(frame[y1:y2, x1:x2]))

#Better change to exclude keypoints (use projection matricies instead) ------ ------- ------ !!!!!!
  def transform(self):
    n_frames = len(self.frames)
    list_dfs = []
    for moment, frame_out, keypoints_out, idx in zip(self._times, self.objects.xyxy, self.keypoints, np.arange(n_frames)):
      frame_out = frame_out.cpu()
      n_objects = frame_out.size()[0]
      #detected objects coordinates in 3dim frame system
      pred_3d = np.vstack((frame_out.numpy()[:, [0,2]].mean(axis=1), 
                         frame_out.numpy()[:, 1], 
                         np.full(n_objects, 1))).T
      #detected objects coordinates in 3dim frame system
      transformed = self._get_projection(keypoints_out, self._true_coord, pred_3d)

      #logging result
      out = pd.DataFrame()
      out['frame'] = [idx for x in range(n_objects)]
      out['object'] = [x for x in range(n_objects)]
      out['x'] = transformed[:,0]
      out['y'] = transformed[:,1]
      out['time'] = [moment for x in range(n_objects)]

      list_dfs.append(out)

    list_dfs = list(compress(list_dfs, self.filter))
    self.output = pd.concat(list_dfs, ignore_index=True)


  def funnel(self, config_file, weights_file, gt):
    #stage1 - detect objects
    self.detect()
    #stage2 - detect keypoints
    self.find_keypoints(config_file=config_file, weights_file=weights_file)
    #stage3 - homography and filters
    self.project(ground_truth=gt)
    #stage3 - transormation
    self.transform()
    #stage4 - color
    self.__color_extract()
    self.output['color'] = self.colors