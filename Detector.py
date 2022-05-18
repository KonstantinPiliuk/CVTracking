import numpy as np
import pandas as pd
import cv2
import torch
from numpy.linalg import norm
from itertools import compress
from CoordsNet import CoordsNet
from Logger import insert_data

class Detector():

  def __init__(self, vid, weights, log=True, dialect='mysql', host=None, user=None, pwd=None, database=None):
    self.vid = vid
    self.weights = weights

    self.log = log
    self.dialect = dialect
    self.host = host
    self.user = user
    self.database = database
    self.pwd = pwd

    self.output = None
    self.frames = []
    self.homography = None
    self.objects = None
    self.keypoints = None
    self.gt_coords = {'center':[105/2,30,1], 'center-up':[105/2,30+9.15,1], 'center-down':[105/2,30-9.15,1],
    'center-left':[105/2-9.15,30,1], 'center-right':[105/2+9.15,30,1]}
    self.filter = None
    self.colors = []

    self._bad_homography_filter = None
    self._no_points_filter = None
    self._yolo_features = None
    self._times = []
    self._true_coord = []
    self._gray_frames = []

    #read video to frames
    video = cv2.VideoCapture(self.vid)
    while True:
      _, img = video.read()
      if img is None:
        #print('Video read successfully')
        break

      self.frames.append(img[:,:,::-1])
      self._gray_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
      self._times.append(video.get(cv2.CAP_PROP_POS_MSEC))

    video.release()
    cv2.destroyAllWindows()

    #add new game to game reference
    if self.log:
      game_df = pd.DataFrame([{'HOME_SIDE': 'Team 1', 'AWAY_SIDE': 'Team 2', 'GAME_DT': '2022-05-01'}])
      insert_data(game_df,'games_ref', dialect=self.dialect, host=self.host, user=self.user, pwd=self.pwd, database=self.database)

  def detect(self, model_type='yolov5l', limit_classes=True):
    if model_type not in ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']:
      raise Exception("One of the models must be chosen: 'yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'")
    model = torch.hub.load('ultralytics/yolov5', model_type)

    if limit_classes:
      model.classes = [0]

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model.model.model.model[24].m[0].register_forward_hook(get_activation('conv_80'))
    self.objects = model(self.frames)
    self._yolo_features = activation['conv_80'].cpu().numpy()

  def find_keypoints(self):
    if self._yolo_features is None:
      raise Exception("No YOLO features yet. Consider finding objects first")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CoordsNet().to(device)
    model.load_state_dict(torch.load(self.weights, map_location=torch.device(device)))
    model.eval()

    out = []
    for feature in self._yolo_features:
      inp = torch.from_numpy(feature).to(device)
      out.append(model(inp).cpu().detach().numpy())
    self.keypoints = np.array(out).reshape(len(self.frames), 5, 2)

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

  def project(self):
    #dict with true coordinates should preserve the order of points,
    #in which they are predicted
    self._true_coord = np.array(list(self.gt_coords.values()))
    self.homography = [self._get_homography(x, self._true_coord) for x in self.keypoints]

    #measure unsimilarity beetween homographies
    benchmark = np.median(np.array(self.homography), axis=0)
    h_dist = [norm(H-benchmark) for H in self.homography]

    #eleminate observations with deviations in keypoints detections
    self._bad_homography_filter = [True if x<0.5 else False for x in h_dist]

    #eleminate observations with no keypoints detected
    self._no_points_filter = [True if x.sum() != 5 else False for x in self.keypoints]

    #common filter
    self.filter = np.multiply(self._bad_homography_filter,self._no_points_filter).tolist()

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

  def _log(self):
    #objects
    objs = [df for df in self.objects.pandas().xyxy]
    obj = pd.concat(objs, keys=[str(a) for a in range(len(objs))], names=['FRAME_ID', 'DETECTION_ID']).reset_index()
    insert_data(obj, 'detected_objects', dialect=self.dialect, host=self.host, user=self.user, pwd=self.pwd, database=self.database)

    #keypoints
    kpts = [pd.DataFrame({'x_pos': x[:,0], 'y_pos': x[:,1]}) for x in self.keypoints]
    kpt = pd.concat(kpts, keys=[str(x) for x in range(len(kpts))], names=['FRAME_ID', 'KEYPOINT_ID']).reset_index()
    insert_data(kpt, 'detected_keypoints', dialect=self.dialect, host=self.host, user=self.user, pwd=self.pwd, database=self.database)

    #keypoints reference
    ref = pd.DataFrame([{'nm': k, 'x_pos': v[0], 'y_pos': v[1]} for k, v in self.gt_coords.items()]).reset_index()
    insert_data(ref, 'keypoints_ref', dialect=self.dialect, host=self.host, user=self.user, pwd=self.pwd, database=self.database)

    #filters
    flt = pd.DataFrame({
      'bad_homography': self._bad_homography_filter, 'no_keypoints': self._no_points_filter, 'common_filter': self.filter
      }).reset_index()
    insert_data(flt, 'homography_filters', dialect=self.dialect, host=self.host, user=self.user, pwd=self.pwd, database=self.database)

    #transformation
    insert_data(self.output, 'fct_transform', dialect=self.dialect, host=self.host, user=self.user, pwd=self.pwd, database=self.database)

  def funnel(self):
    #stage1 - detect objects
    self.detect()
    #stage2 - detect keypoints
    self.find_keypoints()
    #stage3 - homography and filters
    self.project()
    #stage3 - transormation
    self.transform()
    #stage4 - color
    self.__color_extract()
    self.output['color'] = self.colors
    #stage5 - logging
    if self.log:
      self._log()