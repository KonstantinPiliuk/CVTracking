import numpy as np
import pandas as pd
import copy
from scipy.spatial.distance import cdist
from numpy.linalg import norm, pinv
from Extractor import Extractor
from Logger import insert_data

''' util functions for Tracker class '''
def check_dicts_keys(dicts_lst: list):
  a=0
  for dct in dicts_lst:
    if dct.keys() != dicts_lst[0].keys():
      a += 1
  if a == 0:
    pass
  else:
    raise Exception('All input dictionaries must have same keys')

class Tracker():

  def __init__(self, log=None, dialect='mysql', host=None, user=None, pwd=None, database=None):
    self.df           = None
    self.states_hist  = Extractor()
    self.matches_hist = []
    self.frames_hist  = []
    self.__pos_hist   = []

    self.log          = log
    self.dialect      = dialect
    self.host         = host
    self.user         = user
    self.database     = database
    self.pwd          = pwd

  @staticmethod
  def __check_df_input(df):
    '''
    Check whether dataframe contain all necessary columns
    '''
    if 'x' in df.columns and 'y' in df.columns and 'object' in df.columns and 'color' in df.columns and 'time' in df.columns:
      return df
    else:
      raise Exception("Input DataFrame must contain following columns: 'x', 'y', 'object', 'color', 'time'")

  def get_current_state(self, frame):
    '''
    for given frame extracts position, time and color for objects
    ------------
    parameters:
    -- frame: frame at input DataFrame for which extraction is performed
    ------------
    output:
    -- state: dict of dicts {object: {'pos': (x,y), 'color': 123, 'time': 123}}
    '''
    xy = {int(k):(v1,v2) for k,v1,v2 in self.df.loc[self.df['frame'] == frame, ['object', 'x', 'y']].values}
    time = {int(k):v for k,v in self.df.loc[self.df['frame'] == frame, ['object', 'time']].values}
    color = {int(k):v for k,v in self.df.loc[self.df['frame'] == frame, ['object', 'color']].values}
    #check if dicts have same keys
    check_dicts_keys([xy, color, time])
    #state dictionary
    state = {}
    for (k,v1), v2, v3 in zip(xy.items(), color.values(), time.values()):
      state[k] = {'pos': v1, 'color': v2, 'time': v3}
    return state

  def __init_history(self, frame):
    ''' 
    for the 1st frame sets last values = current values for position, time and color.
    Acceleration, velocity and time delta sets equal 0. Variance-Covariance matrix sets to default
    -------------------------------------------------
    parameters:
    -- frame: frame at input DataFrame
    ------------
    output:
    -- state: dict of dicts {object: {'pos': (x,y), 'color': 123, 'time': 123, 
    'acc': 123, 'vel': 123, 't_delta': 123, 'cov': numpy.array, 'exp_pos': (x,y)}}    
    '''
    state = self.get_current_state(frame)
    last_acc = {k: (0.0,0.0) for k in state.keys()}
    last_vel = {k: (0.0,0.0) for k in state.keys()}
    last_delta = {k: 0.0 for k in state.keys()}
    last_cov = {k: np.identity(3) for k in state.keys()}
    for k,acc,vel,delta,cov in zip(state.keys(), last_acc.values(), last_vel.values(), last_delta.values(), last_cov.values()):
      state[k]['acc'] = acc
      state[k]['vel'] = vel
      state[k]['t_delta'] = delta
      state[k]['cov'] = cov
      state[k]['exp_pos'] = state[k]['pos']
    return state

  def __first_frame(self, frame):
    '''
    Checks whether given frame is the 1st one
    -------------------------------------------------
    parameters:
    -- frame: frame at input DataFrame
    ------------
    output:
    -- True if a frame is the 1st one
    '''
    return frame == self.frames_hist[0]

  @staticmethod
  def __update_covariance(pos_hist, last_state):
    '''
    collects all observations of x,y and color for object
    and estimates its covariance matrix
    -----------------
    parameters:
    -- pos_hist: observed players coordinates and colors at each frame
    -- last_state: dictionary of all tracked objects with their last characteristics
    ------------------
    output:
    -- last_state dict of dicts {object: {'pos': (x,y), 'color': 123, 'time': 123, 
    'acc': 123, 'vel': 123, 't_delta': 123, 'cov': numpy.array, 'exp_pos': (x,y)}}
    '''
    for obj in last_state.keys():
      obs = []
      for state in pos_hist:
        #leave non-empty obersvations of object
        if (isinstance(state[obj]['pos'], tuple)) and (obj in state.keys()):  
          xy = state[obj]['pos']
          color = state[obj]['color']
          obs.append([xy[0], xy[1], color])  #3-dim pos_x, pos_y, color
    
      obs = np.array(obs).transpose(1,0)
      #if enough observations given use calculated covariance
      if obs.shape[1] > 2:
        last_state[obj]['cov'] = np.cov(obs)
      else:
        last_state[obj]['cov'] = np.identity(3)

    return last_state

  @staticmethod
  def __extrapolate_position(last_state, max_t_delta=1.2, max_vel=12, max_acc=9.8):
    '''
    Calculates expected positions on the next frame for detected objects
    -----------------------
    parameters:
    -- last_state: dictionary of all tracked objects with their last characteristics
    -- max_t_delta: maximal time gap since last observation
    -- max_vel: maximal velocity
    -- max_acc: maximal acceleration
    -----------------------
    output:
    -- last_state: dict of dicts {object: {'pos': (x,y), 'color': 123, 'time': 123, 
    'acc': 123, 'vel': 123, 't_delta': 123, 'cov': numpy.array, 'exp_pos': (x,y)}}
    '''
    #calculate expected position
    for k in last_state.keys():
      t_delta = last_state[k]['t_delta']
      xy = last_state[k]['pos']
      vel = last_state[k]['vel']
      acc = last_state[k]['acc']
      #if object hasn't been tracked for a while - don't calculate its expected position
      #otherwise it could be very far from its normal position (large t_delta)
      if t_delta < max_t_delta:
        # x1 = x0 + v1*t + a/2*t^2
        # extremal human velocity and acceleration limits are set (12 m/s, 9.8 m/s^2)
        next_x = xy[0] + min(max_vel, vel[0])*t_delta + (min(max_acc, acc[0])*t_delta**2)/2
        next_y = xy[1] + min(max_vel, vel[1])*t_delta + (min(max_acc, acc[1])*t_delta**2)/2
        last_state[k]['exp_pos'] =  (next_x, next_y)
      else:
        last_state[k]['exp_pos'] =  xy

    return last_state

  @staticmethod
  def __matching(last_state, cur_state, max_vel=12, min_t_delta=0.03, max_t_delta=1.2):
    '''
    Makes correspondences between already tracked objects and new ones
    --------------
    parameters:
    -- last_state: dictionary of all tracked objects with their last characteristics
    -- cur_state: dictionary of new objects characteristics
    -- max_t_delta: maximal time gap since last observation
    -- min_t_delta: minimal time gap since last observation
    -- max_vel: maximal velocity
    -- max_acc: maximal acceleration
    ---------------
    output:
    -- dictionary with corresponding objects {new_object: old_object}
    '''
    matches = {}  #dict {new_object: old_object}
    options = list(last_state.keys())  #all found objects
    #current_objects = list(last_state.keys())

    for new_id in cur_state.keys():
      dists = {}

      for exp_id in last_state.keys():
        new_pos = np.array(cur_state[new_id]['pos'])
        old_pos = np.array(last_state[exp_id]['pos'])
        #maximal possible distance from previous observation = max_velocity*time(from min to max)
        #after some threshold in time (max) distance doesn't increase
        max_dist = max_vel * min(max(last_state[exp_id]['t_delta'], min_t_delta), max_t_delta)

        #point is a candidate for new observation if they are closer, than max dist
        if exp_id in options and norm(new_pos - old_pos) <= max_dist:
          vi = pinv(last_state[exp_id]['cov'])
          obs_xy = np.append(np.array(cur_state[new_id]['pos']), cur_state[new_id]['color']).reshape(-1,3)
          exp_xy = np.append(np.array(last_state[exp_id]['exp_pos']), last_state[exp_id]['color']).reshape(-1,3)
          dists[exp_id] = cdist(obs_xy, exp_xy, metric='mahalanobis', VI=vi).item()

      #for a new point find an old one with the smallest mahalanobis distance in 3-dim (x,y,color)
      try:
        match = min(dists, key=dists.get)
      #if no match found, create a new object
      except ValueError:
        match = max(last_state.keys())+1
        options.append(match)
        last_state[match] = {'pos': cur_state[new_id]['pos'], 'color': cur_state[new_id]['color'], 
                            'time': cur_state[new_id]['time'], 'acc': (0.0,0.0),
                            'vel': (0.0, 0.0), 't_delta': 0, 'cov': np.identity(3),
                             'exp_pos': cur_state[new_id]['pos']}
      matches[new_id] = match
      options.remove(match)

    return matches
       
  @staticmethod
  def __state_update(last_state, cur_state, matches):
    '''
    Rewrites properties of tracked objects with new frame's objects after correspondence
    ----------
    parameters:
    -- last_state: dictionary of all tracked objects with their last characteristics
    -- cur_state: dictionary of new objects characteristics
    -- matches: dictionary of correspondence between new and existing objects
    ----------
    output: last_state: dict of dicts {object: {'pos': (x,y), 'color': 123, 'time': 123, 
    'acc': 123, 'vel': 123, 't_delta': 123, 'cov': numpy.array, 'exp_pos': (x,y)}}
    '''
    new_time = max([x['time'] for x in copy.deepcopy(cur_state).values()])

    for new_obj in matches.keys():
      last_obj = matches[new_obj]

      #old and new positions and velocity
      last_xy = last_state[last_obj]['pos']
      new_xy = cur_state[new_obj]['pos']
      last_vel = last_state[last_obj]['vel']
      new_vel = (0.0,0.0)

      #time delta between detections
      t_delta = (cur_state[new_obj]['time'] - last_state[last_obj]['time'])/1000
      #player velocity and acceleration
      if t_delta > 0:
        v = (new_xy[0] - last_xy[0]) / t_delta, (new_xy[1] - last_xy[1]) / t_delta
        a = (new_vel[0] - last_vel[0]) / t_delta, (new_vel[1] - last_vel[1]) / t_delta
      else:
        v = (0.0,0.0)
        a = (0.0,0.0)

      #update
      last_state[last_obj] = {'pos': cur_state[new_obj]['pos'], 'color': cur_state[new_obj]['color'], 
                           'time': cur_state[new_obj]['time'], 'acc': a, 'vel': v, 
                           't_delta': t_delta, 'cov': np.identity(3), 'exp_pos': last_state[last_obj]['exp_pos']}

    #for unmatched objects increase time delta
    for unmatched_object in last_state.keys():
      if unmatched_object not in matches.values():
        last_state[unmatched_object]['t_delta'] += (new_time - last_state[unmatched_object]['time'])/1000
        last_state[unmatched_object]['time'] = new_time  
         
    return last_state

  def __ref(self, last_state, matches):
    '''
    Updates self.__pos_hist, self.states_hist, and self.matches_hist attributes
    '''
    output_state = copy.deepcopy(last_state)
    for obj in output_state.keys():
      if obj not in matches.values():
        output_state[obj]['pos'] = np.nan

    self.__pos_hist.append(last_state)
    self.states_hist.append(output_state)
    self.matches_hist.append(matches)

  def _log(self):
    '''
    Insert data to sql tables
    '''
    #matches
    mtch = pd.DataFrame([{'FRAME_ID': id, 'DETECTION_NUM': k, 'OBJECT_ID': v} for x,id in zip(
      self.matches_hist, self.frames_hist) for k,v in x.items()])
    insert_data(mtch, 'matches_ref', dialect=self.dialect, host=self.host, user=self.user, pwd=self.pwd, database=self.database)

    #result
    #position
    pos = pd.DataFrame(self.states_hist.extract('pos'))
    cols = pos.columns
    pos.insert(loc=0, column='FRAME_ID', value = self.frames_hist)
    pos = pos.melt(id_vars='FRAME_ID', value_vars=cols, var_name='OBJECT_ID', value_name='POS')
    pos['POS_X']=np.nan
    pos['POS_Y']=np.nan
    pos['POS_X']=[k[0] if isinstance(k,tuple) else k for k in pos['POS']]
    pos['POS_Y']=[k[1] if isinstance(k,tuple) else k for k in pos['POS']]
    pos = pos[['OBJECT_ID', 'FRAME_ID', 'POS_X', 'POS_Y']]
    pos.head()
    #time
    sec = pd.DataFrame(self.states_hist.extract('time')) 
    sec.insert(loc=0, column='FRAME_ID', value = self.frames_hist)
    sec = sec.melt(id_vars='FRAME_ID', value_vars=cols, var_name='OBJECT_ID', value_name='MSECS')
    #color
    clr = pd.DataFrame(self.states_hist.extract('color')) 
    clr.insert(loc=0, column='FRAME_ID', value = self.frames_hist)
    clr = clr.melt(id_vars='FRAME_ID', value_vars=cols, var_name='OBJECT_ID', value_name='COLOR')
    #velocity
    vel = pd.DataFrame(self.states_hist.extract('vel')) 
    vel.insert(loc=0, column='FRAME_ID', value = self.frames_hist)
    vel = vel.melt(id_vars='FRAME_ID', value_vars=cols, var_name='OBJECT_ID', value_name='VEL')
    vel['VEL_X']=np.nan
    vel['VEL_Y']=np.nan
    vel['VEL_X']=[k[0] if isinstance(k,tuple) else k for k in vel['VEL']]
    vel['VEL_Y']=[k[1] if isinstance(k,tuple) else k for k in vel['VEL']]
    vel = vel[['OBJECT_ID', 'FRAME_ID', 'VEL_X', 'VEL_Y']]
    #acceleration
    acc = pd.DataFrame(self.states_hist.extract('acc')) 
    acc.insert(loc=0, column='FRAME_ID', value = self.frames_hist)
    acc = acc.melt(id_vars='FRAME_ID', value_vars=cols, var_name='OBJECT_ID', value_name='ACC')
    acc['ACC_X']=np.nan
    acc['ACC_Y']=np.nan
    acc['ACC_X']=[k[0] if isinstance(k,tuple) else k for k in acc['ACC']]
    acc['ACC_Y']=[k[1] if isinstance(k,tuple) else k for k in acc['ACC']]
    acc = acc[['OBJECT_ID', 'FRAME_ID', 'ACC_X', 'ACC_Y']]
    #expected_postions
    exp = pd.DataFrame(self.states_hist.extract('exp_pos')) 
    exp.insert(loc=0, column='FRAME_ID', value = self.frames_hist)
    exp = exp.melt(id_vars='FRAME_ID', value_vars=cols, var_name='OBJECT_ID', value_name='EXP_POS')
    exp['EXP_POS_X']=np.nan
    exp['EXP_POS_Y']=np.nan
    exp['EXP_POS_X']=[k[0] if isinstance(k,tuple) else k for k in exp['EXP_POS']]
    exp['EXP_POS_Y']=[k[1] if isinstance(k,tuple) else k for k in exp['EXP_POS']]
    exp = exp[['OBJECT_ID', 'FRAME_ID', 'EXP_POS_X', 'EXP_POS_Y']]
    #time delta
    dlt = pd.DataFrame(self.states_hist.extract('t_delta')) 
    dlt.insert(loc=0, column='FRAME_ID', value = self.frames_hist)
    dlt = dlt.melt(id_vars='FRAME_ID', value_vars=cols, var_name='OBJECT_ID', value_name='T_DELTA')
    #merge
    out = pos.merge(sec, how='left', on=['OBJECT_ID', 'FRAME_ID']).merge(
      clr, how='left', on=['OBJECT_ID', 'FRAME_ID']).merge(
          vel, how='left', on=['OBJECT_ID', 'FRAME_ID']).merge(
              acc, how='left', on=['OBJECT_ID', 'FRAME_ID']).merge(
                  exp, how='left', on=['OBJECT_ID', 'FRAME_ID']).merge(
                      dlt, how='left', on=['OBJECT_ID', 'FRAME_ID']
                  )
    insert_data(out, 'fct_track', dialect=self.dialect, host=self.host, user=self.user, pwd=self.pwd, database=self.database)

  def fit(self, df, max_acc=9.8, max_vel=12, min_t_delta=0.03, max_t_delta=1.2):
    '''
    Fits Tracker algorithm to input dataframe. Runs all step at once.
    ---------
    parameters:
    -- df: pandas.DataFrame with results of Detector stage
    -- max_acc: maximal acceleration
    -- max_vel: maximal velocity
    -- min_t_delta: minimal time delta since last detection
    -- max_t_delta: maximal time delta since last detection
    '''
    self.df = self.__check_df_input(df)
    self.frames_hist = list(df['frame'].sort_values().unique())

    for frame in self.frames_hist:
      #get new observation and write it to current state
      cur_state = self.get_current_state(frame)

      #initialize history values for first frame in the loop
      if self.__first_frame(frame):
        last_state = self.__init_history(frame)
      #after first frame update covariance matrix with accumulated history values
      else:
        last_state = self.__update_covariance(self.__pos_hist, last_state)

      #calculate expected position (previous position corrected to time, velocity and acceleration)
      last_state = self.__extrapolate_position(last_state, max_t_delta=max_t_delta, max_vel=max_vel, max_acc=max_acc)

      #match new detections with existing
      matches = self.__matching(last_state, cur_state, max_vel=max_vel, min_t_delta=min_t_delta, max_t_delta=max_t_delta)

      #update states of matched objects and t_delta for unmatched ones
      last_state = self.__state_update(last_state, cur_state, matches)

      #write state and match to logs
      self.__ref(last_state, matches)

    #log
    if self.log == 'sql':
      self._log()
    elif isinstance(self.log, str):
      pd.DataFrame(self.states_hist.extract('pos')).to_csv(self.log)