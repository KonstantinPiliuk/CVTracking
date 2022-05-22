class Extractor(list):

  def extract(self, argument='pos'):
    '''
    Works with Tracker stage output to convert its results into pandas.DataFrame
    ----------
    parameters:
    -- pos: one of the characteristics of the objects. 
    Might take values: 'pos', 'color', 'time', 'acc', 'vel', 't_delta', 'cov', 'exp_pos'
    ----------
    output:
    -- list of dictionaries with objects characteristics for all frames [{object: value}]
    '''
    if argument not in ['pos', 'color', 'time', 'acc', 'vel', 't_delta', 'cov', 'exp_pos']:
      raise Exception("One of the arguments must be chosen: 'pos', 'color', 'time', 'acc', 'vel', 't_delta', 'cov', 'exp_pos")
    
    out = []
    for state in self:
      new_state = {}
      for k,v in state.items():
        new_state[k] = v[argument]
      out.append(new_state)
    return out