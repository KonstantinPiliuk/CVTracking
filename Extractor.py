class Extractor(list):

  def extract(self, argument='pos'):
    if argument not in ['pos', 'color', 'time', 'acc', 'vel', 't_delta', 'cov', 'exp_pos']:
      raise Exception("One of the arguments must be chosen: 'pos', 'color', 'time', 'acc', 'vel', 't_delta', 'cov', 'exp_pos")
    
    out = []
    for state in self:
      new_state = {}
      for k,v in state.items():
        new_state[k] = v[argument]
      out.append(new_state)
    return out