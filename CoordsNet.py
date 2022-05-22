import torch.nn as nn
import numpy as np

def padding_adjust(n, x):
  '''
  Computes parameters for padding and pooling layers for given feature map and required grid
  -----------------
  parameters:
  -- n: grid size for pooling
  -- x: feature map
  -----------------
  output:
  -- dict of parameters: {'padding_size' :(p_left, p_right, p_upper, p_lower), 
          'pooling_params': {'kernel_size': kernel, 'stride_size': stride}}
  '''
  if len(x.shape) == 3:
    c,h,w = x.shape
  elif len(x.shape) == 4:
    b,c,h,w = x.shape
  else:
    raise Exception('Please pass tensor with shape [batch_size,] channels, height, width')

  (p_left, p_right, p_upper, p_lower) = 0,0,0,0
  if np.floor(h/n) < n-1 and h%n > np.floor(h/n):
    if (n - h%n) % 2 == 0:
      p_upper = (n - h%n) // 2
      p_lower = p_upper
    else:
      p_upper = int(np.ceil((n - h%n)/2))
      p_lower = int(np.floor((n - h%n)/2))

  if np.floor(w/n) < n-1 and w%n > np.floor(w/n):
    if (n - w%n) % 2 == 0:
      p_left = (n - w%n) // 2
      p_right = p_left
    else:
      p_left = int(np.ceil((n - w%n)/2))
      p_right = int(np.floor((n - w%n)/2))

  kernel = (int(np.ceil((h+p_upper+p_lower)/n)), int(np.ceil((w+p_left+p_right)/n)))
  stride = (int(np.floor((h+p_upper+p_lower)/n)), int(np.floor((w+p_left+p_right)/n)))

  return {'padding_size' :(p_left, p_right, p_upper, p_lower), 
          'pooling_params': {'kernel_size': kernel, 'stride_size': stride}}


class Padder(nn.Module):
  def __init__(self, padding_size):
    '''
    Adds padding to feature map according to given padding size
    ------------
    parameters:
    -- padding_size: tuple of paddings (p_left, p_right, p_upper, p_lower)
    '''
    super(Padder, self).__init__()
    self.silu = nn.SiLU()
    self.pad = nn.ReplicationPad2d(padding_size)

  def forward(self, x):
    out = self.silu(x)
    out = self.pad(out)
    return out


class Pooler(nn.Module):
  def __init__(self, params, c_in=255, c_out=11):
    '''
    Runs MaxPooling for given grid
    -------
    Parameters:
    -- params: parameters dictionary after padding_adjust()
    -- c_in: input channels for convolution
    -- c_out: output channels for convolution
    '''
    super(Pooler, self).__init__()
    self.pool = nn.MaxPool2d(kernel_size=params['kernel_size'], stride=params['stride_size'])
    self.conv = nn.Conv2d(c_in, c_out, 1, 1)
    self.silu = nn.SiLU()

  def forward(self, x):
    out = self.pool(x)
    out = self.conv(out)
    out = self.silu(out)

    return out.view(out.shape[0], -1) if len(out.shape) == 4 else out.view(1, -1)


class CoordsNet(nn.Module):
  def __init__(self, params, map_size=13, c_in=255, c_out=11, hidden_size=100, class_size=5, reg_size=10):
    '''
    Keypoints detection DNN
    ------
    parameters:
    -- params: parameters dictionary after padding_adjust()
    -- map_size: square grid size
    -- c_in: input channels for convolution
    -- c_out: output channels for convolution
    -- hidden_size: hidden linear layer size
    -- class_size: number of keypoints
    -- reg_size: number of keypoints coordinates
    '''
    super(CoordsNet, self).__init__()
    self.pad = Padder(padding_size=params['padding_size'])
    self.pool = Pooler(params=params['pooling_params'], c_in=c_in, c_out=c_out)    
    self.l1 = nn.Linear(map_size*map_size*c_out, hidden_size)
    self.silu = nn.SiLU()
    self.classifier = nn.Linear(hidden_size, class_size)
    self.regressor = nn.Linear(hidden_size, reg_size)

  def forward(self, x):
    out = self.pad(x)
    out = self.pool(out)
    out = self.l1(out)
    out = self.silu(out)
    return self.classifier(out), self.regressor(out)