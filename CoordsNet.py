import torch.nn as nn

class CoordsNet(nn.Module):

  def __init__(self, input_size=7680, hidden_size=500, reg_size=10):
    super(CoordsNet, self).__init__()
    self.silu0 = nn.SiLU()
    self.conv = nn.Conv2d(255, 2, 1, 1)
    self.silu1 = nn.SiLU()
    self.l1 = nn.Linear(input_size, hidden_size)
    self.silu2 = nn.SiLU()
    self.l2 = nn.Linear(hidden_size, reg_size)

  def forward(self, x, input_size=7680):
    out = self.silu0(x)
    out = self.conv(out)
    out = self.silu1(out)
    out = out.view(-1, input_size)
    out = self.l1(out)
    out = self.silu2(out)
    out = self.l2(out)
    return out