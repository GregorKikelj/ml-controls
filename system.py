import torch
from torch import nn

def clamp(x, lo, hi):
    if x < lo:
        return lo, True
    elif x > hi:
        return hi, True
    return x, False

class x_controller:
    def __init__(self):
        self.x = 0
        self.x_dot = 0
        self.x_ddot = 0

        self.dt = 1/100
        self.rateLimit = 1/100

    def step(self, a):
        self.x_ddot, _ = clamp(a, self.x_ddot - self.rateLimit, self.x_ddot + self.rateLimit)
        self.x_dot += self.dt * self.x_ddot
        self.x += self.dt * self.x_dot
        
        self.x, _ = clamp(self.x, -1 , 1)


def interp(x, xp, fp):
  N = len(xp)

  def get_interp(xv):
    hi = 0
    while hi < N and xv > xp[hi]:
      hi += 1
    low = hi - 1
    return fp[-1] if hi == N and xv > xp[low] else (
      fp[0] if hi == 0 else
      (xv - xp[low]) * (fp[hi] - fp[low]) / (xp[hi] - xp[low]) + fp[low])

  return [get_interp(v) for v in x] if hasattr(x, '__iter__') else get_interp(x)
