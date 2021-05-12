import numpy as np

class AdaGrad():
  def __init__(self,lr=0.001):
    self.lr = lr
    self.g = None
    self.e = 1e-7

  def update(self, param, grad):
    if self.g is None:
      self.g = np.zeros_like(param)
    self.g = self.g + grad*grad
    return param - self.lr*grad/(np.sqrt(self.g+self.e))