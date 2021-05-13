import numpy as np

class AdaDelta():
  def __init__(self,lr=0.001,gamma=0.9):
    self.lr = lr
    self.g = None
    self.s = None
    self.e = 1e-7
    self.gamma = gamma

  def update(self, param, grad):
    if self.g is None and self.s is None:
      self.g = np.zeros_like(param)
      self.s = np.zeros_like(param)
    d_t = np.multiply(np.sqrt(self.s+self.e)/np.sqrt(self.g+self.e),grad)
    self.g = self.gamma*self.g + (1-self.gamma)*np.square(grad)
    self.s = self.gamma*self.s + (1-self.gamma)*np.square(d_t)
    return param - d_t