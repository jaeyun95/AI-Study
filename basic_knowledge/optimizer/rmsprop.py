import numpy as np

class RMSProp():
  def __init__(self,lr=0.001,p=0.5):
    self.lr = lr
    self.g = None
    self.e = 1e-7
    self.p = p

  def update(self, param, grad):
    if self.g is None:
      self.g = np.zeros_like(param)
    self.g = self.p*self.g + (1-self.p)*grad*grad
    return param - self.lr*grad/(np.sqrt(self.g+self.e))