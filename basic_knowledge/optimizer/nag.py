import numpy as np

class NesterovAcceleratedGradient():
  def __init__(self,lr=0.001,momentum=0.9):
    self.lr = lr
    self.momentum = momentum
    self.v = None

  def update(self, param, grad):
    if self.v is None:
      self.v = np.zeros_like(param)
    self.v = self.momentum*self.v-self.lr*grad
    return param + self.momentum*self.v - self.lr*grad