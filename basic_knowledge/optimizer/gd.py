import numpy as np

class GradientDescent():
  def __init__(self,lr=0.001):
    self.lr = lr

  def update(self, param, grad):
    return param - self.lr*grad
