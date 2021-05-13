import numpy as np

class Adam():
  def __init__(self,lr=0.001,beta_1=0.9,beta_2=0.999):
    self.lr = lr
    self.m = None
    self.v = None
    self.e = 1e-7
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.beta_1_n = beta_1
    self.beta_2_n = beta_2

  def update(self, param, grad):
    if self.m is None and self.v is None:
      self.m = np.zeros_like(param)
      self.v = np.zeros_like(param)
    self.m = self.beta_1*self.m + (1 - self.beta_1) * grad
    self.v = self.beta_2*self.v + (1 - self.beta_2) * grad * grad
    m_hat = self.m / (1 - self.beta_1_n)
    v_hat = self.v / (1 - self.beta_2_n)
    self.beta_1_n *= self.beta_1
    self.beta_2_n *= self.beta_2
    return param - (self.lr / np.sqrt(self.v + self.e)) * self.m 