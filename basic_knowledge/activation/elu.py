import numpy as np

class ELU():
  def __init__(self, alpha=0.01):
    self.alpha = alpha

  def forward(self, x):
    return self.alpha * (np.exp(x) + 1) if x <= 0 else x