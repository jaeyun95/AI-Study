import numpy as np

class Sigmoid():
  def forward(self, x):
    return 1 / (1 + np.exp(-x))