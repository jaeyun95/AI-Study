import numpy as np

class Tanh():
  def forward(self, x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))