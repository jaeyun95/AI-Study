
class PReLU():
  def __init__(self,alpha):
    self.alpha = alpha

  def forward(self, x):
    return max(alpha * x, x)