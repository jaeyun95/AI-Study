
class ReLU():
  def forward(self, x):
    return max(x,0)