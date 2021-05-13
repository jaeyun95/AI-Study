
class LeakyReLU():
  def forward(self, x):
    return max(0.01*x,x)