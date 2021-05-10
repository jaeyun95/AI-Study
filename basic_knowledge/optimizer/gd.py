class GradientDescent():
  def __init__(self,lr=0.001):
    self.lr = lr

  def update(self, param, grad):
    return param - self.lr*grad

X = np.random.rand(10)
Y = np.random.rand(10)

W = np.random.uniform(-1,1)
b = np.random.uniform(-1,1)
gd = GradientDescent()