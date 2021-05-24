import math

class NaiveBatesClassifier():
  def __init__(self,k=0.001):
    self.k = k
    self.yes_prob = {}
    self.no_prob = {}

  def fit(self,params_y, params_n):
    for key, val in params_y.items():
      self.yes_prob[key] = val + self.k
    for key, val in params_n.items():
      self.no_prob[key] = val + self.k
  
  def predict(self,codition):
    yes = 0
    no = 0
    for c in condition:
      yes += math.log(self.yes_prob[c])
      no += math.log(self.no_prob[c])
      
    return "yes" if yes > no else "no"


naive = NaiveBatesClassifier()

data_y = {"sunny":2/9,"cool":3/9,"high":3/9,"true":3/9}
data_n = {"sunny":3/5,"cool":1/5,"high":4/5,"true":3/5}
condition = ["sunny","cool","high","true"]

naive.fit(data_y,data_n)
print(naive.predict(condition))