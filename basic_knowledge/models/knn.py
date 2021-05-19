import numpy as np
import collections

class KNearestNeibhbors():
  def __init__(self,k=5):
    self.k = k
    self.datas = []
    self.labels = []

  def fit(self,datas,labels):
    self.datas = datas
    self.labels = labels

  def predict(self,test):
    dis = []
    for i,data in enumerate(self.datas):
      dis.append([i,self.distance(data,test)])
    top_k = sorted(dis,key=lambda x:x[1])[:self.k]
    labels = [self.labels[id] for id, dis in top_k]
    counts = collections.Counter(labels)

    return counts.most_common(1)[0][0]

  def distance(self,p1,p2):
    return np.sqrt(np.sum(np.power(p2-p1,2)))