#(1) example01
'''
This is example about tensor generation.
'''
import torch
import numpy as np

## tensor generation
x1 = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
x2 = torch.rand(3,3)
x3 = torch.FloatTensor([[1,2,3],[4,5,6],[7,8,9]])
x4 = torch.LongTensor([[1,2,3],[4,5,6],[7,8,9]])
x5 = torch.ByteTensor([True,True,False])
print(x1)
print(x2)
print(x3)
print(x4)
print(x5)

## trun something into tensor
# turn list into tensor
list_example = [[1,2,3],[4,5,6]]
list_to_tensor = torch.Tensor(list_example)
print(list_to_tensor)

# turn numpy array into tensor
numpy_example = np.array([[1,2,3],[4,5,6]])
numpy_to_tensor = torch.Tensor(numpy_example)
print(numpy_to_tensor)

## trun tensor into something
# turn tensor into list
tensor_to_list = list_to_tensor.tolist()
print(tensor_to_list)

# turn tensor into numpy array
tensor_to_numpy = numpy_to_tensor.numpy()
print(tensor_to_numpy)

## tensor generation using GPU
tensor_example = torch.rand(3,3).cuda()
print(tensor_example)
