#(2) example02
'''
This is example about treating tensor.
'''
import torch

# tensor generation for treating tensor
x1 = torch.rand(3,3)
x2 = torch.rand(3,3)
print(x1)
print(x2)

## calculation of tensor
# plus of tensor
x3 = x1 + x2
print(x3)
# minus of tensor
x4 = x1 - x2
print(x4)
# multiplation of tensor
x5 = torch.mm(x1,x2)
print(x5)
# division of tensor
x6 = torch.div(x1,x2)
print(x6)

## dimension tranform of tensor
# using unsqueeze
x7 = torch.unsqueeze(x1,0)
print('before : ',x1.shape,', after : ',x7.shape)
# using squeeze
x8 = torch.squeeze(x7)
print('before : ',x7.shape,', after : ',x8.shape)
# using view
x9 = x1.view(9)
print('before : ',x1.shape,', after : ',x9.shape)
# using transpose
tensor_example = torch.rand(4,5)
x10 = tensor_example.transpose(0,1)
print('before : ',tensor_example.shape,', after : ',x10.shape)

## minimum, maximum of tensor
# minimum of tensor
x11 = torch.min(x1)
min_location = torch.argmin(x1)
print('minimum : ',x11,', location : ',min_location)
# maximum of tensor
x12 = torch.max(x1)
max_location = torch.argmax(x1)
print('maximum : ',x12,', location : ',max_location)

## cat, stack of tensor
# using cat
x13 = torch.cat([x1,x2],dim = 0)
print(x13.shape)
# using stack
x14 = torch.stack([x1,x2],dim = 0)
print(x14.shape)

## sum, mean of tensor
# using sum
x15 = torch.sum(x1)
print(x15)
# using mean
x16 = torch.mean(x1)
print(x16)
