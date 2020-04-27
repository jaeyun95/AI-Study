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
x5 = x1 - x2
print(x5)
# multiplation of tensor
x6 = torch.mm(x1,x2)
print(x6)
# division of tensor
x7 = torch.div(x1,x2)
print(x7)

## dimension tranform of tensor
# using unsqueeze
x8 = torch.unsqueeze(x1,0)
print('before : ',x1.shape,', after : ',x8.shape)
# using squeeze
x9 = torch.squeeze(x7)
print('before : ',x7.shape,', after : ',x9.shape)
# using view
x10 = x1.view(9)
print('before : ',x1.shape,', after : ',x10.shape)
# using transpose
tensor_example = torch.rand(4,5)
x11 = tensor_example.transpose(0,1)
print('before : ',tensor_example.shape,', after : ',x11.shape)

## minimum, maximum of tensor
# minimum of tensor
x12 = torch.min(x1)
min_location = torch.argmin(x1)
print('minimum : ',x12,', location : ',min_location)
# maximum of tensor
x13 = torch.max(x1)
max_location = torch.argmax(x1)
print('maximum : ',x13,', location : ',max_location)

## cat, stack of tensor
# using cat
x14 = torch.cat([x1,x2],dim = 0)
print(x14.shape)
# using stack
x15 = torch.stack([x1,x2],dim = 0)
print(x15.shape)

## sum, mean of tensor
# using sum
x16 = torch.sum(x1)
print(x16)
# using mean
x17 = torch.mean(x1)
print(x17)
