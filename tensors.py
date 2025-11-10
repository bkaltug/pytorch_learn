# Tensors are data structures that are similar to arrays and matrices but can also run on a GPU.

import torch
import numpy as np

# # Initializing tensors

# # Directly from the data 
# data = [[1,2],[3,4]]
# x_data = torch.tensor(data)

# # From a numpy array
# np_array = np.array(data)
# x_np = torch.from_numpy(np_array)

# # From another tensor
# x_ones = torch.ones_like(x_data)
# print(f"Ones tensor: \n{x_ones}\n")

# x_rand = torch.rand_like(x_data, dtype=torch.float)
# print(f"Random tensor: \n {x_rand} \n")

# shape = (2,3,)

# rand_tensor = torch.rand(shape)
# ones_tensor = torch.ones(shape)
# zeros_tensor = torch.zeros(shape)

# print(f"Random tensor 2: \n {rand_tensor} \n")
# print(f"Ones tensor 2: \n {ones_tensor} \n")
# print(f"Zeros tensor: \n {zeros_tensor} \n")


# # Attributes of a Tensor

# tensor = torch.rand(3,4)

# print(f"Shape of the tensor: {tensor.shape}")
# print(f"Datatype of the tensor: {tensor.dtype}")
# print(f"Device of the tensor: {tensor.device}")   

# Operations on a Tensor

# Moving the tensor to GPU
device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
tensor2 = torch.ones(4,4)
tensor2 = tensor2.to(device)

# Classic numpy-like indexing and slicing
print(f"First Row: {tensor2[0]}")
print(f"First Column: {tensor2[:,0]}")
print(f"Last Column: {tensor2[...,-1]}")
tensor2[:,1] = 0
print(tensor2)

# Joining tensors

# To concatenate
joint_tensors = torch.cat([tensor2,tensor2,tensor2],dim=1)
print(joint_tensors)

# Arithmetic operations

# Matrix multiplication between two tensors, t1,t2,t3

# tensor.T returns the transpose of a tensor
t1 = tensor2 @ tensor2.T
t2 = tensor2.matmul(tensor2.T)
t3 = torch.rand_like(t1)

torch.matmul(tensor2,tensor2.T,out=t3)

# Element-wise multiplication

z1 = tensor2 * tensor2
z2 = tensor2.mul(tensor2)
z3 = torch.rand_like(tensor2)

torch.mul(tensor2,tensor2,out=z3)

# Single element tensors

agg = tensor2.sum()
agg_item = agg.item()
print(agg_item, type (agg_item))

# In - place operations

tensor2.add_(5)
print(tensor2)

# Tensor to numpy array

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# Numpy array to tensor

n2 = np.ones(5)
t2 = torch.from_numpy(n2)

np.add(n2, 1, out=n2)
print(f"t: {t2}")
print(f"n: {n2}")
