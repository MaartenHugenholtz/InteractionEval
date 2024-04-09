import torch
from torch.nn.functional import interpolate
A = torch.randn(6, 26, 151)
output = interpolate(A, size=150)
print(output.shape)