import torch
from sparta.common import TeSA
from sparta.propagation.talgebra import propagate_matmul

# test the propagation of matmul
tesa_in1 = TeSA(torch.tensor([[1,1],[0,0]]))
tesa_in2 = TeSA(torch.tensor([[1,1],[1,1]]))
tesa_out = TeSA(torch.tensor([[1,1],[1,1]]))

propagate_matmul(tesa_in1, tesa_in2, tesa_out)

print(tesa_in1.tesa)
print(tesa_in2.tesa)
print(tesa_out.tesa)

print('='*20)

# test the propagation of matmul
tesa_in1 = TeSA(torch.tensor([[1,0],[0,1]]))
tesa_in2 = TeSA(torch.tensor([[1,1],[1,1]]))
tesa_out = TeSA(torch.tensor([[1,1],[1,1]]))

propagate_matmul(tesa_in1, tesa_in2, tesa_out)

print(tesa_in1.tesa)
print(tesa_in2.tesa)
print(tesa_out.tesa)

print('='*20)

# test the propagation of matmul
tesa_in1 = TeSA(torch.tensor([[0,1],[0,1]]))
tesa_in2 = TeSA(torch.tensor([[1,1],[1,1]]))
tesa_out = TeSA(torch.tensor([[1,1],[1,1]]))

propagate_matmul(tesa_in1, tesa_in2, tesa_out)

print(tesa_in1.tesa)
print(tesa_in2.tesa)
print(tesa_out.tesa)

print('='*20)

# test the propagation of matmul
tesa_in1 = TeSA(torch.tensor([[0,1],[0,1]]))
tesa_in2 = TeSA(torch.tensor([[1,1],[1,1]]))
tesa_out = TeSA(torch.tensor([[0,0],[1,1]]))

propagate_matmul(tesa_in1, tesa_in2, tesa_out)

print(tesa_in1.tesa)
print(tesa_in2.tesa)
print(tesa_out.tesa)