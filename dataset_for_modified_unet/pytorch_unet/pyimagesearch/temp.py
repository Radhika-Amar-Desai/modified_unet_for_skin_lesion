import torch

t1 = torch.randn ( 1, 32, 56, 56 )
t2 = torch.randn ( 1, 16, 108, 108 )

t = torch.concatenate ( (t1,t2), dim=1 )

print ( t1.size(1) )