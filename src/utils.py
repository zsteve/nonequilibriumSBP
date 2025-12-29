import torch 

cos_dist = torch.vmap(lambda u, v: 0.5*(1-torch.dot(u, v) / (u.norm() * v.norm())))

l2_dist = torch.vmap(lambda u, v: (u-v).norm()**2)

