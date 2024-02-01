import torch

len_keep = 2  # 保留两个
x = torch.rand((1,4,1))  # 对应 N,L,D
print("x", x)
noise = torch.rand(1, 4)
print("noise", noise)
ids_shuffle = torch.argsort(noise, dim=1)  # noise从小到大的序号
print("ids_shuffle", ids_shuffle)
ids_restore = torch.argsort(ids_shuffle, dim=1)  # noise从打乱ids_shuffle到恢复的序号
print("ids_restore", ids_restore)
N, L, D = x.shape

# keep the first subset
ids_keep = ids_shuffle[:, :len_keep]
print("ids_keep", ids_keep)
x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
print("x_masked", x_masked)
# generate the binary mask: 0 is keep, 1 is remove
mask = torch.ones([N, L], device=x.device)
mask[:, :len_keep] = 0
print("mask", mask)
# unshuffle to get the binary mask
mask = torch.gather(mask, dim=1, index=ids_restore)
print("mask", mask.shape)  # 在原始图像中的掩码位置