import torch
import numpy as np


# 随机初始化一个2x2的矩阵，或者张量
# 若初始化为零，使用torch.zeros()
# 自定义数组，使用torch.tensor(),同np.array()
x = torch.randn(2, 2)
y = torch.zeros(3, 3)
print(x, y)

# 计算
# 同numpy，有广播机制
a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
print(a)
b = torch.tensor([11, 12, 13, 14, 15, 16, 17, 18, 19, 10])
c = a.add(b)
print(c)

# 维度变换,拉成1维度的
x = x.view(-1, 4)
y = y.view(-1, 9)
# 打印向量尺寸
print(x, y, x.size(), y.size())

# tensor转换为numpy的array类型
nx = x.numpy()
ny = y.numpy()
print("nx :\n", nx, "\nny:\n", ny)

# numpy的array转换成torch所用的tensor
x2 = torch.from_numpy(nx.reshape((2, 2)))
print(x2)

# using CUDA/GPU
# Apple m1暂未适配torch，无法调用GPU
# 后续代码测试会放到win平台，部署使用openVINO
if torch.cuda.is_available():
    print("GPU Detected!")
    gpu_num = torch.cuda.current_device()
    print("GPU Name: %s \n" % torch.cuda.get_device_name(gpu_num))
    device = torch.device("cuda:0")
    # GPU版本的加法运算
    result = a.cuda() + b.cuda()
    print(result)
