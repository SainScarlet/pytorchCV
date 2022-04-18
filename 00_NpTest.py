import numpy as np

a = np.array([1, 2, 3, 4, 5, 6])
b = np.array([8, 7, 6, 5, 4, 3])
print(a.shape, b.shape)
print(a)

# 改维度，等价于aa = a.reshape(3, 2)
# 或者a = np.array([1, 2, 3, 4, 5, 6]).reshape(3, 2)
aa = np.reshape(a, (3, 2))
# 这个意思可以理解为6个标签的4维度
bb = np.reshape(b, (1, 1, 1, 6))
print(aa, aa.shape)
print(bb, bb.shape)
# 重新转换b的维度
b1 = np.squeeze(bb)
print(b1, b1.shape)

# 寻找最大值
index = np.argmax(bb)
# 从打印结果来看，argmax返回的是最大值的位置
# bb是4维的数组，如果没有降维到b1，bb[index]输出会是[[[8，7，6，5，4，3]]],即降了一维度的bb
# 解决上面的问题，就在不使用的维度填[0]，即bb[0][0][0][index]
print("find max:", index, bb, bb[0][0][0][index])

# 交换维度，简单理解为转置
aaa = aa.transpose((1, 0))
print(aaa, aaa.shape)
# 恢复矩阵的初始，即不管多大维度，都会降到1维
a2 = np.reshape(aa, -1)
print(a2, a2.shape)

# 创建零矩阵
m1 = np.zeros((6, 6), dtype=np.uint8)
# 在指定的间隔内返回均匀间隔的数字，例如头=6，尾=10，返回100个，其实是6-10的99等分布
m2 = np.linspace(6, 10, 100)
print(m1, m2)
