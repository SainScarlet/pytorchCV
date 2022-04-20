import numpy as np
import torch
import matplotlib.pyplot as plt

# # 自动梯度
# # 随机3个张量，需要梯度
# x = torch.randn(1, 5, requires_grad=True)
# y = torch.randn(5, 3, requires_grad=True)
# z = torch.randn(3, 1, requires_grad=True)
# print("x:\n", x, "\ny:\n", y, "\nz:\n", z)
#
# # 矩阵相乘
# xy = torch.matmul(x, y)
# print("xy:\n", xy)
# # 三个相乘
# xyz = torch.matmul(xy, z)
# # BP的写法
# xyz.backward()
# # 注意下结果，z的grad就是xy的转置，其他两个同理
# print("\nxyz grad:\n", x.grad, "\n", y.grad, "\n", z.grad)
# # 测试x梯度
# zy = torch.matmul(y, z).view(-1, 5)
# print(zy)

"""
线性回归部分
"""
# # 线性回归
# x = np.array([1, 2, 0.5, 2.5, 2.6, 3.1], dtype=np.float32).reshape((-1, 1))
# y = np.array([3.7, 4.6, 1.65, 5.68, 5.98, 6.95], dtype=np.float32).reshape(-1, 1)
#
#
# # torch的模型结构，class定义一个类
# # 所有模型继承torch.nn.Module
# # torch.nn.Linear为线性回归模型
# class LinearRegressionModel(torch.nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LinearRegressionModel, self).__init__()
#         self.linear = torch.nn.Linear(input_dim, output_dim)
#
#     # 定义计算，即前向传播
#     def forward(self, x):
#         out = self.linear(x)
#         return out
#
#
# # 模型定义
# input_dim = 1
# output_dim = 1
# model = LinearRegressionModel(input_dim, output_dim)
# # 损失定义
# criterion = torch.nn.MSELoss()
# # 优化器定义
# # 学习率定义
# learning_rate = 0.01
# # 随机梯度下降
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#
# # 官网示例
# for epoch in range(100):
#     epoch += 1
#     # Convert numpy array to torch Variable
#     # np转换为torch的tensor，x需要梯度，真实值又称为标签
#     inputs = torch.from_numpy(x).requires_grad_()
#     labels = torch.from_numpy(y)
#
#     # Clear gradients w.r.t. parameters 运行前清空优化器
#     optimizer.zero_grad()
#
#     # Forward to get output 前向传播
#     outputs = model(inputs)
#
#     # Calculate Loss 计算损失，label标签就是真实值
#     loss = criterion(outputs, labels)
#
#     # Getting gradients w.r.t. parameters BP算法
#     loss.backward()
#
#     # Updating parameters
#     # 优化权重，w-lr*dx，同时还有b，具体看基石那一章
#     optimizer.step()
#
#     print('epoch {}, loss {}'.format(epoch, loss.item()))
#
#
# # 上述是一个模型的训练过程
# # 下面是对训练模型的应用
# # Purely inference
# predicted_y = model(torch.from_numpy(x).requires_grad_()).data.numpy()
# print("标签Y:\n", y)
# print("预测Y:\n", predicted_y)
#
# # Clear figure
# plt.clf()
#
# # Plot true data
# # plot[fmt]表示的线型，g绿色，o为点，下面'--'为虚线，具体看matplotlib那一章
# plt.plot(x, y, 'go', label='True data', alpha=0.5)
#
# # Plot predictions
# plt.plot(x, predicted_y, '--', label='Predictions', alpha=0.5)
#
# # Legend and plot 增加图例，loc指定其位置
# plt.legend(loc='best')
# plt.show()


"""
非线性回归部分
"""
# 非线性回归
# -5到5均分20个数据，注意头尾都算入20个数据中
x = np.linspace(-5, 5, 20, dtype=np.float32)
# sigmoid扭曲成曲线
_b = 1/(1 + np.exp(-x))
# np的随机正态分布，第一个是中心，第二个是标准差，即增加噪声
y = np.random.normal(_b, 0.005)

# 20x1， -1，1是拉成一列，1，-1是拉成一行
x = np.float32(x.reshape(-1, 1))
y = np.float32(y.reshape(-1, 1))


class LogicRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogicRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # 和线性回归区别点
        out = torch.sigmoid(self.linear(x))
        return out


# 模型参数
input_dim = 1
output_dim = 1
model = LogicRegressionModel(input_dim, output_dim)
# 损失函数使用的bi-cross即二值交叉熵
criterion = torch.nn.BCELoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 训练使用同线性回归
for epoch in range(1000):
    epoch += 1
    # Convert numpy array to torch Variable
    inputs = torch.from_numpy(x).requires_grad_()
    labels = torch.from_numpy(y)

    # Clear gradients w.r.t. parameters
    optimizer.zero_grad()

    # Forward to get output
    outputs = model(inputs)

    # Calculate Loss
    loss = criterion(outputs, labels)

    # Getting gradients w.r.t. parameters
    loss.backward()

    # Updating parameters
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))

# Purely inference
predicted_y = model(torch.from_numpy(x).requires_grad_()).data.numpy()
print("标签Y:", y)
print("预测Y:", predicted_y)

# Clear figure
plt.clf()

# Get predictions
predicted = model(torch.from_numpy(x).requires_grad_()).data.numpy()

# Plot true data
plt.plot(x, y, 'go', label='True data', alpha=0.5)

# Plot predictions
plt.plot(x, predicted_y, '--', label='Predictions', alpha=0.5)

# Legend and plot
plt.legend(loc='best')
plt.show()
