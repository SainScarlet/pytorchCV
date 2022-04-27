import torch as t
from torch.utils.data import DataLoader
import torchvision as tv
import cv2 as cv
import numpy as np

transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                   tv.transforms.Normalize((0.5,), (0.5,)), ])

# 模型加载
train_ts = tv.datasets.MNIST(root='/Users/remilia/Documents/02-Work/05-Python/0-source/data',
                             train=True, download=True, transform=transform)
test_ts = tv.datasets.MNIST(root='/Users/remilia/Documents/02-Work/05-Python/0-source/data',
                            train=False, download=True, transform=transform)
train_dl = DataLoader(train_ts, batch_size=32, shuffle=True, drop_last=False)
test_dl = DataLoader(test_ts, batch_size=64, shuffle=True, drop_last=False)


# 定义CNN的class，继承t.nn.Module
# 程序结构可以查看conv2d的文档
class CNN_Mnist(t.nn.Module):
    def __init__(self):
        # 父类函数的初始化
        super(CNN_Mnist, self).__init__()
        # 定义CNN部分模型结构，一层一行，注意在sequential中这样写层结尾的逗号勿忘记
        self.cnn_layers = t.nn.Sequential(
            # 第一个卷积层
            # 输入通道数：灰度图单通道
            # 输出通道数：其实是kernel种类，8个种类导致卷积输出是8@28*28（参考LeNet此处是6@28*28）
            # padding补充边界，和kernel尺寸相关，这里是为了输出尺寸不变
            # stride是卷积步长
            t.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, stride=1),
            # 最大池化层
            # 无其他参数，只定义kernel和步长，注意这里是局部池化，无重叠的kernel
            t.nn.MaxPool2d(kernel_size=2, stride=2),
            # 激活函数
            t.nn.ReLU(),
            # 第二层卷积
            # 这里LeNet使用的输出是16，是对上面的6张map做不规则组合得到
            t.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1, stride=1),
            # 第二个最大池化层
            t.nn.MaxPool2d(kernel_size=2, stride=2),
            # 激活函数
            t.nn.ReLU()
        )
        # 定义后面的全连接模型结构，同使用sequential定义
        self.fc_layers = t.nn.Sequential(
            # 注意，两次卷积都有padding，不同与LeNet，使得卷积尺寸不变
            # 第一次卷积得到14*14，第二次卷积得到7*7
            # 32是一次批处理32张，即batch_size
            # 200个神经元
            t.nn.Linear(7*7*32, 200),
            t.nn.ReLU(),
            # 第二个全连接层到100
            t.nn.Linear(200, 100),
            t.nn.ReLU(),
            # 第三个全连接层，到分类10
            t.nn.Linear(100, 10),
            # dim=1是指第二个维度的值，即预测值，dim=0是n
            t.nn.LogSoftmax(dim=1)
        )

    # 前向传播的计算，x是输入
    def forward(self, x):
        out = self.cnn_layers(x)
        # reshape成二维
        # 注意n、c、h、w四个参数，单通道所以c省略
        out = out.view(-1, 7*7*32)
        # 送到上面定义的全连接层
        out = self.fc_layers(out)
        return out


# # 构建完模型，需测试一下模型参数，再开始训练
# # 参数正常后，注释掉即可
# model = CNN_Mnist()
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())


def train_and_test():
    # 用gpu训练写法
    # model = CNN_Mnist().cuda()
    # 无gpu
    model = CNN_Mnist()
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    # 交叉熵损失，这次不用NLLLoss
    loss = t.nn.CrossEntropyLoss()
    # 优化器定义
    optimizer = t.optim.Adam(model.parameters(), lr=1e-3)

    # 实际测试下来，准确率稳定在98-99%之间，提升训练epoch效果并不明显
    for s in range(30):
        print("run in epoch : %d" % s)
        for i, (x_train, y_train) in enumerate(train_dl):
            # 注意这里可以有cuda的加速
            # 纯cpu无加速注释掉即可
            # x_train = x_train.cuda()
            # y_train = y_train.cuda()
            y_pred = model.forward(x_train)
            train_loss = loss(y_pred, y_train)
            # 老四样，每个epoch中，每100个打印loss，梯度归零，反向传播，更新参数
            if (i + 1) % 100 == 0:
                print(i + 1, train_loss.item())
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

    # 和ANN中不一样，这里是批量测试
    # 同时注意输入不像nn那样需要reshape
    total = 0
    correct_count = 0
    for test_images, test_labels in test_dl:
        # 注意这里也可以使用cuda加速
        # pred_labels = model(test_images.cuda())
        pred_labels = model(test_images)
        # 在pred_labels上的第二个维度找预测的最大值
        predicted = t.max(pred_labels, 1)[1]
        # 同理这里也是
        # correct_count += (predicted == test_labels.cuda()).sum()
        correct_count += (predicted == test_labels).sum()
        total += len(test_labels)
    print("total acc : %.4f\n" % (correct_count / total))

    # 训练好的模型保存
    t.save(model.state_dict(), './0_model/02_cnn_mnist_model.pt')
    model.eval()


if __name__ == "__main__":
    # train_and_test()  # 需要训练的时候取消注释
    model = CNN_Mnist()
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

