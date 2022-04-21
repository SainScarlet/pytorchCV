import torch as t
# dataloader用来加载数据集，shuffle洗牌很有用
from torch.utils.data import DataLoader
# pytorch的数据集都在torchvision.datasets中
# tv.datasets.xxx
# 常用的诸如CIFAR100, MNIST, COCO, VOC等
# 报PIL的错用__version__ 替换原来的PILLOW_VERSION，报错的最后一行的functional文件
import torchvision as tv
import cv2 as cv
import numpy as np

# 定义transform，后续数据集加载使用，使用torchvision的功能
# tv.transforms.ToTensor()自带归一化，图像加载会自动0-1
# tv.transforms.Normalize中第一个是-0.5即归一化后中心化变为(-0.5,0.5)，没有中心化的数据可能会有收敛问题
# 第二个0.5即标准差，相当于/0.5，最终transform处理结果为(-1,1)
# 特别注意的是如果彩色图像，需要标明三个通道的数据即(0.5, 0.5, 0.5)，灰度图就可以一个数
transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                   tv.transforms.Normalize((0.5,), (0.5,)), ])

# datasets中参数
# train = True加载训练集，False加载测试集
# download会下载数据集，下载路径在root中指定
# transform是必须的，需要把加载的数据集变形成torch所用的tensor
train_ts = tv.datasets.MNIST(root='/Users/remilia/Documents/02-Work/05-Python/0-source/data',
                             train=True, download=True, transform=transform)
test_ts = tv.datasets.MNIST(root='/Users/remilia/Documents/02-Work/05-Python/0-source/data',
                            train=False, download=True, transform=transform)
train_dl = DataLoader(train_ts, batch_size=32, shuffle=True, drop_last=False)
test_dl = DataLoader(test_ts, batch_size=64, shuffle=True, drop_last=False)

# # 显示测试，测试完成后可注释
# index = 0
# for i_batch, sample_batch in enumerate(train_dl):
#     print(i_batch, sample_batch[0].size(), sample_batch[1].size())
#     img = sample_batch[0][0].numpy().reshape((28, 28))
#     print(img.shape)
#     cv.imshow("mnist_test", img)
#     cv.waitKey(0)
#     if index == 4:
#         break
#     index += 1


# 定义模型
model = t.nn.Sequential(
    # 隐藏层，输入 神经元数量
    # 激活函数
    t.nn.Linear(784, 100),
    t.nn.ReLU(),
    # 全连接输出层
    # 分类，注意维度是1，并不是所想的10分类的10
    t.nn.Linear(100, 10),
    t.nn.LogSoftmax(dim=1)
)


# 定义训练及测试函数
def train_mnist():

    # 损失函数的定义
    # NLLLoss = Negative Log Likelihood Loss 负log似然损失，具体查看文档
    # 参数reduction="mean"求平均损失，"sum"可以求和损失
    # NLLLoss使用的是LogSoftmax()，需注意输出层
    loss_fn = t.nn.NLLLoss(reduction="mean")
    # 优化器定义，adam自适应动量梯度下降，注意科学计数法，e=10，1e-3即10^-3，0.1%
    optimizer = t.optim.Adam(model.parameters(), lr=1e-3)

    # 训练循环，即epoch
    for s in range(20):
        print("run in epoch : %d" % s)
        for i, (x_train, y_train) in enumerate(train_dl):
            x_train = x_train.view(x_train.shape[0], -1)
            y_pred = model(x_train)
            train_loss = loss_fn(y_pred, y_train)
            # 每100个打印一下loss
            if (i + 1) % 100 == 0:
                print(i + 1, train_loss.item())
            # 训练常规过程，梯度归零，反向传播，更新参数
            model.zero_grad()
            train_loss.backward()
            optimizer.step()

    total = 0
    correct_count = 0
    for test_images, test_labels in test_dl:
        for i in range(len(test_labels)):
            # 读取测试集的图，不需要梯度
            img = test_images[i].view(1, 784)
            with t.no_grad():
                # 测试集的图经过训练好的模型来给出预测
                pred_labels = model(img)
            plabels = t.exp(pred_labels)
            probs = list(plabels.numpy()[0])
            pred_label = probs.index(max(probs))
            # 读取测试集的真实标签
            true_label = test_labels.numpy()[i]
            # 预测标签和真实标签对比，累加计数器
            if pred_label == true_label:
                correct_count += 1
            # 每测试一张，总数+1
            total += 1
    # 打印准确率，2位有效数字
    print("total acc : %.2f\n" % (correct_count / total))
    # 保存模型字典形式，pt是pytorch的模型文件，注意路径文件夹必须已存在，否则报错
    t.save(model.state_dict(), './0_model/01_nn_mnist_model.pt')


def img_pre():
    img0 = cv.imread('/Users/remilia/Documents/02-Work/05-Python/0-source/mnist_test_320x328_8.jpg', -1)
    img = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
    # 高斯去噪
    img = cv.GaussianBlur(img, (5, 5), 0, 0)
    img = cv.GaussianBlur(img, (3, 3), 0, 0)
    # 加亮度去噪
    c = 0.8
    b = 90
    h, w = img.shape
    black_image = np.zeros([h, w], img.dtype)
    img = cv.addWeighted(img, c, black_image, 1 - c, b)
    # 阈值化，裁切，拉伸mnist数据形式
    ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
    roi = thresh[50:240, 90:250]
    roi = cv.bitwise_not(roi)
    img_result = cv.resize(roi, (28, 28))
    # 显示
    cv.imshow('Original', img0)
    cv.imshow('ImgResult', img_result)
    return img_result


if __name__ == "__main__":
    # train_mnist()  # 需要训练的时候取消注释
    # 打印保存模型的字典数据，这里打印size，权重矩阵也可以打印
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # 推理
    # 字典形式的模型文件加载方式，因为字典模型只保留参数，故需要一个已有的同结构的模型来加载这些参数
    model.load_state_dict(t.load("./0_model/01_nn_mnist_model.pt"))
    # model.eval()处理dropout和bn（批规范化）层，否则推理结果可能不正确
    model.eval()

    # 测试图片输入，测试的话需要把图片reshape成和测试集一样的shape
    # 图片的预处理，去噪阈值化等等
    image = img_pre()
    # 同一开始的transform，归一化，中心化，（-1，1），array(1x784)，tensor
    img_f = np.float32(image) / 255.0 - 0.5
    img_f = img_f / 0.5
    img_f = np.reshape(img_f, (1, 784))
    img_f = t.from_numpy(img_f)

    # 开始预测
    pred_labels = model(img_f)
    plabels = t.exp(pred_labels)
    probs = list(plabels.detach().numpy()[0])
    pred_label = probs.index(max(probs))
    print("predict digit number: ", pred_label)
    cv.waitKey(0)
    cv.destroyAllWindows()
