import torch as t
from torch.utils.data import DataLoader
import torchvision as tv
# 可视化组件其实来自TensorFlow中的tensorboard
# 使用方法：
# 训练前需定义write地址，在数据集加载后可以加载第一个batch_size大小的图片，并写入tensorboard中作为预览
# 训练中要把模型写入tensorboard，loss需要按照标量方式写入
# 所有数据写入完成后注意关掉writer
# 读取数据在终端中输入tensorboard --logdir=PATH
# 浏览器中输入 http://localhost:6006/ 查看
from torch.utils.tensorboard import SummaryWriter

transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                   tv.transforms.Normalize((0.5,), (0.5,)),
                                   ])

# 模型加载
train_ts = tv.datasets.MNIST(root='/Users/remilia/Documents/02-Work/05-Python/0-source/data',
                             train=True, download=True, transform=transform)
test_ts = tv.datasets.MNIST(root='/Users/remilia/Documents/02-Work/05-Python/0-source/data',
                            train=False, download=True, transform=transform)
train_dl = DataLoader(train_ts, batch_size=32, shuffle=True, drop_last=False)
test_dl = DataLoader(test_ts, batch_size=64, shuffle=True, drop_last=False)

# summary数据写入地址，必须已存在
writer = SummaryWriter('/Users/remilia/Documents/02-Work/05-Python/02_pytorchCV/1_writer/CNN')


# get some random training images
# 循环读取训练集图片和标签
dataiter = iter(train_dl)
images, labels = dataiter.next()

# create grid of images
# 创新image的网络
img_grid = tv.utils.make_grid(images)

# 写入tensorboard中
writer.add_image('mnist_images', img_grid)


class CNN_Mnist(t.nn.Module):
    def __init__(self):
        super(CNN_Mnist, self).__init__()
        self.cnn_layers = t.nn.Sequential(
            t.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, stride=1),
            t.nn.MaxPool2d(kernel_size=2, stride=2),
            t.nn.ReLU(),
            t.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1, stride=1),
            t.nn.MaxPool2d(kernel_size=2, stride=2),
            t.nn.ReLU()
        )
        self.fc_layers = t.nn.Sequential(
            t.nn.Linear(7*7*32, 200),
            t.nn.ReLU(),
            t.nn.Linear(200, 100),
            t.nn.ReLU(),
            t.nn.Linear(100, 10),
            t.nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        out = self.cnn_layers(x)
        out = out.view(-1, 7*7*32)
        out = self.fc_layers(out)
        return out


def train_and_test():
    model = CNN_Mnist()
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    loss = t.nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=1e-3)

    # 写入模型 add_graph是因为模型开发是基于图的开发，不像传统的基于对象
    # 注意这个地方如果前面使用cuda加载，这里也需要cuda
    # writer.add_graph(model, images.cuda())
    writer.add_graph(model, images)

    for s in range(50):
        # 新增loss参数，方便后面写入，每次epoch循环loss重新计算
        m_loss = 0.0
        print("run in epoch : %d" % s)
        for i, (x_train, y_train) in enumerate(train_dl):
            # x_train = x_train.cuda()
            # y_train = y_train.cuda()
            y_pred = model.forward(x_train)
            train_loss = loss(y_pred, y_train)
            # 把当前epoch的loss写入
            m_loss += train_loss.item()
            if (i + 1) % 100 == 0:
                print(i + 1, train_loss.item())
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        # 写入loss标量
        writer.add_scalar('training loss',
                          # m_loss / 1000,
                          m_loss/60000,
                          s * len(train_dl) + i)

    t.save(model.state_dict(), './0_model/02_cnn_mnist_model.pt')
    model.eval()

    # 一定注意训练完关闭writer
    writer.close()

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


if __name__ == "__main__":
    train_and_test()
