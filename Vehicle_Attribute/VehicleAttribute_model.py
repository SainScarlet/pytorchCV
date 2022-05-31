import torch
import torch.nn.functional as tf
from VA_CustomDataset import VehicleAttrsDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# # 检查是否可以利用GPU
# train_on_gpu = torch.cuda.is_available()
# if not train_on_gpu:
#     print('CUDA is not available.')
# else:
#     print('CUDA is available!')


# 定义残差网络块
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(ResidualBlock, self).__init__()

        self.skip = torch.nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.skip = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels))

        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.block(x)
        identity = self.skip(x)
        out += identity
        out = tf.relu(out)
        return out


class VehicleAttributesResNet(torch.nn.Module):
    def __init__(self):
        super(VehicleAttributesResNet, self).__init__()
        self.cnn_layers = torch.nn.Sequential(
            # 卷积层 (64x64x3的图像)
            # 单卷积（in out kernel）
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            # 这一步32x32x32
            torch.nn.MaxPool2d(2, 2),

            ResidualBlock(32, 64),
            torch.nn.MaxPool2d(2, 2),

            # 32x32x32
            ResidualBlock(64, 128),
            torch.nn.MaxPool2d(2, 2),
        )
        # 全局最大池化
        self.global_max_pooling = torch.nn.AdaptiveMaxPool2d((1, 1))
        # linear layer (N*9*9*128 ->N*128 )

        self.color_fc_layers = torch.nn.Sequential(
            torch.nn.Linear(128, 7),
            torch.nn.Sigmoid()
        )

        self.type_fc_layers = torch.nn.Sequential(
            torch.nn.Linear(128, 4),
        )

    def forward(self, x):
        # stack convolution layers
        x = self.cnn_layers(x)

        # 8x8x128
        b, c, h, w = x.size()
        out = self.global_max_pooling(x).view(b, -1)

        # 全连接层
        out_color = self.color_fc_layers(out)
        out_type = self.type_fc_layers(out)
        return out_color, out_type


if __name__ == "__main__":
    # create a complete CNN
    model = VehicleAttributesResNet()
    print(model)

    # # 使用GPU
    # if train_on_gpu:
    #     model.cuda()

    ds = VehicleAttrsDataset(
        "/Users/remilia/Documents/02-Work/05-Python/0-CustomDb/vehicle_attrs_dataset"
    )
    # 数据集的读取与batch设置
    num_train_samples = ds.num_of_samples()
    bs = 16
    dataloader = DataLoader(ds, batch_size=bs, shuffle=True)

    # 训练模型的次数
    num_epochs = 10
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    # 损失函数（交叉熵）
    cross_loss = torch.nn.CrossEntropyLoss()
    index = 0
    for epoch in range(num_epochs):
        train_loss = 0.0
        for i_batch, sample_batched in enumerate(dataloader):
            images_batch, color_batch, type_batch = \
                sample_batched['image'], sample_batched['color'], sample_batched['type']

            # cuda加速
            # if train_on_gpu:
            #     images_batch, color_batch, type_batch = images_batch.cuda(), color_batch.cuda(), type_batch.cuda()

            optimizer.zero_grad()

            # forward pass: compute predicted outputs by passing inputs to the model
            m_color_out_, m_type_out_ = model(images_batch)
            color_batch = color_batch.long()
            type_batch = type_batch.long()

            # calculate the batch loss
            # 两个交叉熵误差求和
            loss = cross_loss(m_color_out_, color_batch) + cross_loss(m_type_out_, type_batch)

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()

            # update training loss
            train_loss += loss.item()
            if index % 100 == 0:
                print('step: {} \tTraining Loss: {:.6f} '.format(index, loss.item()))
            index += 1

        # 计算平均损失
        train_loss = train_loss / num_train_samples

        # 显示训练集与验证集的损失函数
        print('Epoch: {} \tTraining Loss: {:.6f} '.format(epoch, train_loss))

    # save model
    model.eval()
    # torch.save(model, './model/vehicle_attributes.pt')
    torch.save(model.state_dict(), './model/vehicle_attributes.pt')
