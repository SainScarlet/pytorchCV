import torch
import torch.nn.functional as tf
from CustomDataset import FaceLandmarksDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision as tv

# 检查是否可以利用GPU
# torch暂不支持调用M1的神经处理单元
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.')
else:
    print('CUDA is available!')


# 全局最大池化，但是还不是全局深度最大池化
class GlobalMaxPool2d(torch.nn.Module):
    def __init__(self):
        super(GlobalMaxPool2d, self).__init__()

    def forward(self, x):
        assert len(x.size()) == 4, x.size()
        b, c, h, w = x.size()
        return tf.max_pool2d(x, (w, h)).view(b, h*w)


# 全局深度池化的定义
class ChannelPool(torch.nn.MaxPool1d):
    def __init__(self, channels, isize):
        super(ChannelPool, self).__init__(channels)
        self.kernel_size = channels
        self.stride = isize

    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n, c, w*h).permute(0, 2, 1)
        pooled = torch.nn.functional.max_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices)
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c, w, h).view(n, w*h)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_layers = torch.nn.Sequential(
            # 卷积层 (64x64x3的图像)
            # 参数含义in_channel,out_channel,kernel_size,padding=1目的是kernel=3保持图像大小不变
            # 注意这里卷积后没有接池化，而是第二个卷积，然后是BN层，激活层，池化层
            # 这种结构称之为stacked-conv（深度卷积神经网络），来源于VGG
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.PReLU(),
            # 参数含义kernel_size,stride, 这一步输出32*32*32
            torch.nn.MaxPool2d(2, 2),

            # 32x32x32
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.PReLU(),
            torch.nn.MaxPool2d(2, 2),

            # 64x16x16
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.PReLU(),
            torch.nn.MaxPool2d(2, 2)
            # 输出为128*8*8
        )
        self.dw_max = ChannelPool(128, 8*8)
        # linear layer (16*16 -> 10)
        # 5个点，10个值
        self.fc = torch.nn.Linear(64, 10)

    def forward(self, x):
        # stack convolution layers
        x = self.cnn_layers(x)

        # 16x16x128
        # 深度最大池化层
        out = self.dw_max(x)
        # 全连接层
        out = self.fc(out)
        return out


# 自定义损失函数
def myloss_fn(pred_y, target_y):
    target_y = target_y.view(-1, 10)
    sum = torch.zeros(len(target_y)).cuda()
    # sum = torch.zeros(len(target_y))
    for i in range(0, len(target_y)):
        t_item = target_y[i]
        p_item = pred_y[i]
        # 求标签两眼的距离
        dx = t_item[0] - t_item[2]
        dy = t_item[1] - t_item[3]
        id = torch.sqrt(dx*dx + dy*dy)
        # N个点求
        for t in range(0, len(t_item), 2):
            dx = p_item[t] - t_item[t]
            dy = p_item[t+1] - t_item[t+1]
            dist = torch.sqrt(dx*dx + dy*dy)
            sum[i] += (dist / id)
        sum[i] = sum[i] / 5
    # return torch.sum(sum)
    return torch.sum(sum).cuda()


if __name__ == "__main__":
    # create a complete CNN
    model = Net()
    print(model)

    # 使用GPU
    if train_on_gpu:
        model.cuda()

    # ds = FaceLandmarksDataset(
    #     "/Users/remilia/Documents/02-Work/05-Python/0-CustomDb/landmark_dataset/"
    #     "landmark_output.txt"
    # )

    # ROG
    ds = FaceLandmarksDataset(
        "D:/2_Work/1_python/0-CustomDb/landmark_dataset/landmark_output_ROG.txt"
    )

    # 在dataset的类里面定义
    num_train_samples = ds.num_of_samples()
    dataloader = DataLoader(ds, batch_size=16, shuffle=True)

    # tensorboard 记录
    writer = SummaryWriter('./model/writer')

    # 报错，后面需要想想怎么修改
    # 循环读取训练集图片和标签
    # dataiter = iter(dataloader)
    # images, _ = dataiter.next()

    # create grid of images
    # 创新image的网络
    # img_grid = tv.utils.make_grid(images)

    # 写入tensorboard中
    # writer.add_image('face_landmark', img_grid)
    # writer.add_graph(model, images)

    # 训练模型的次数
    num_epochs = 10
    # SGD or Adam，初步测试Adam会更好用一点
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        for i_batch, sample_batched in enumerate(dataloader):
            images_batch, landmarks_batch = \
                sample_batched['image'], sample_batched['landmarks']

            # cuda加速
            if train_on_gpu:
                images_batch, landmarks_batch = images_batch.cuda(), landmarks_batch.cuda()

            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(images_batch)
            # calculate the batch loss
            loss = myloss_fn(output, landmarks_batch)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()
            # 计算平均损失
        train_loss = train_loss / num_train_samples

        writer.add_scalar('training loss',
                          train_loss,
                          epoch * len(dataloader) + i_batch)

        # 显示训练集与验证集的损失函数
        print('Epoch: {} \tTraining Loss: {:.6f} '.format(epoch, train_loss))

    # save model
    # torch.save保存模型兼容性太差，不同环境可能会报错，类似_pickle.picklingerror
    # 还是推荐使用字典保存模型参数
    torch.save(model, './model/landmark_full.pt')
    # torch.save(model.state_dict(), './model/landmark.pt')
    model.eval()

    writer.close()
