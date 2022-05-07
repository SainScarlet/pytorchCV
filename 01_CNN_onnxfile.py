import torch as t
# 注意python的命名要求，数字不能开头,故01_CNN改名为CNN
# from CNN import CNN_Mnist

# 保存成onnx格式
# model = CNN_Mnist()

model = t.nn.Sequential(
    t.nn.Linear(784, 100),
    t.nn.ReLU(),
    t.nn.Linear(100, 10),
    t.nn.LogSoftmax(dim=1)
)

model.load_state_dict(t.load('./0_model/01_nn_mnist_model.pt'))
# model.load_state_dict(t.load('./0_model/02_cnn_mnist_model.pt'))
model.eval()

# dummy一个虚假的输入数据格式
dummy_input = t.randn(1, 1, 784)
# 最好在函数中指定in和out的名字，防止netron显示时产生误会，特别示output
# 注意in out的写法，需加入[]
# t.onnx.export(model, dummy_input, './0_model/02_cnn_mnist_model.onnx',
#               verbose=True, input_names=['ImgInput'], output_names=['PredictNumber'])
t.onnx.export(model, dummy_input, './0_model/01_nn_mnist_model.onnx',
              verbose=True, input_names=['ImgInput'], output_names=['PredictNumber'])
