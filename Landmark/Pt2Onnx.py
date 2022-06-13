import torch as t
# from Landmark_CNN import Net
from Landmark_CNN_GPU import Net, ChannelPool

# model = Net()

# # 字典数据的加载
# model.load_state_dict(t.load('./model/landmark.pt'))
# model.eval()

# 完整模型的加载
# M1加载CUDA训练完的模型有点问题，需要尝试
model = t.load('./model/landmark_full_1w.pt')
model.cpu()

# dummy一个虚假的输入数据格式
dummy_input = t.randn(1, 3, 64, 64)
# 最好在函数中指定in和out的名字，防止netron显示时产生误会，特别示output
# 注意in out的写法，需加入[]
# t.onnx.export(model, dummy_input, './model/landmark.onnx',
#               verbose=True, input_names=['ImgInput'], output_names=['LandmarkPoint'])

t.onnx.export(model, dummy_input, './model/landmark_1w.onnx',
              verbose=True, input_names=['ImgInput'], output_names=['LandmarkPoint'])
