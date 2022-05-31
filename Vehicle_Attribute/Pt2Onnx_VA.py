import torch as t
from VehicleAttribute_model import VehicleAttributesResNet

model = VehicleAttributesResNet()

model.load_state_dict(t.load('./model/vehicle_attributes_demo.pt'))
model.eval()

# dummy一个虚假的输入数据格式
dummy_input = t.randn(1, 3, 64, 64)
# 最好在函数中指定in和out的名字，防止netron显示时产生误会，特别示output
# 注意in out的写法，需加入[]
# t.onnx.export(model, dummy_input, './0_model/02_cnn_mnist_model.onnx',
#               verbose=True, input_names=['ImgInput'], output_names=['PredictNumber'])
t.onnx.export(model, dummy_input, './model/vehicle_attributes_demo.onnx',
              verbose=True, input_names=['ImgInput'], output_names=['VA_color', 'VA_type'])
