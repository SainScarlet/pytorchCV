# 将新版本的特性引进当前版本中
# 在原本python2中引入python3的写法
from __future__ import print_function

import cv2 as cv
import numpy as np
import torch
import logging as log
from VehicleAttribute_model import ResidualBlock, VehicleAttributesResNet
from openvino.inference_engine import IENetwork, IECore

color_labels = ["white", "gray", "yellow", "red", "green", "blue", "black"]
type_labels = ["car", "bus", "truck", "van"]

# model_dir = "/Users/remilia/Documents/02-Work/05-Python/02_pytorchCV/Vehicle_Attribute/model/"
model_dir = "D:/2_Work/1_python/pytorchCV/Vehicle_Attribute/model/"
model_xml = model_dir + "vehicle-detection-adas-0002.xml"
model_bin = model_dir + "vehicle-detection-adas-0002.bin"

net = IENetwork(model=model_xml, weights=model_bin)
# net = IECore.read_network(model=model_xml, weights=model_bin)

ie = IECore()
log.info("Device info:")
versions = ie.get_versions("CPU")

# pt完整模型加载
cnn_model = torch.load("./model/vehicle_attributes_demo.pt")
print(cnn_model)

# cnn_model = cv.dnn.readNetFromONNX('./model/vehicle_attributes_demo.onnx')

input_blob = next(iter(net.inputs))
n, c, h, w = net.inputs[input_blob].shape
print(n, c, h, w)

# capture = cv.VideoCapture("/Users/remilia/Documents/02-Work/05-Python/0-source/VAtest.mp4")
capture = cv.VideoCapture("D:/2_Work/1_python/0-source/VAtest.mp4")
# capture = cv.VideoCapture(0)

ih = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
iw = capture.get(cv.CAP_PROP_FRAME_WIDTH)

input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))

log.info("Loading model to the device")
exec_net = ie.load_network(network=net, device_name="CPU")
log.info("Creating infer request and starting inference")

while True:
    ret, src = capture.read()

    print(ret, src)

    if ret is not True:
        break
    images = np.ndarray(shape=(n, c, h, w))
    images_hw = []
    ih, iw = src.shape[:-1]
    images_hw.append((ih, iw))
    if (ih, iw) != (h, w):
        image = cv.resize(src, (w, h))
    # HWC to CHW
    image = image.transpose((2, 0, 1))
    images[0] = image
    res = exec_net.infer(inputs={input_blob: images})

    # 解析SSD输出内容
    res = res[out_blob]
    license_score = []
    license_boxes = []
    data = res[0][0]
    index = 0
    for number, proposal in enumerate(data):
        if proposal[2] > 0.75:
            ih, iw = images_hw[0]
            label = np.int(proposal[1])
            confidence = proposal[2]
            xmin = np.int(iw * proposal[3])
            ymin = np.int(ih * proposal[4])
            xmax = np.int(iw * proposal[5])
            ymax = np.int(ih * proposal[6])

            # 防止box超出画面边界导致的卡死
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax >= iw:
                xmax = iw - 1
            if ymax >= ih:
                ymax = ih - 1

            vehicle_roi = src[ymin:ymax, xmin:xmax, :]
            img = cv.resize(vehicle_roi, (72, 72))
            img = (np.float32(img) / 255.0 - 0.5) / 0.5
            img = img.transpose((2, 0, 1))
            x_input = torch.from_numpy(img).view(1, 3, 72, 72)
            # x_input = img.reshape(1, 3, 72, 72)

            color_, type_ = cnn_model(x_input.cuda())
            color_, type_ = cnn_model(x_input)

            # cnn_model.setInput(x_input)
            # color_, type_ = cnn_model.forward()

            # torch.max()[1]只返回最大值的每个索引
            predict_color = torch.max(color_, 1)[1].cpu().detach().numpy()[0]
            predict_type = torch.max(type_, 1)[1].cpu().detach().numpy()[0]

            # # onnx
            # predict_color = color_.index(max(color_))
            # predict_type = type_.index(max(type_))

            attrs_txt = "color:%s, type:%s" % (color_labels[predict_color], type_labels[predict_type])
            cv.rectangle(src, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv.putText(src, attrs_txt, (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv.imshow("Vehicle Attributes Recognition Demo", src)
    res_key = cv.waitKey(1)
    if res_key == 27:
        break
