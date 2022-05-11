import numpy as np
import cv2 as cv


def mnist_onnx():
    mnist_net = cv.dnn.readNetFromONNX('./0_model/02_cnn_mnist_model.onnx')

    img = img_pre()
    print(img.shape)

    # 预测
    mnist_net.setInput(img)
    result = mnist_net.forward()
    pred_label = np.argmax(result, 1)
    print("predict label : %d" % pred_label)
    cv.waitKey(0)
    cv.destroyAllWindows()


def img_pre():
    img0 = cv.imread('/Users/remilia/Documents/02-Work/05-Python/0-source/mnist_test_320x328_7.jpg', -1)
    # 预处理，降噪
    img = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (5, 5), 0, 0)
    img = cv.GaussianBlur(img, (3, 3), 0, 0)
    c = 0.8
    b = 90
    h, w = img.shape
    black_image = np.zeros([h, w], img.dtype)
    img = cv.addWeighted(img, c, black_image, 1 - c, b)

    # 阈值化，裁切，拉伸mnist数据形式
    ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
    roi = thresh[50:240, 90:250]
    roi = cv.bitwise_not(roi)
    roi_size = cv.resize(roi, (28, 28))

    # 输入标准化，归一化，中心化等, 0.00392=1/255
    img_result = cv.dnn.blobFromImage(roi_size, 0.00392, (28, 28), 127.0) / 0.5

    # 显示
    cv.imshow('Original', img0)
    cv.imshow('ImgResult', roi_size)
    return img_result


if __name__ == "__main__":
    mnist_onnx()
