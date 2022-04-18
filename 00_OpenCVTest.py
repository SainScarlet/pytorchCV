import cv2 as cv
import numpy as np

# image = cv.imread("/Users/remilia/Documents/02-Work/05-Python/0-source/Lenna.jpeg")
#
# h, w, c = image.shape  # HWC
# print(h, w, c)
#
# # 目的就是不同模型要求的输入图像顺序不一样，openCV读到的是HWC，转换为CHW
# blob = np.transpose(image, (2, 0, 1))
# print(blob.shape)  # CHW
#
# # 转换为浮点数，/255目的是归一化
# fi = np.float32(image) / 255.0  # 0 ~1
# cv.imshow("fi", fi)
#
# # 转换为灰度图
# # 切片方法：获取三通道数据image[:,:,(0,1,2)]
# gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# # 灰度图没有通道
# print(gray.shape)
# cv.imshow("gray", gray)
#
# # 图像拉伸
# dst = cv.resize(image, (224, 224))
# cv.imshow("zoom out", dst)
#
# # ROI区域，即部分截图
# box = [200, 200, 200, 200]  # x, y, w, h
# # 同上面的通道切片操作
# roi = image[200:400, 200:400, :]
# cv.imshow("roi", roi)
#
# # 创建一个黑图
# m1 = np.zeros((512, 512, 3), dtype=np.uint8)
# # 不修改尺寸，改通道颜色
# m1[:, :] = (0, 0, 255),  # BGR
# cv.imshow("m1", m1)
#
# # 画框
# # 左上角坐标，右下角坐标，颜色，线宽，线型，shift为坐标偏移量，即输入坐标/2^x画图
# cv.rectangle(image, (200, 200), (400, 400), (0, 0, 255), 2, 8)
# cv.imshow("input", image)
# cv.waitKey(0)


# 摄像头捕获
cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret is not True:
        break
    cv.rectangle(frame, (500, 500), (1000, 1000), (0, 0, 255), 2, 8, 5)
    cv.rectangle(frame, (500, 500), (1000, 1000), (255, 0, 255), 2, 8, 0)
    cv.imshow("frame", frame)
    # 如果没有需要的话，在videoCapture中waitkey=1
    c = cv.waitKey(24)
    if c == 27:  # ESC 键值
        break

# 释放资源并关闭窗口
cap.release()
cv.destroyAllWindows()
# 视频退出时有时候会卡住，这样就不会卡了
cv.waitKey(1)
