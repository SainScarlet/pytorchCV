import torch
import cv2 as cv
import numpy as np
model_bin = "/Users/remilia/Documents/02-Work/05-Python/02_pytorchCV/Landmark/model/" \
            "opencv_face_detector_uint8.pb"
config_text = "/Users/remilia/Documents/02-Work/05-Python/02_pytorchCV/Landmark/model" \
              "/opencv_face_detector.pbtxt"


def image_landmark_demo():
    # 如果加载字典数据需要先构建模型，所以最好还是使用onnx
    # cnn_model = torch.load("./model/landmark.pt")

    # onnx
    cnn_model = cv.dnn.readNetFromONNX('./model/landmark.onnx')

    # 测试图片
    image = cv.imread("/Users/remilia/Documents/02-Work/05-Python/0-source/test6.png")
    cv.imshow("input", image)

    h, w, c = image.shape
    # 同模型输入调整推理输入
    img = cv.resize(image, (64, 64))
    img = (np.float32(img) / 255.0 - 0.5) / 0.5
    # h, w, c --> c, h, w
    img = img.transpose((2, 0, 1))

    # onnx的大坑！！
    # 特别注意Netron查看一下onnx的输入类型，onnx的输入这里类型为float32
    # 故前面也好，这里也是，不能按照模型的tensor输入
    # x_input = torch.from_numpy(img).view(1, 3, 64, 64)
    x_input = img.reshape(1, 3, 64, 64)
    print(x_input.shape)

    # probs = cnn_model.forward(x_input)

    cnn_model.setInput(x_input)
    probs = cnn_model.forward()
    print(probs.shape)
    lm_pts = probs.reshape(5, 2)
    print(lm_pts)
    # # # probs = cnn_model(x_input.cuda())
    # lm_pts = probs.view(5, 2).cpu().detach().numpy()
    # print(lm_pts)
    #
    for x, y in lm_pts:
        print(x, y)
        x1 = x*w
        y1 = y*h
        cv.circle(image, (np.int32(x1), np.int32(y1)), 2, (0, 0, 255), 2, 8, 0)
    cv.imshow("face_landmark_Pic", image)
    #
    # cv.imwrite("/Users/remilia/Documents/02-Work/05-Python/0-source/Landmark_result", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def video_landmark_demo():
    # cnn_model = torch.load("./model_landmarks.pt")

    cnn_model = cv.dnn.readNetFromONNX('./model/landmark.onnx')
    capture = cv.VideoCapture(0)
    # capture = cv.VideoCapture("/Users/remilia/Documents/02-Work/05-Python/0-source/TestVideo.mp4")
    # 笨办法，复制会很奇怪的错，后面需要优化
    # cap = cv.VideoCapture("/Users/remilia/Documents/02-Work/05-Python/0-source/TestVideo.mp4")

    # load tensorflow model
    # 第三方训练好的tensorflow模型
    # github上很多，需要注意输入格式
    net = cv.dnn.readNetFromTensorflow(model_bin, config=config_text)
    while True:
        ret, frame = capture.read()
        # ret, pic = cap.read()
        if ret is not True:
            break

        # # 检测视频和模型是否正确加载，测试完成后注释
        # cv.imshow('test', frame)
        # if cv.waitKey(24) & 0xFF == ord(' '):
        #     break

        # 图像翻转，内置摄像头时使用
        frame = cv.flip(frame, 1)

        h, w, c = frame.shape
        # cv中的transform
        # 输入图像，缩放比裁剪，减均值（104，177，123 论文中常用的三个值）
        blobImage = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        net.setInput(blobImage)
        cvOut = net.forward()

        # # 绘制检测矩形
        for detection in cvOut[0, 0, :, :]:
            score = float(detection[2])
            print(score)
            if score > 0.9:
                left = detection[3]*w
                top = detection[4]*h
                right = detection[5]*w
                bottom = detection[6]*h
                print(left, top, right, bottom)

                if left > 0 and top > 0 and right > 0 and bottom > 0:

                    # roi和landmark部分
                    roi = frame[np.int32(top):np.int32(bottom), np.int32(left):np.int32(right), :]
                    # print(roi.shape)
                    cv.imshow("test", roi)

                    rw = right - left
                    rh = bottom - top
                    img = cv.resize(roi, (64, 64))
                    img = (np.float32(img) / 255.0 - 0.5) / 0.5
                    img = img.transpose((2, 0, 1))
                    # x_input = torch.from_numpy(img).view(1, 3, 64, 64)
                    x_input = img.reshape(1, 3, 64, 64)

                    cnn_model.setInput(x_input)
                    probs = cnn_model.forward()
                    # probs = cnn_model(x_input.cuda())
                    lm_pts = probs.reshape(5, 2)
                    # lm_pts = probs.view(5, 2).cpu().detach().numpy()

                    for x, y in lm_pts:
                        x1 = x * rw
                        y1 = y * rh
                        cv.circle(roi, (np.int32(x1), np.int32(y1)), 4, (0, 0, 255), 6, 8, 0)

                    # 绘制矩形
                    cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), thickness=3)
                    cv.putText(frame, "score:%.2f" % score, (int(left), int(top)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    # w_n = int(0.35*w)
                    # h_n = int(0.35*h)
                    # frame = cv.resize(frame, (w_n, h_n))
                    # pic = cv.resize(pic, (w_n, h_n))

                    cv.imshow("face detection & landmark", frame)
                    # cv.imshow("original", pic)

                    cv.moveWindow("face detection & landmark", 250, -100)
                    # cv.moveWindow("original", 0, 0)

                    if cv.waitKey(24) & 0xFF == ord(' '):
                        break

            #         c = cv.waitKey(1)
            #         if c == 27:
            #             break
            #         cv.imshow("face detection + landmark", frame)

    capture.release()
    # cap.release()
    cv.destroyAllWindows()
    # 视频退出时有时候会卡住，这样就不会卡了
    cv.waitKey(1)


if __name__ == "__main__":
    video_landmark_demo()
    # image_landmark_demo()
