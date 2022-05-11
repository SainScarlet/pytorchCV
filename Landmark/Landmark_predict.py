import torch
import cv2 as cv
import numpy as np
model_bin = "D:/projects/opencv_tutorial/data/models/face_detector/opencv_face_detector_uint8.pb";
config_text = "D:/projects/opencv_tutorial/data/models/face_detector/opencv_face_detector.pbtxt";


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
    cnn_model = torch.load("./model_landmarks.pt")
    # capture = cv.VideoCapture(0)
    capture = cv.VideoCapture("D:/images/video/example_dsh.mp4")

    # load tensorflow model
    net = cv.dnn.readNetFromTensorflow(model_bin, config=config_text)
    while True:
        ret, frame = capture.read()
        if ret is not True:
            break
        frame = cv.flip(frame, 1)
        h, w, c = frame.shape
        blobImage = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        net.setInput(blobImage)
        cvOut = net.forward()
        # 绘制检测矩形
        for detection in cvOut[0, 0, :, :]:
            score = float(detection[2])
            if score > 0.5:
                left = detection[3]*w
                top = detection[4]*h
                right = detection[5]*w
                bottom = detection[6]*h

                # roi and detect landmark
                roi = frame[np.int32(top):np.int32(bottom), np.int32(left):np.int32(right),:]
                rw = right - left
                rh = bottom - top
                img = cv.resize(roi, (64, 64))
                img = (np.float32(img) / 255.0 - 0.5) / 0.5
                img = img.transpose((2, 0, 1))
                x_input = torch.from_numpy(img).view(1, 3, 64, 64)
                probs = cnn_model(x_input.cuda())
                lm_pts = probs.view(5, 2).cpu().detach().numpy()
                for x, y in lm_pts:
                    x1 = x * rw
                    y1 = y * rh
                    cv.circle(roi, (np.int32(x1), np.int32(y1)), 2, (0, 0, 255), 2, 8, 0)

                # 绘制
                cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), thickness=2)
                cv.putText(frame, "score:%.2f" % score, (int(left), int(top)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                c = cv.waitKey(1)
                if c == 27:
                    break
                cv.imshow("face detection + landmark", frame)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    # video_landmark_demo()
    image_landmark_demo()
