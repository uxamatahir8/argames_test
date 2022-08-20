import os
import time
import cv2
import numpy as np
import random
import datetime as dt


class ObjectDetection:
    def __init__(self):
        PROJECT_PATH = os.path.abspath(os.getcwd())
        MODELS_PATH = os.path.join(PROJECT_PATH, "models")
        print(PROJECT_PATH)
        print(MODELS_PATH)
        self.MODEL = cv2.dnn.readNetFromDarknet(
            os.path.join(MODELS_PATH, "cross-hands.cfg"),
            os.path.join(MODELS_PATH, "cross-hands.weights")
            
        )

        self.CLASSES = ['hand']
        # with open(os.path.join(MODELS_PATH, "coco.names"), "r") as f:
        #     self.CLASSES = [line.strip() for line in f.readlines()]

        self.OUTPUT_LAYERS = [
            self.MODEL.getLayerNames()[i - 1] for i in self.MODEL.getUnconnectedOutLayers()
        ]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        self.COLORS /= (np.sum(self.COLORS**2, axis=1)**0.5/255)[np.newaxis].T

    def detectObj(self, snap):
        height, width, channels = snap.shape
        blob = cv2.dnn.blobFromImage(
            snap, 1/255, (416, 416), swapRB=True, crop=False
        )

        self.MODEL.setInput(blob)
        outs = self.MODEL.forward(self.OUTPUT_LAYERS)

        # ! Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # * Object detected
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    # * Rectangle coordinates
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.CLASSES[class_ids[i]])
                color = self.COLORS[i]
                cv2.rectangle(snap, (x, y), (x + w, y + h), color, 2)
                cv2.putText(snap, label, (x, y - 5), font, 2, color, 2)
        return snap


class VideoStreaming(object):
    def __init__(self):
        super(VideoStreaming, self).__init__()
        self.VIDEO = cv2.VideoCapture(0)
        # self.VIDEO = cv2.VideoCapture("Watch1.mp4")

        self.MODEL = ObjectDetection()

        self._preview = True
        self._flipH = False
        self._detect = False
        self._exposure = self.VIDEO.get(cv2.CAP_PROP_EXPOSURE)
        self._contrast = self.VIDEO.get(cv2.CAP_PROP_CONTRAST)

    @property
    def preview(self):
        return self._preview

    @preview.setter
    def preview(self, value):
        self._preview = bool(value)

    @property
    def flipH(self):
        return self._flipH

    @flipH.setter
    def flipH(self, value):
        self._flipH = bool(value)

    @property
    def detect(self):
        return self._detect

    @detect.setter
    def detect(self, value):
        self._detect = bool(value)

    @property
    def exposure(self):
        return self._exposure

    @exposure.setter
    def exposure(self, value):
        self._exposure = value
        self.VIDEO.set(cv2.CAP_PROP_EXPOSURE, self._exposure)

    @property
    def contrast(self):
        return self._contrast

    @contrast.setter
    def contrast(self, value):
        self._contrast = value
        self.VIDEO.set(cv2.CAP_PROP_CONTRAST, self._contrast)

    def show(self):
        lastTime = dt.datetime.now()
        currentTime = dt.datetime.now()
        fps = self.VIDEO.get(cv2.CAP_PROP_FPS)
        ball_images = ['ball.png' , 'ball1.png' , 'ball2.png']
        ball_image_index = 0
        x_balls =[]
        y_balls = []
        score = 0
        ball_length = 30
        balls = []
        k = 0
        for i in range(0 , ball_length):
            x_balls.insert(len(x_balls) , random.randint(20 , int(self.VIDEO.get(cv2.CAP_PROP_FRAME_WIDTH))-20))
            y_balls.insert(len(y_balls) , 0)
            balls.insert(len(balls) , random.randint(0 , len(ball_images)-1))
            # balls.insert(len(balls) , ball_image_index)
            # ball_image_index += 1
            # ball_image_index %= len(ball_images)  
        while(self.VIDEO.isOpened()):
            ret, snap = self.VIDEO.read()
            if self.flipH:
                snap = cv2.flip(snap, 1)
            currentTime = dt.datetime.now()
            if ret == True:

                add =random.randint(10 , 25)
                frame_id  = self.VIDEO.get(cv2.CAP_PROP_POS_FRAMES)

                if (currentTime-lastTime).seconds > 0.3:
                    lastTime = dt.datetime.now()
                    k +=1
                    print(k) 
                # if int(frame_id) % (int(int(fps+0.5)*0.5)) == 0:
                #     k+=1
                #     if k >=3:
                #         k -= 1
                #     print(k)
                # for i in range(0, k+1):
                for i in range(0, k+1):
                    s_img = cv2.imread(ball_images[balls[i]], -1)
                    # s_img = cv2.resize(s_img, (30, 30))
                    alpha_s = s_img[:, :, 3] / 255.0
                    alpha_l = 1.0 - alpha_s
                    if y_balls[i] + s_img.shape[1] + add < snap.shape[0] :
                        y_balls[i] += add
                    else:
                        y_balls[i] = 0
                        x_balls[i] = random.randint(500 , 1000)
                        score -= 1
                    y1, y2 = y_balls[i], y_balls[i] + s_img.shape[0]
                    x1, x2 = x_balls[i], x_balls[i] + s_img.shape[1]
                    for c in range(0, 3):
                        snap[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] + alpha_l * snap[y1:y2, x1:x2, c])

                if self._preview:
                    # snap = cv2.resize(snap, (0, 0), fx=0.5, fy=0.5)
                    if self.detect:
                        snap = self.MODEL.detectObj(snap)

                else:
                    snap = np.zeros((
                        int(self.VIDEO.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        int(self.VIDEO.get(cv2.CAP_PROP_FRAME_WIDTH))
                    ), np.uint8)
                    label = "camera disabled"
                    H, W = snap.shape
                    font = cv2.FONT_HERSHEY_PLAIN
                    color = (255, 255, 255)
                    cv2.putText(snap, label, (W//2 - 100, H//2),
                                font, 2, color, 2)

                frame = cv2.imencode(".jpg", snap)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.01)

            else:
                break
        print("off")
