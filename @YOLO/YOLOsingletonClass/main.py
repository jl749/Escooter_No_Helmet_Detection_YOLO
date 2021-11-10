import cv2
import numpy as np

BLOB_SIZE = (320, 320)
CONFIDENCE = 0.5
YOLO_PATH = ''


def readImg(myImg: np.ndarray):
    height, width, channels = myImg.shape
    myBlob = cv2.dnn.blobFromImage(myImg, 0.00392, BLOB_SIZE, (0, 0, 0), True, crop=False)
    return (height, width), myBlob


class MyYolo:
    __instance = None
    font = cv2.FONT_HERSHEY_SIMPLEX

    @staticmethod
    def GET():
        """ Static access method. """
        if MyYolo.__instance is None:
            MyYolo()
        return MyYolo.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if MyYolo.__instance is not None:
            raise Exception("Singleton class can not be retrieved")
        else:
            MyYolo.__instance = self
            self.net = cv2.dnn.readNet(YOLO_PATH + "yolov3_E.weights", YOLO_PATH + "yolov3_E.cfg")
            self.classes = []
            with open(YOLO_PATH + "obj.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]

            self.layer_names = self.net.getLayerNames()  # 'conv_0', 'bn_0', 'leaky_1', 'conv_1', 'bn_1'
            self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def forward(self, img):
        size, blob = readImg(img)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # interpret result on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > CONFIDENCE:
                    center_x = int(detection[0] * size[1])
                    center_y = int(detection[1] * size[0])
                    w = int(detection[2] * size[1])
                    h = int(detection[3] * size[0])

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        return class_ids, indexes, boxes

    def _extractLabels(self, class_ids, indexes, boxes):
        # labels = []
        # for i in range(len(boxes)):
        #     if i in indexes:
        #         label = self.classes[class_ids[i]]
        #         labels.append(label)
        # return labels
        return [self.classes[class_ids[i]] for i in range(len(boxes)) if i in indexes]

    def mark_objs(self, img, class_ids, indexes, boxes):
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = self.classes[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, label, (x, y + 30), self.font, 1, (0, 255, 0), 1)

        return img

    def processVideo(self, videoPath, savePath):
        cap = cv2.VideoCapture(videoPath)  # ~/video.mp4
        if not cap.isOpened():
            print("failed to read VideoCapture object")
            return

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        # VideoWriter_fourcc('M', 'J', 'P', 'G') / cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(savePath, 0x7634706d , 10, (frame_width, frame_height))
        while cap.isOpened():
            ret, frame = cap.read()  # read next frame
            if ret:
                # cv2.imshow("Vid_Frame", frame)
                class_ids_, indexes_, boxes_ = self.forward(frame)
                # objsDetected = model.extractLabels(class_ids_, indexes_, boxes_)
                frame = self.mark_objs(frame, class_ids_, indexes_, boxes_)
                out.write(frame)
                # if (cv2.waitKey(30) & 0xFF) == ord('q'):  # EventListener
                #     break
            else:
                break

        cap.release()

    def processCam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("failed to read VideoCapture object")
            return

        while cap.isOpened():
            ret, frame = cap.read()  # read next frame
            if ret:
                class_ids_, indexes_, boxes_ = self.forward(frame)
                # objsDetected = model.extractLabels(class_ids_, indexes_, boxes_)
                frame = self.mark_objs(frame, class_ids_, indexes_, boxes_)
                cv2.imshow("YOLOv3", frame)
                if (cv2.waitKey(30) & 0xFF) == ord('q'):
                    break
            else:
                break

        cap.release()


if __name__ == '__main__':
    model = MyYolo().GET()
    # model.processVideo('scooter_test2.mp4', 'out.mp4')
    model.processCam()

    cv2.destroyAllWindows()



