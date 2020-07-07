from facenet_pytorch import MTCNN
import cv2
import matplotlib.pyplot as plt


class FaceDetection:
    def __init__(self, image_path):
        self.image = image_path

    def detect_face(self):
        img = cv2.cvtColor(cv2.imread(self.image),cv2.COLOR_BGR2RGB)
        detector = MTCNN()
        #detector.detect_faces(img)
        print(detector)

image_path = 'Examples\\example_02.png'

fd = FaceDetection(image_path)
fd.detect_face()
