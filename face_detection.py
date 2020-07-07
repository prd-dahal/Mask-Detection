from facenet_pytorch import MTCNN
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class FaceDetection:
    def __init__(self, image_path):
        self.image_path = image_path

    def detect_face(self):
        img = cv2.imread(self.image_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        mtcnn =  MTCNN(margin=40, select_largest=False, post_process=False)
        face = mtcnn(img)
        face =face.permute(1,2,0).int().numpy()
        face_img = Image.fromarray(face.astype('uint8'),'RGB')
        return face_img






