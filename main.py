import cv2
import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from PIL import Image
from face_detection import FaceDetection

def preprocessing(image):

    resized_image = image.resize((224,224))          
    arr = np.array(resized_image)
    arr = np.expand_dims(arr, axis=0)  
    return arr
    
#json_file = open('face_mask_detection_structure.json','r')
#loaded_model_json = json_file.read()
#json_file.close()


loaded_model = load_model('saved_models/mask_detector.model')

# load weights

#loaded_model.load_weights('face_mask_detection_weights.h5')

print("Loaded Model From the Disk")


mask_image_face_object =  FaceDetection('Examples/example_01.png')
non_mask_image_face_object = FaceDetection('Examples/example_02.png')

mask_image = preprocessing(mask_image_face_object.detect_face())
non_mask_image = preprocessing(non_mask_image_face_object.detect_face())

print("The prediction for mask_image is {}".format(loaded_model.predict(mask_image)))
print("The prediction for non_mask_image is {}".format(loaded_model.predict(non_mask_image)))
#print("The prediction for example_mask_image is {}".format(loaded_model.predict(image)))


