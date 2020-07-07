import cv2
import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

def preprocessing(image_path):
    image = tf.keras.preprocessing.image.load_img(
    image_path, grayscale=False, color_mode="rgb", target_size=(224,224), interpolation="nearest"
    )
    arr = input_arr = tf.keras.preprocessing.image.img_to_array(image)
    arr = np.array(arr)
    arr = np.expand_dims(arr, axis=0)  
    return arr
    
#json_file = open('face_mask_detection_structure.json','r')
#loaded_model_json = json_file.read()
#json_file.close()


loaded_model = load_model('saved_models\\mask_detector.model')

# load weights

#loaded_model.load_weights('face_mask_detection_weights.h5')

print("Loaded Model From the Disk")


image = preprocessing('Examples\\test_example1.jpg')
mask_image = preprocessing('Examples\\example_01.png')
non_mask_image = preprocessing('Examples\\example_02.png')

print("The prediction for mask_image is {}".format(loaded_model.predict(mask_image)))
print("The prediction for non_mask_image is {}".format(loaded_model.predict(non_mask_image)))
print("The prediction for example_mask_image is {}".format(loaded_model.predict(image)))


