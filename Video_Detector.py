
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.models import Model

from imutils.video import VideoStream
import imutils
import time

import tensorflow as tf

import cv2
import numpy as np

model = tf.keras.models.load_model('imagenet_model.h5')

base_dir = "C:\\Users\\srira\\Documents\\Python\\ImageClassification\\Resnet Rohan\\Face_Detector"
prototxt_path = "C:\\Users\\srira\\Documents\\Python\\ImageClassification\\Resnet Rohan\\Face_Detector/deploy.prototxt"
caffemodel_path = "C:\\Users\\srira\\Documents\\Python\\ImageClassification\\Resnet Rohan\\Face_Detector/weights.caffemodel"

# Read the model
face_model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)


while True:
    
    frame = vs.read()
    frame = imutils.resize(frame, width=800)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_model.setInput(blob)
    detections = face_model.forward()

    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        confidence = detections[0, 0, i, 2]

      # If confidence > 0.5, show box around face
        if (confidence > 0.5):
            image = frame[startY:endY, startX:endX] #The extracted face
            image = cv2.resize(image, (224,224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)

            (noMask, mask, incorrectMask) = model.predict(image)[0]

            if max(mask, noMask, incorrectMask)==mask:
            	label = "Mask" 
            	color = (0, 255, 0)
            elif max(mask, noMask, incorrectMask)==noMask:
            	label = "No Mask"
            	color = (0, 0, 255)
            else:
            	label = "incorrectMask"
            	color = (0, 255, 255)

        	
            label = "{}: {:.2f}%".format(label, max(mask, noMask, incorrectMask) * 100)
            cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
    key = cv2.waitKey(1) & 0xFF

    cv2.imshow('Video', frame)

    if key == ord("q"):
        break

cv2.destroyAllWindows()   
vs.stop()