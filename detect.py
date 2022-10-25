import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
from keras.utils import plot_model
import os
import scipy
import tensorflow as tf

emotion_key = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

cam = cv2.VideoCapture(0)

emotion_model = keras.models.load_model("model.h5")

while True:

    ret, frame = cam.read()

    if not ret:
        break

    bounding_box = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3,minNeighbors=5)

    for (x,y,w,h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+2, y+h+10), (255,0,0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x+w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48,48)), -1), 0)
        emotion_prediction = emotion_model.predict(cropped_img)
        maxIndex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_key[maxIndex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame, (600,500), interpolation=cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()