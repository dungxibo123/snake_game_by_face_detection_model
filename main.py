import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow_hub as hub
labels = ['down','left','right','up']

vid = cv2.VideoCapture(0)
#model.save('./model/saved.h5')
model = tf.keras.models.load_model('FirstModel.h5')
model.summary()
# font
font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (50, 50)

# fontScale
fontScale = 1

# Blue color in BGR
color = (255, 0, 0)
face_cascade = cv2.CascadeClassifier('data_generator/haarcascade_frontalface_default.xml')

# Line thickness of 2 px
thickness = 2
while(True):
    ret, frame = vid.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)
    img = frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# Draw rectangle around the faces
    for (x, y, w, h) in faces:

        predict_img = img[y:y+h, x:x+w,:]
        predict_img = cv2.resize(predict_img,(224,224),interpolation = cv2.INTER_AREA)
        predict_img = tf.expand_dims(predict_img,axis=0)
        pre = model.predict(predict_img)
        print(pre)
        lab = labels[np.argmax(pre)]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img,lab,(x, y-30),font,fontScale,color,thickness,cv2.LINE_AA )
        print('Prediction = {}'.format(lab))
        #saved_img = img[y: y+h, x:x+w, :]
        #cv2.imwrite('data/left/{}.jpg'.format(count), saved_img)
    cv2.imshow('Frame',img)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cv2.waitKey()

