"""
refernce:
https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/

pip install tensorflow 
pip install opencv-python

"""
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2
import tensorflow as tf

face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model('teachablemachine_keras/keras_model.h5')
size = (224, 224)

cap = cv2.VideoCapture(0)

while True:
	success, image_bgr = cap.read()
	image_org = image_bgr.copy()
	image_bw = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
	image_rgb = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
	faces = face_classifier.detectMultiScale(image_bw)
	for (x, y, w, h) in faces:
		face_rgb = Image.fromarray(image_rgb[y:y+h, x:x+w])
		data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
		image = face_rgb
		image = ImageOps.fit(image, size, Image.ANTIALIAS)
		im_array = np.asarray(image)
		nomalized_image_array = (im_array.astype(np.float32) / 127.0) - 1

		data[0] = nomalized_image_array
		pred = model.predict(data)
		score = tf.nn.softmax(pred[0])
		label = np.argmax(score)
		print(label)
		if pred[0][0] > pred[0][1]:
			cv2.putText(image_bgr, 'With_Mask', (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
		else:
			cv2.putText(image_bgr, 'No_mask', (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 0, 255), 2)


	cv2.imshow('LIVE',   image_bgr)
	key = cv2.waitKey(1)
	if key == ord('q'): 
		break

cap.release()
cv2.destroyAllWindows()