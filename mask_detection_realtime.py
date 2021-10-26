"""
pip install keras
pip install tensorflow
"""
import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
import os
import time

#
import playsound
#

# ref : DataFlair
# https://data-flair.training/blogs/face-mask-detection-with-python/
# Real-Time Face Mask Detector with Python, OpenCV, Keras

model=load_model("./mask_detection_model")

dirname = os.path.dirname(__file__)
f_haar_path = os.path.join(dirname, 'haarcascade_frontalface_default.xml') # path ของ haarcascade face
# ใช้ haarcascade classifier
face_detector = cv2.CascadeClassifier(f_haar_path)

#read video
webcam = cv2.VideoCapture(0)
ret, img = webcam.read()
img_h, img_w = img.shape[:2]
size = 4


while True:
	ret, new_img = webcam.read()
	# gray = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
	# สำหรับตรวจจับใบหน้า
	faces = face_detector.detectMultiScale(new_img, 1.3, 5)
	for x, y, w, h in faces:
		face_img = new_img[y:x+h, x:x+w] # ดึงพิกัดใบหน้า
		resized = cv2.resize(face_img, (224, 224)) # ให้ภาพใบหน้า fit กับโมเดล (224,224)
		img_array = tf.keras.preprocessing.image.img_to_array(resized) # แปลงใบหน้าเป็น array
		img_array = tf.expand_dims(img_array, 0) #ขยายมิติภาพฟิตกับโมดล
		predictions = model.predict(img_array) # ทำนายบน ROI (Region of Interest)
		score = tf.nn.softmax(predictions[0]) # ผลลัพธ์
		label = np.argmax(score)

		# Post-Processing

		cv2.rectangle(new_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
		if label == 0:
			cv2.putText(new_img, "mask", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

		if label == 1:
			cv2.putText(new_img, "No mask", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

		# pass
		# แสดงผลหลังจากทำนาย
	new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
	cv2.imshow('live-Image', new_img)
	# eval_js('showimg("{}")'.format(image2byte(new_img)))
	# print(np.argmax(score), 100 * np.max(score))


	# if Esc key is press then break out of the loop 

	# สำหรับหยุดการรัน
	if cv2.waitKey(1) == ord('q'):
		break
		
# ปิดวิดีโอ
webcam.release()
# ปิดหน้าต่าง
cv2.destroyAllWindows()