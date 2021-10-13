"""
pip install keras
pip install tensorflow
"""
import cv2
import numpy as np
from keras.models import load_model
import os

# ref : DataFlair
# https://data-flair.training/blogs/face-mask-detection-with-python/
# Real-Time Face Mask Detector with Python, OpenCV, Keras

model=load_model("./mask_detection_model")

labels_dict={0:'without mask',1:'mask'}
color_dict={0:(0,0,255),1:(0,255,0)}

dirname = os.path.dirname(__file__)
f_haar_path = os.path.join(dirname, 'haarcascade_frontalface_default.xml') # path ของ haarcascade face

# ใช้ haarcascade classifier
face_detector = cv2.CascadeClassifier(f_haar_path)

img = cv2.imread('output_18_0.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_detector.detectMultiScale(gray, 1.3, 5)

for x, y, w, h in faces:
	face_img = new_img[y:x+h, x:x+w] # ดึงพิกัดใบหน้า
	resized = cv2.resize(face_img, (224, 224)) # ให้ภาพใบหน้า fit กับโมเดล (224,224)
	img_array = tf.keras.preprocessing.image.img_to_array(resized) # แปลงใบหน้าเป็น array
	img_array = tf.expand_dims(img_array, 0) #ขยายมิติภาพฟิตกับโมดล
	predictions = model.predict(img_array) # ทำนายบน ROI (Region of Interest)
	score = tf.nn.softmax(predictions[0]) # ผลลัพธ์
	label = np.argmax(score)

	# Post-Processing

	if label == 0:
		cv2.rectangle(new_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
		cv2.putText(new_img, "mask", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
	elif label == 1:
		cv2.rectangle(new_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
		cv2.putText(new_img, "No mask", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
	else:
		None

# pass
# แสดงผลหลังจากทำนาย
new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
cv2.imshow('liveimg', new_img)
# eval_js('showimg("{}")'.format(image2byte(new_img)))
print(np.argmax(score), 100 * np.max(score))

cv2.waitKey(0) 
cv2.destroyAllWindows() 