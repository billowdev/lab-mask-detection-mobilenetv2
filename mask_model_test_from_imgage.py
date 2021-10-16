import tensorflow as tf
from keras.models import load_model
import numpy as np
import cv2
import os

dirname = os.path.dirname(__file__)
f_haar_path = os.path.join(dirname, 'haarcascade_frontalface_default.xml') # path ของ haarcascade face
face_detector = cv2.CascadeClassifier(f_haar_path)

model=load_model("./mask_detection_model")

def Detection(img):
	new_img = cv2.resize(img, (img.shape[1] // 1, img.shape[0] // 1)) # resize ขนาดภาพเพื่อให้ง่ายต่อการตรวจจับ
	faces = face_detector.detectMultiScale(img) # สำหรับตรวจจับใบหน้า โดย 
	for x, y, w, h in faces: # วนซ้ำพิกัดบนใบหน้า
		face_img = new_img[y:x+h, x:x+w] # ดึงพิกัดใบหน้า
		resized = cv2.resize(face_img, (224, 224)) # ให้ภาพใบหน้า fit กับโมเดล (224,224)
		img_array = tf.keras.preprocessing.image.img_to_array(resized) # แปลงใบหน้าเป็น array
		img_array = tf.expand_dims(img_array, 0) #ขยายมิติภาพฟิตกับโมดล
		predictions = model.predict(img_array) # ทำนายบน ROI (Region of Interest)
		score = tf.nn.softmax(predictions[0]) # ผลลัพธ์
		label = np.argmax(score) # หาค่าสูงสุด

		new_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) # วาดสี่เหลี่ยมรอบใบหน้า

		if label == 0:
			new_img = cv2.putText(new_img, "mask", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # แสดงข้อความ "mask"
			cv2.imshow('Mask Image', new_img)
			print("\n\n mask")

		if label == 1:
			new_img = cv2.putText(new_img, "No mask", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # แสดงข้อความ "mask"
			cv2.imshow('No Mask Image', new_img)
			print("\n\n No mask")

image = cv2.imread('with_mask_1022.jpg')   # mask
image2 = cv2.imread('output_17_0.png')   # nomask
Detection(image)
Detection(image2)

# cv2.imshow("image", new_img)
cv2.waitKey(0) 
cv2.destroyAllWindows() 
