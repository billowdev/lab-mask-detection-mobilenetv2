"""
pip install keras
pip install tensorflow

"""

import cv2
import numpy as np
from keras.models import load_model
import os
import tensorflow as tf

model=load_model("./mask_detection_model")
img = cv2.imread('output_18_0.png')
resized = cv2.resize(img, (224, 224)) # ให้ภาพใบหน้า fit กับโมเดล (224,224)
img_array = tf.keras.preprocessing.image.img_to_array(resized) # แปลงใบหน้าเป็น array
img_array = tf.expand_dims(img_array, 0) #ขยายมิติภาพฟิตกับโมดล

# print(model.predict(img_array))

predictions = model.predict(img_array) # ทำนายบน ROI (Region of Interest)
score = tf.nn.softmax(predictions[0]) # ผลลัพธ์
label = np.argmax(score)
print("-----\n")
# print(score)
# print(label)
if label == 0:
	cv2.imshow('Mask Image', img)
	print("\n\n Mask")
elif label == 1:
	cv2.imshow('No Mask Image', img)
	print("\n\n No mask")
else:
	None

cv2.waitKey(0) 
cv2.destroyAllWindows() 
