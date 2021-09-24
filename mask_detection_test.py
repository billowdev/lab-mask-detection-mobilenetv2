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

model=load_model("./mask_detection_model.h5")

labels_dict={0:'without mask',1:'mask'}
color_dict={0:(0,0,255),1:(0,255,0)}

size = 4
webcam = cv2.VideoCapture(0) #Use camera 0


f_haar_path = os.path.join(dirname, 'haarcascade_frontalface_default.xml') # path ของ haarcascade face
# ใช้ haarcascade classifier
classifier = cv2.CascadeClassifier(f_haar_path)

while True:
	(rval, img) = webcam.read()
	img=cv2.flip(img,1,1) # สลับเพื่อให้เหมือนส่งกระจก

	# ลดขนาดภาพเพื่อความเร็ว
	mini = cv2.resize(img, (img.shape[1] // size, img.shape[0] // size))

	# สำหรับตรวจจับใบหน้า
	faces = classifier.detectMultiScale(mini)

	# วาดสี่เหลี่ยมรอบใบหน้า
	for f in faces:
		(x, y, w, h) = [v * size for v in f] # Scale the shapesize backup
		#Save just the rectangle faces in SubRecFaces
		face_imgg = img[y:y+h, x:x+w]
		resized = cv2.resize(face_img,(150,150))
		normalized = resized/255.0
		reshaped = np.reshape(normalized,(1,150,150,3))
		reshaped = np.vstack([reshaped])
		result = model.predict(reshaped)
		# ผลลัพธ์
		print(result)
		
		label=np.argmax(result,axis=1)[0]
		cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
		cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
		cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
		
	# แสดงภาพ
	cv2.imshow('liveimg',   img)
	key = cv2.waitKey(10)

	# if Esc key is press then break out of the loop 

	# สำหรับหยุดการรัน
	if cv2.waitKey(1) == ord('q'):
		break
		
# ปิดวิดีโอ
webcam.release()
# ปิดหน้าต่าง
cv2.destroyAllWindows()