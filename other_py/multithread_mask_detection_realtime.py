import tensorflow as tf
import cv2
import time
import numpy as np
from keras.models import load_model
#
import playsound
#
model=load_model("./mask_detection_model")
results={0:'without mask',1:'mask'}

GR_dict={0:(0,0,255),1:(0,255,0)}

rect_size = 4
cap = cv2.VideoCapture(0) 
haarcascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

import concurrent.futures


def play_sound(label):
	try:
		if label == 0:
			playsound.playsound('wearmask.mp3', True)
		elif label == 1:
			playsound.playsound('thxwarmask.mp3', True)
		else:
			pass
		time.sleep(2)
	except:
		time.sleep(4)
		pass

while True:
	(rval, im) = cap.read()
	im=cv2.flip(im,1,1) 
	
	rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
	faces = haarcascade.detectMultiScale(rerect_size)
	for f in faces:
		(x, y, w, h) = [v * rect_size for v in f] 
		
		face_img = im[y:y+h, x:x+w]
		rerect_sized = cv2.resize(face_img,(224,224))
		normalized=rerect_sized/255.0
		reshaped=np.reshape(normalized,(1,224,224,3))
		reshaped = np.vstack([reshaped])
		result = model.predict(reshaped)
		score = tf.nn.softmax(result[0]) # ผลลัพธ์

		res = np.argmax(score)

		with concurrent.futures.ThreadPoolExecutor() as executor:
			future = executor.submit(play_sound, res)
			future.result()

		label=np.argmax(result,axis=1)[0]

		cv2.rectangle(im,(x,y),(x+w,y+h),GR_dict[label],2)
		cv2.rectangle(im,(x,y-40),(x+w,y),GR_dict[label],-1)

		cv2.putText(im, results[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

	cv2.imshow('LIVE',   im)
	key = cv2.waitKey(1)
	if key == ord('q'): 
		break


cap.release()
cv2.destroyAllWindows()
