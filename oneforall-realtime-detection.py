"""
refernce:
https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/

pip install tensorflow 
pip install opencv-python
"""


import tensorflow.keras	 # นำเข้า ไลบรารี่ เพื่อใช้ ในการโหลดโมเดลและ ทำนาย
from PIL import Image, ImageOps 
# PIL.อิมเมจ fromarray(obj, mode=None)[แหล่งที่มา]
#  สร้างหน่วยความจำรูปภาพจากวัตถุที่ส่งออกอินเทอร์เฟซอาร์เรย์ 
#  (โดยใช้โปรโตคอลบัฟเฟอร์) ถ้า obj ไม่อยู่ติดกัน 
#  จะเรียกเมธอด tobytes และใช้ frombuffer()
import numpy as np # เพื่อใช้ในการแปลงภาพปกติ เป็น array Image เพื่อทำการ pass paramiter เข้าไปในการทำนายโมเดล
import cv2 # ใช้ในการ โชว์รูปภาพ และ อ่านรูปภาพ เพื่อใช้ในการทำนาย
import tensorflow as tf # เป็นตัวช่วยในการ หาค่า max ของ ผลลัพธ์ที่ได้จากการทำนาย

# เรียก ใช้ แคสเคด คลาสชิฟายเพื่อใช้ในการช่วยดีเทคใบหน้า
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml') 
np.set_printoptions(suppress=True)
# โมเดลที่ได้จากการ เทรนโดยใช้โปรแกรม Teachable Machine บนเว็๋บไซต์ ซึ่งพัฒนาโดย Google
model = tensorflow.keras.models.load_model('teachablemachine_keras/keras_model.h5')
# สร้างตัวแปรเพื่อเก็บขนาดภาพ 224x224
size = (224, 224)
# cap สำหรับกล้องวิดีโอ
cap = cv2.VideoCapture(0)
# รันจนกว่าจะกด q หรือ ปิดโปรแกรมไป
while True:
	success, image_bgr = cap.read() # อ่านภาพจากกล้อง
	image_bw = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) # แปลงเป็นภาพ ไบนารี่
	# แปลงภาพเป็น RGB ซึ่ง opencv จะอ่านภาพเข้ามาในปริภูมิที่เป็น BGR เราต้องแปลงเป็น ปริภูมิRGB1
	image_rgb = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB) 
	faces = face_classifier.detectMultiScale(image_bw) # พาสค่าพารามิเตอร์รูปภาพ ไบนารี่เข้าไปใน เฟสดีเทคเพื่อใช้ในการหาพิกัดใบหน้า
	for (x, y, w, h) in faces: # ทำการหาพิกัด ใบหน้าโดยการวนซ้ำ
		face_rgb = Image.fromarray(image_rgb[y:y+h, x:x+w]) # รูปใบหน้า rgb
		data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32) # ข้อมูลภาพ ใบหน้า ทำการฟิตเป็นขนาด 224x224
		image = face_rgb # แทนค่าภาพเป็น ตัวแปร image
		image = ImageOps.fit(image, size, Image.ANTIALIAS) # ครอปใบหน้า โดย ฟิตภาพ เข้ากับ ตัวแปรขนาดภาพที่ตั้งไว้เมื่อตอนต้นคือ size  
		im_array = np.asarray(image) # แปลงภาพ ใบหน้าเป็น ภาพ array
		nomalized_image_array = (im_array.astype(np.float32) / 127.0) - 1 # ทำการ นอมอไล อิมเมจ
		# วิธี astype() ใช้เพื่อส่งวัตถุแพนด้าไปยัง dtype ที่ระบุ ฟังก์ชัน astype() 
		# ยังให้ความสามารถในการแปลงคอลัมน์ที่มีอยู่ที่เหมาะสมเป็นประเภทหมวดหมู่ 
		# เสริมความแข็งแกร่งให้กับรากฐานของคุณด้วย Python Programming Foundation Course และเรียนรู้พื้นฐาน
		data[0] = nomalized_image_array # เลือกมา index ที่ 0
		pred = model.predict(data) # ทำการทำนาย ภาพ โดย ใส่ค่า รูปภาพ ใบหน้าที่ผ่านกระบวนการที่กล่าวมาข้างต้น เพื่อทำการทำนาย
		score = tf.nn.softmax(pred[0]) # ทำการ 
		# ทีเอฟ นน. softmax สร้างเพียงผลลัพธ์ของการใช้ฟังก์ชัน 
		# softmax กับเทนเซอร์อินพุต softmax "squishes" อินพุตเพื่อให้ sum(input) = 1 
		# เป็นวิธีการทำให้เป็นมาตรฐาน รูปร่างของเอาต์พุตของ softmax เหมือนกับอินพุต: เพียงแค่ทำให้ค่าเป็นมาตรฐาน
		label = np.argmax(score) # ข้อมูล label รูปภาพ ซึ่ง 0 = สวมแมสก์ 1 = ไม่สวมแมสก์
		print(label) # แสดงค่า label 

		if pred[0][0] > pred[0][1]:
			# แสดงข้อความบน box ว่า With_Mask
			cv2.putText(image_bgr, 'With_Mask', (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			# วาดสี่เหลี่ยมรอบใบหน้า
			cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
		else:
			# แสดงข้อความบน box ว่า No Mask
			cv2.putText(image_bgr, 'No_mask', (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			# วาดสี่เหลี่ยมรอบใบหน้า
			cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 0, 255), 2)

	# ใช้ cv2 ในการ โชว์ figure ที่ทำการ Live ดีเทคเรียลไทม์
	cv2.imshow('LIVE',   image_bgr) 
	key = cv2.waitKey(1) # รอ user ทำการ กด q เพื่อปิด โปรแกรม
	if key == ord('q'): 
		break

cap.release() # ทำการปิดกล้อง
cv2.destroyAllWindows() # ทำการปิด figure