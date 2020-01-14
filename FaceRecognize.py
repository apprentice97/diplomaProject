import cv2
import numpy as np

images=[]
images.append(cv2.imread("C:\\Users\\19093\\Desktop\\test\\a1.tif", cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread("C:\\Users\\19093\\Desktop\\test\\a2.tif", cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread("C:\\Users\\19093\\Desktop\\test\\b1.tif", cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread("C:\\Users\\19093\\Desktop\\test\\b2.tif", cv2.IMREAD_GRAYSCALE))
labels=[0,0,1,1]
recongnizer = cv2.face.createLBPHFaceRecognizer()
recongnizer.train(images, np.array(labels))
predict_image = cv2.imread("C:\\Users\\19093\\Desktop\\test\\b3.tif", cv2.IMREAD_GRAYSCALE)
label,confidence = recongnizer.predict(predict_image)
print("label = ", label)
print("confidence = ", confidence)