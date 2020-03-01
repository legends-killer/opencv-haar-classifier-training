import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('./temp/cascade.xml')

img = cv2.imread('./images/t2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(gray, 10, 5, cv2.CASCADE_SCALE_IMAGE, (150,150), (500,500))

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
