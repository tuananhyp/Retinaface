import cv2 

image = cv2.imread("/home/ubutu/1.jpg")

image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("/home/ubutu/1_gray.jpg", image_cvt)