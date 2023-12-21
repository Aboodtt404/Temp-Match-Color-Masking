import cv2
import numpy as np

image = cv2.imread('./Images/purple-flower.jpg')

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_bound = np.array([120, 50, 50])
upper_bound = np.array([150, 255, 255])

mask = cv2.inRange(hsv, lower_bound, upper_bound)

kernel = np.ones((5, 5), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=1)
mask = cv2.erode(mask, kernel, iterations=1)

result = cv2.bitwise_and(image, image, mask=mask)

cv2.imshow('Image', image)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
