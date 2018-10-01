import cv2
import numpy as np
from matplotlib import pyplot as plt

original_image = cv2.imread("../data/data/simpsons.jpg")
original_image_to_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
hsv_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

roi = cv2.imread("../data/data/barts_face.jpg",0)
w, h = roi.shape[::-1]
# hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# hue, saturation, value = cv2.split(hsv_roi)


res = cv2.matchTemplate(original_image_to_gray, roi, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(original_image, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)


# Histogram ROI
# roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
# mask = cv2.calcBackProject([hsv_original], [0, 1], roi_hist, [0, 180, 0, 256], 1)

# Filtering remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# mask = cv2.filter2D(mask, -1, kernel)
# _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

# mask = cv2.merge((mask, mask, mask))
# result = cv2.bitwise_and(original_image, mask)
# cv2.imshow("Result", result)
cv2.imshow("Roi", roi)
cv2.imwrite('simpsons.png',original_image)
cv2.imshow("Après detection", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()