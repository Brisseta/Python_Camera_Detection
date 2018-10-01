# template matching
import cv2
import numpy as np
# reading in the image
img_rgb = []
img_rgb[0] = cv2.imread('../data/data/waves.jpg')
img_rgb[1] = cv2.imread('../data/data/waves.jpg')
img_rgb[2] = cv2.imread('../data/data/waves.jpg')
img_rgb[3] = cv2.imread('../data/data/waves.jpg')
# converting to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# reading the template to be matched.
template = cv2.imread('../data/data/waves.jpg',0)
w, h = template.shape[::-1]
# threshold option, where if something is maybe an 80% match, then we say it's a match.
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
# marking all the matches on the original image, using the cordinates found in gray image.
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
cv2.imshow('Detected',img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()