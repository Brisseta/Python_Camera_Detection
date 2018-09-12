import cv2
from matplotlib.pyplot import *

img1 = cv2.imread("../data/data/dog.jpg")
img2 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
figure(figsize=(4,4))
imshow(img2)

red, green, blue = cv2.split(img2)
figure("séparation des couleurs",figsize=(12, 4))
subplot(131)
imshow(red, cmap=cm.gray)
subplot(132)
imshow(green, cmap=cm.gray)
subplot(133)
imshow(blue, cmap=cm.gray)
waitforbuttonpress()
