import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import gridspec as gs

#load an image
img = cv.imread('../data/data/dog.jpg')
#image comme grayscale

img_single_chanel = cv.imread('../data/data/dog.jpg',0)

# print some details about the images
print('The shape of img without second arg is: {}'.format(img.shape))
print('The shape of img_single_channel is:     {}'.format(img_single_chanel.shape))

cv.imshow('OpenCv imshow(',img)
cv.waitKey(0)
cv.destroyAllWindows()

"""
Partie suivante du TP

"""

height,windth,channels = img.shape[:3]
print("Image height {} , Image Windth  {}, number of chanels : {}".format(height,windth,channels))
array_blue = img[:,:,0]
array_vert = img[:,:,1]
array_rouge = img[:,:,2]
"""
    Analyse par histograme d'une ligne de l'image avec deux points de repères

"""
cv.imshow('in blue',array_blue)
cv.imshow('in green',array_vert)
cv.imshow('in red',array_rouge)
cv.waitKey(0)
cv.destroyAllWindows()
gs = gs.GridSpec(1,2)
ax0 = plt.subplot(gs[0])
ax0.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
ax0.axhline(50 , color='black')
ax0.axvline(100 , color='k' , linewidth=2) , ax0.axvline(255, color = 'k' , linewidth=2)

# imageslice
ax1 = plt.subplot(gs[1])
ax1.plot(array_blue[49, :], color='blue')
ax1.plot(array_vert[49, :], color='green')
ax1.plot(array_rouge[49, :], color='red')
ax1.axvline(100, color='k', linewidth=2), ax1.axvline(225, color='k', linewidth=2)
plt.suptitle('Examen des valeurs du plan de couleur pour une seule ligne')
plt.show()

"""
    Rognage d'une image
"""

cropped = img[109:310,9:160]
cv.imshow("Image rognée en 310 , 160 ",img)
cv.waitKey(0)
cv.destroyAllWindows()

cap = cv.VideoCapture("../data/data/bike.avi")
while(1):
    frame = cap.read()

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    mask = cv.inRange(hsv,lower_blue,upper_blue)

    res = cv.bitwise_and(frame,frame,mask=mask)

    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
    cv.destroyAllWindows()


