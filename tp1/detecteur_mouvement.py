# import the necessary packages
from imutils.video import VideoStream
import imutils
import cv2
import threading
from termcolor import colored


vs = VideoStream(src=0).start()
mouvementDetecte = False


def reset_Timer(cnts):
    mouvementDetecte=False
    cnts = None

def sendAlert():
    print(colored("Attention un mouvement a été detecté",'red'))

# initialize the first frame in the video stream
firstFrame = None
# loop over the frames of the video
while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    frame = vs.read()

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if frame is None:
        break

    # convertion pour comparaison en "mode gris"
    frame = imutils.resize(frame, width=750)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Si la permière image est vide on initialise
    if firstFrame is None:
        firstFrame = gray
        continue
    #différence absolue entre le le record et le "monde gris"
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=4)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 700:
            continue

        #contour de l'image
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255),1)
        mouvementDetecte = True
        #Toutes les n secondes la detection est reinitialisée
        timer = threading.Timer(2, reset_Timer(cnts))
        if mouvementDetecte :
            alert_timer = threading.Timer(10,sendAlert())

    cv2.imshow("Camera de sécurité", frame)

    # cv2.imshow("Thresh", thresh)
    # cv2.imshow("Monde Gris vue", frameDelta)
    key = cv2.waitKey(1) & 0xFF

    # Touche q pour quitter
    if key == ord("q"):
        break

vs.stop()
cv2.destroyAllWindows()