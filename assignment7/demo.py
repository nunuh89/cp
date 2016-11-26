"""Harris Corner Detection"""

import numpy as np
import cv2

#Read image
img=cv2.imread("octagon.png")
#print "Read image from file; size:{}x{}". format(img.shape[1],img.shape[0] #[detector)
cv2.imshow("Image",img)

# Convert to grayscale
img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


orb = cv2.ORB()

kp1, des1 = orb.detctAndCompute(img1,None)
kp2, des2 = orb.detctAndCompute(img2,None)

img1_kp=cv2.drawKeypoints(img1,kp1, flags=cv2.DRAW_MATCHES_FLAGS
