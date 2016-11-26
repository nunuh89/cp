# ASSIGNMENT 5
# Your Name

import numpy as np
import scipy as sp
import scipy.signal
import cv2


x = np.array([[0, 1], [2, 3], [4, 5]])
h,w=x.shape
x1=np.zeros(shape=(2*h,2*w))

for i in range(0,h):
 for j in range(0,w):
   x1[2*i,2*j]=x[i,j]
