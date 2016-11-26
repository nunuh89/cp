import cv2
import numpy as np,sys

A = cv2.imread('1.jpg')
B = cv2.imread('2.jpg')
C = cv2.imread('3.jpg')
D = cv2.imread('4.jpg')

matrixnumber=4

# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in xrange(matrixnumber):
    G = cv2.pyrDown(G)
    gpA.append(G)

# generate Gaussian pyramid for B
G = C.copy()
gpC = [G]
for i in xrange(matrixnumber):
    G = cv2.pyrDown(G)
    gpC.append(G)
    
# generate Gaussian pyramid for C
G = D.copy()
gpD = [G]
for i in xrange(matrixnumber):
    G = cv2.pyrDown(G)
    gpD.append(G)
    
# generate Gaussian pyramid for D
G = B.copy()
gpB = [G]
for i in xrange(matrixnumber):
    G = cv2.pyrDown(G)
    gpB.append(G)
    
# generate Laplacian Pyramid for A
lpA = [gpA[matrixnumber-1]]
for i in xrange(matrixnumber-1,0,-1):
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i-1],GE)
    lpA.append(L)

# generate Laplacian Pyramid for B
lpB = [gpB[matrixnumber-1]]
for i in xrange(matrixnumber-1,0,-1):
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i-1],GE)
    lpB.append(L)

# generate Laplacian Pyramid for C
lpC = [gpC[matrixnumber-1]]
for i in xrange(matrixnumber-1,0,-1):
    GE = cv2.pyrUp(gpC[i])
    L = cv2.subtract(gpC[i-1],GE)
    lpC.append(L)

# generate Laplacian Pyramid for D
lpD = [gpD[matrixnumber-1]]
for i in xrange(matrixnumber-1,0,-1):
    GE = cv2.pyrUp(gpD[i])
    L = cv2.subtract(gpD[i-1],GE)
    lpD.append(L)
    
# Now add left and right halves of images in each level
LS = []
for la,lb,lc,ld in zip(lpA,lpB,lpC,lpD):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:cols/4], lb[:,cols/4:cols/2],lc[:,cols/2:cols*3/4],ld[:,cols*3/4:]))
    LS.append(ls)

# now reconstruct
ls_ = LS[0]
for i in xrange(1,matrixnumber):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])

# image with direct connecting each half
#real = np.hstack((A[:,:cols/2],B[:,cols/2:]))

cv2.imwrite('Pyramid_blending2.jpg',ls_)
#cv2.imwrite('Direct_blending.jpg',real)
