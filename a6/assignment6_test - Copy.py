import sys
import os
import numpy as np
import cv2
from scipy.stats import norm
from scipy.signal import convolve2d
import math

import assignment6

def viz_gauss_pyramid(pyramid):
  """ This function creates a single image out of the given pyramid.
  """
  height = pyramid[0].shape[0]
  width = pyramid[0].shape[1]

  out = np.zeros((height*len(pyramid), width), dtype = float)

  for idx, layer in enumerate(pyramid):
    if layer.max() <= 1:
      layer = layer.copy() * 255

    out[(idx*height):((idx+1)*height),:] = cv2.resize(layer, (width, height), 
        interpolation = 3)

  return out.astype(np.uint8)

def viz_lapl_pyramid(pyramid):
  """ This function creates a single image out of the given pyramid.
  """
  height = pyramid[0].shape[0]
  width = pyramid[0].shape[1]

  out = np.zeros((height*len(pyramid), width), dtype = np.uint8)

  for idx, layer in enumerate(pyramid[:-1]):
     # We use 3 for interpolation which is cv2.INTER_AREA. Using a value is
     # safer for compatibility issues in different versions of OpenCV.
     patch = cv2.resize(layer, (width, height),
         interpolation = 3).astype(float)
     # scale patch to 0:256 range.
     patch = 128 + 127*patch/(np.abs(patch).max())

     out[(idx*height):((idx+1)*height),:] = patch.astype(np.uint8)

  #special case for the last layer, which is simply the remaining image.
  patch = cv2.resize(pyramid[-1], (width, height), 
      interpolation = 3)
  out[((len(pyramid)-1)*height):(len(pyramid)*height),:] = patch

  return out

def run_blend(black_image, white_image, mask):
  """ This function administrates the blending of the two images according to 
  mask.

  Assume all images are float dtype, and return a float dtype.
  """

  # Automatically figure out the size
  min_size = min(black_image.shape)
  depth = int(math.floor(math.log(min_size, 2))) - 4 # at least 16x16 at the highest level.

  gauss_pyr_mask = assignment6.gaussPyramid(mask, depth)
  gauss_pyr_black = assignment6.gaussPyramid(black_image, depth)
  gauss_pyr_white = assignment6.gaussPyramid(white_image, depth)


  lapl_pyr_black  = assignment6.laplPyramid(gauss_pyr_black)
  lapl_pyr_white = assignment6.laplPyramid(gauss_pyr_white)

  outpyr = assignment6.blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask)
  outimg = assignment6.collapse(outpyr)

  outimg[outimg < 0] = 0 # blending sometimes results in slightly out of bound numbers.
  outimg[outimg > 255] = 255
  outimg = outimg.astype(np.uint8)

  return lapl_pyr_black, lapl_pyr_white, gauss_pyr_black, gauss_pyr_white, \
      gauss_pyr_mask, outpyr, outimg



black_img = cv2.imread('black.jpg')
white_img = cv2.imread('white.jpg')
mask_img = cv2.imread('mask.jpg')
  
black_img = black_img.astype(float)
white_img = white_img.astype(float)
mask_img = mask_img.astype(float) / 255

print "Applying blending."
lapl_pyr_black_layers = []
lapl_pyr_white_layers = []
gauss_pyr_black_layers = []
gauss_pyr_white_layers = []
gauss_pyr_mask_layers = []
out_pyr_layers = []
out_layers = []


for channel in range(3):
  lapl_pyr_black, lapl_pyr_white, gauss_pyr_black, gauss_pyr_white, gauss_pyr_mask,\
      outpyr, outimg = run_blend(black_img[:,:,channel], white_img[:,:,channel], \
                       mask_img[:,:,channel])
      
lapl_pyr_black_layers.append(viz_lapl_pyramid(lapl_pyr_black))
lapl_pyr_white_layers.append(viz_lapl_pyramid(lapl_pyr_white))
gauss_pyr_black_layers.append(viz_gauss_pyramid(gauss_pyr_black))
gauss_pyr_white_layers.append(viz_gauss_pyramid(gauss_pyr_white))
gauss_pyr_mask_layers.append(viz_gauss_pyramid(gauss_pyr_mask))
out_pyr_layers.append(viz_lapl_pyramid(outpyr))
out_layers.append(outimg)
    
lapl_pyr_black_img = cv2.merge(lapl_pyr_black_layers)
lapl_pyr_white_img = cv2.merge(lapl_pyr_white_layers)
gauss_pyr_black_img = cv2.merge(gauss_pyr_black_layers)
gauss_pyr_white_img = cv2.merge(gauss_pyr_white_layers)
gauss_pyr_mask_img = cv2.merge(gauss_pyr_mask_layers)
outpyr = cv2.merge(out_pyr_layers)
outimg = cv2.merge(out_layers)

#print "Writing images to folder {}".format(os.path.join(outfolder, setname))
#cv2.imwrite(os.path.join(outfolder, setname + '_lapl_pyr_black' + ext),
#            lapl_pyr_black_img)
cv2.imwrite('lapl_pyr_black_img.jpg',lapl_pyr_black_img)
#cv2.imwrite(os.path.join(outfolder, setname + '_lapl_pyr_white' + ext),
#            lapl_pyr_white_img)
cv2.imwrite('lapl_pyr_white_img.jpg',lapl_pyr_white_img)
#cv2.imwrite(os.path.join(outfolder, setname + '_gauss_pyr_black' + ext),
#            gauss_pyr_black_img)

cv2.imwrite('gauss_pyr_black_img.jpg',gauss_pyr_black_img)
#cv2.imwrite(os.path.join(outfolder, setname + '_gauss_pyr_white' + ext),
#            gauss_pyr_white_img)
cv2.imwrite('gauss_pyr_white_img.jpg',gauss_pyr_white_img)
#cv2.imwrite(os.path.join(outfolder, setname + '_gauss_pyr_mask' + ext),
#            gauss_pyr_mask_img)
cv2.imwrite('gauss_pyr_mask_img.jpg',gauss_pyr_mask_img)
#cv2.imwrite(os.path.join(outfolder, setname + '_outpyr' + ext),
#            outpyr)
cv2.imwrite('outpyr.jpg',outpyr)
#cv2.imwrite(os.path.join(outfolder, setname + '_outimg' + ext),
#            outimg)
cv2.imwrite('outimg.jpg',outimg)
