import vlfeat
import cv2
import numpy as np

img = cv2.imread("../../../data/spots.jpg",0)

# Invert image to have bright MSER regions
img = 255 - img

r,f = vlfeat.vl_mser(img, min_diversity=0.7,max_variation=0.2,delta=10)

img_color = cv2.cvtColor( img, cv2.cv.CV_GRAY2RGB )
M = np.zeros(img.shape)
for x in r:
    s = vlfeat.vl_erfill(img,x)
    print s
    points = np.unravel_index(s,img.shape, order='F')
    print points
    points = np.array(map(lambda
                          (x,y):[[y,x]],zip(points[0],points[1])),dtype='int32')
    if len(points) < 5:
        continue
    M[np.unravel_index(s,img.shape, order='F')] += 1
    ellipse = cv2.fitEllipse(points)
    cv2.ellipse( img_color, ellipse, (0,0,255))

cv2.imwrite("ellipses.png",img_color)

M = M.astype('float')
M /= M.max()
M *= 255

cv2.imwrite("mser.png",M)
