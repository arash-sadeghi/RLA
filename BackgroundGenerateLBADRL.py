import numpy as np
import cv2 as cv
from math import exp,sqrt
m2pix=512/2
ROBN=20
im_size_x=int(2*m2pix)
im_size_y=im_size_x*2
R=int(0.5*m2pix)
def gauss(x):
    # a = 512 # amplititude of peak
    a = 1.0 # amplititude of peak

    b = im_size_x/2.0 # center of peak
    c = im_size_x/10# standard deviation
    # print (">>",a*exp(-((x-b)**2)/(2*(c**2))))
    return a*exp(-((x-b)**2)/(2*(c**2)))
im=np.zeros((im_size_x,im_size_y))
for i in range(0,R):
    cv.circle(im,(int((im_size_y/4)),int(im_size_x/2)),i,gauss(im_size_x/2-i),2)
#------------------------------------------------------------------------------------------------------
im=cv.rotate(im, cv.ROTATE_90_CLOCKWISE)
Lx=2
Ly=2*Lx
m2pix=512//2
Lxpx=Lx*m2pix
Lypx=2*Lxpx
QRlocs=[(Lxpx,Lypx//4),(Lxpx,Lypx//2),(Lxpx,3*Lypx//4),(0,3*Lypx//4),(0,Lypx//2),(0,Lypx//4)]

for i in QRlocs:
    cv.circle(im,i,10,(255,255,255),-1)
    cv.circle(im,i,int(0.3*m2pix),(255,255,255),1)

#------------------------------------------------------------------------------------------------------

# cv.imshow("im",im)
# cv.waitKey()
for i in range(0,im_size_y):
    for j in range(0,im_size_x):
        im[i,j]=im[i,j]*255
        im[i,j]=255-im[i,j]
cv.imwrite("ground.png",im)