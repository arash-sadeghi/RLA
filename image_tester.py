import cv2 as cv
import numpy as np
from math import cos , sin
import matplotlib.pyplot as plt
ground=cv.imread('BackgroundGeneratedBySim.png')
ground2=np.copy(ground)
Xlen=np.shape(ground)[0]  
Ylen=np.shape(ground)[1]
QRloc={'QR1':(Ylen,Xlen//4),'QR2':(Ylen,Xlen//4*2),'QR3':(Ylen,Xlen//4*3),'QR4':(0,Xlen//4*3),'QR5':(0,Xlen//4*2),'QR6':(0,Xlen//4)}

''' actions '''
length=445;angle=90

actionXY=np.array([length*sin(np.radians(angle)),length*cos(np.radians(angle))])
QR=np.array(QRloc['QR5']) # QR
des=actionXY+QR
des=des.astype(int)


rad=10
pod=np.flip(des) # position of desired point
''' getting locations which lie inside circle '''
Xs=np.arange(pod[0]-rad,pod[0]+rad+1)
Ys=np.arange(pod[1]-rad,pod[1]+rad+1)
ground=cv.circle(ground,(des[1],des[0]),rad,(225,0,0))

all_vals=255-ground2[Xs,Ys]
all_vals=all_vals[:,0]
print(all_vals)
# cv.imshow("ground",ground);cv.waitKey()
plt.imshow(ground)
plt.show()
print('hi')