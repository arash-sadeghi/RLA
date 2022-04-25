from math import exp
import cv2 as cv
import numpy as np
def generateBackground(Lxpx,Lypx,R,visibleRaduis,QRloc,out_path,cue_loc):
    def gauss(x):
        a = 1.0 # amplititude of peak
        b = Lxpx/2.0 # center of peak
        c = Lxpx/11# standard deviation
        return a*exp(-((x-b)**2)/(2*(c**2)))
    im=np.zeros((Lxpx,Lypx))
    for i in range(0,R):
        cv.circle(im,cue_loc,i,gauss(Lxpx/2-i),2)

    #! fo getting the last circle outline
    cv.circle(im,cue_loc,i,1,2)



    for i in QRloc.values():
        cv.circle(im,(i[1] , i[0]),10,(255,255,255),-1)
        # cv.circle(im,tuple(i),visibleRaduis,(255,255,255),1)
    im=255-255*im
    im=cv.flip(im, 1)
    '''writing and reading back the image to have a 3 channel image with pixels between 0-255'''
    cv.imwrite(out_path,im)
    im=cv.imread(out_path)
    im=cv.rectangle(im,(0,0),(Lxpx,Lypx),(0,255,0),3)
    return im
if __name__ == "__main__":
    out_path="data/Background.png"
    QRloc={'QR1': (512, 256), 'QR2': (512, 512), 'QR3': (512, 768), 'QR4': (0, 768), 'QR5': (0, 512), 'QR6': (0, 256)}
    Lxpx=512
    # Lxpx=int(2.82*512/2)
    Lypy=1024
    # Lypy=int(5.65*512/2)
    cue_radius=179
    # cue_radius=int(1*512/2)

    cue_loc=( Lypy//4*3 , Lxpx//2 )
    generateBackground(Lxpx,Lypy, cue_radius , 76,QRloc,out_path,cue_loc)
