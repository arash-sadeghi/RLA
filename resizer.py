import cv2 as cv
import numpy as np
import os
thedir=os.getcwd()
all_folders=[ name for name in os.listdir(thedir) if os.path.isdir(os.path.join(thedir, name)) ]
for folder in all_folders:
    ims=os.listdir(folder)
    print('[+] processing ', folder)
    for imN in ims: 
        im=cv.imread(folder+'/'+imN)
        dim=(im.shape[1]*20,im.shape[0]*20)
        im=cv.resize(im,dim)
        cv.imwrite(folder+'/'+imN,im)
        # cv.imshow('x',x)
        # cv.waitKey()
        # cv.destroyAllWindows()
print('[+] have a nice day')




