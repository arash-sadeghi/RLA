import cv2 as cv
import numpy as np
import os
from termcolor import colored as c
print(c('[+] resizer with stryper capability','red'))
def goToScriptDir():
    ''' with this segment code is callable from any folder '''
    scriptLoc=__file__
    for i in range(len(scriptLoc)):
        # if '/' in scriptLoc[-i-2:-i]: # in running
        if '\\' in scriptLoc: char='\\'
        elif '/' in scriptLoc: char='/'
        else : raise NameError('[-] dir divider cahr error')
        
        if char in scriptLoc[-i-2:-i]: # in debuging

            scriptLoc=scriptLoc[0:-i-2]
            break
    print('[+] code path',scriptLoc)
    os.chdir(scriptLoc)
    ''' done '''
goToScriptDir()

thedir=os.getcwd()
all_folders=[ name for name in os.listdir(thedir) if os.path.isdir(os.path.join(thedir, name)) ]
for folder in all_folders:
    ims=os.listdir(folder)
    print('[+] processing ', folder)
    for imN in ims: 
        im=cv.imread(folder+'/'+imN)
        indx=np.arange(0,im.shape[1])%2==1
        im[0,indx]=255-im[0,indx]
        dim=(im.shape[1]*20,im.shape[0]*20)
        im=cv.resize(im,dim)
        cv.imwrite(folder+'/'+imN,im)

print('[+] have a nice day')




