import cv2 as cv
import os
import numpy as np

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

file_name='QtableRob0'
FPS=10
''' im dimention is 1024*512 this is what shape returns 
but its inverse should be passed to the size of video '''
# size=(1024, 512)
size=(800,140)
# size=(140,800)

fourcc = cv.VideoWriter_fourcc(*'mp4v')

video = cv.VideoWriter('alpha 1.mp4',fourcc, FPS, size,True)

allFiles=os.listdir(file_name)
tobeDeleted=[]


for files in allFiles:
    if os.path.splitext(files)[1]!='.png':
        tobeDeleted.append(files)
for i in tobeDeleted:
    allFiles.remove(i)


''' inside all files type is string so cannot sort properly
so delete .png and sort as integer '''
for c,v in enumerate(allFiles):
    allFiles[c]=allFiles[c][0:-4]
allFiles.sort(key=int)

def enhance(im):
    im=im.astype('int32')
    im=255-im
    im*=1
    if np.any(im>255):
        pass
        # print('hi')
    im[im>255]=255
    im=255-im
    im=im.astype('uint8')
    return im

for im in allFiles:
    # os.remove(file_name+'/'+im+'.png')
    imMat=cv.imread(file_name+'/'+im+'.png')
    imMat=enhance(imMat) # for tables only
    video.write(cv.resize(imMat,size))
    per=allFiles.index(im)/len(allFiles)*100
    if per%4==0 and per>1:
        print('done percentage: ',per,'%')

video.release()
print('done for ',file_name)

