import numpy as np
import os
def DirLocManage(returnchar=False):
    ''' with this segment code is callable from any folder '''
    if os.name=='nt':
        dirChangeCharacter='\\'
    else:
        dirChangeCharacter='/'
    if returnchar==False:
        scriptLoc=__file__
        for i in range(len(scriptLoc)):
            # if '/' in scriptLoc[-i-2:-i]: # in running
            if dirChangeCharacter in scriptLoc[-i-2:-i]: # in debuging
                scriptLoc=scriptLoc[0:-i-2]
                break
        # print('[+] code path',scriptLoc)
        os.chdir(scriptLoc)
    return dirChangeCharacter
    ''' done '''
DirLocManage()

data=np.load("SAR_robot0.npy")

e=[]
for cc,vv in enumerate(data):
    sample=vv
    for c,v in enumerate(data[cc:]):
        if np.all(v[0:2]==sample[0:2]):
            print(v,sample)
            if v[2]!=sample[2]:
                e.append([v,sample])

    if np.size(np.unique(e))>1:
        print(e,"-----------------\n")
print("hi")