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
differences=[]
for cc,vv in enumerate(data):
    sample=vv
    for c,v in enumerate(data[cc:]):
        if np.all(v[1:3]==sample[1:3]): # 0 is for time
            # print(v,sample)
            if v[3]!=sample[3]:
                e.append([v,sample])

    if np.size(np.unique(e))>1:
        E=np.vstack(e)
        E=E[:,3]
        l=np.array([int(np.min(E)),int(np.mean(E)),int(np.max(E))])
        if l[0]==0:
            print('*x*')
        differences.append(l)
        # print(l,"-----------------")
        ''' for 6 6: min 124 max 136 mean 132 '''
    e=[]
differences=np.unique(np.array(differences),axis=0)
with open('MinMeanMaxdifferences_.npy','wb') as f:
    np.save(f,differences)
np.savetxt('MinMeanMaxdifferences_.csv',differences,delimiter=',')
print("hi")