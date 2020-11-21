import numpy as np
import matplotlib.pyplot as plt
import os
from time import time as TIME
from time import ctime
'''
this code will plot all .npy files in its own location
so do not put any .npy file in its location if you dont
want it to be plotted
'''

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

FinalTime=116000*5
pointN=FinalTime//(1000*5)
itNum=5
ticknum=11
samplingPeriod=10

allFiles=os.listdir()
# label=['BEECLUST','RL without comminucation']
palete=['r','b','g','purple','pink']
tobeDeleted=[]
for files in allFiles:
    # if os.path.splitext(files)[1]!='.npy' or not ('results' in os.path.splitext(files)[0]):
    if os.path.splitext(files)[1]!='.npy':

        tobeDeleted.append(files)

for i in tobeDeleted:
    allFiles.remove(i)

label=list(map(lambda x: os.path.splitext(x)[0],allFiles))

''' determining max data len and storing files in a list '''
datas=[]
datasLen=[]
for count,file_ in enumerate(allFiles):
    with open(file_,'rb') as _:
        datas.append(np.load(_))
        ''' storing number of datas for each iteration for each file '''
        datasLen.append(np.shape(datas[count])[-1]) 
 
for count,file_ in enumerate(allFiles):
    data=datas[count]
    oneIt=data[0,:,:]
    x=np.arange(0,np.shape(oneIt)[0])#*((samplingPeriod*datasLen[count])/len(averagedDataMean))

    plt.plot(x,oneIt[:,0],label='robot '+file_)

devisionScale=100
plt.yticks(fontsize=12)
plt.ylim(0,1)
plt.xlim(0,)
plt.legend(fontsize=12.5,loc=4)
plt.xlabel('Time [s] / '+str(devisionScale),fontsize=15,fontweight='bold')
plt.ylabel('epsilon parameter',fontsize=15,fontweight='bold')
plt.title('10 Robots with ',fontsize=15,fontweight='bold')
plt.grid()
plt.savefig(ctime(TIME()).replace(':','_')+'.png')
plt.show()
print('hi')