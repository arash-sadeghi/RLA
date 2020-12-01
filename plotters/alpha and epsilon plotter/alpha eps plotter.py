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
plt.figure(figsize=(15,8))
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

itNum=5
ticknum=11
samplingPeriod=10

allFiles=os.listdir()


palete=['r','b','g','purple','pink']
tobeDeleted=[]
for files in allFiles:
    # if os.path.splitext(files)[1]!='.npy' or not ('results' in os.path.splitext(files)[0]):
    if os.path.splitext(files)[1]!='.npy':

        tobeDeleted.append(files)

for i in tobeDeleted:
    allFiles.remove(i)

'''this piece of code takes the file name and what 
ever is after character x puts it as the legend for that file'''
label=list(map(lambda x: os.path.splitext(x)[0],allFiles))
for c,v in enumerate(label):
    label[c]=v[v.find('x')+1:]

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

    plt.plot(x,oneIt[:,0],label=label[count])

''' regration approximation '''
# plt.plot(x,0.999**(x/55)-0.1,label=r'$0.999^{x/55}-0.1$')


devisionScale=100000
plt.ylim(-0.1,1.1)
plt.yticks(fontsize=12)

''' cheated by knowing the final time '''
xt=np.linspace(0,1200000,13)
plt.xticks((xt//10).astype(int),(xt//devisionScale).astype(int))

''' indicator of when the env has changed 
np.max(plt.yticks(fontsize=12)[0]) gives the maximum y tick
cheated knowing the mid time 
10 is the sampling period'''
plt.vlines(580000/10,-0.1,np.max(plt.yticks()[0]),color="black",linestyles='--',label='when env has changed',linewidth=2)


plt.legend(fontsize=12.5,loc=1)
plt.xlabel('Time [s] / '+str(devisionScale),fontsize=15,fontweight='bold')
plt.ylabel(r'$\mathbf{\epsilon}$',fontsize=15,fontweight='bold')
plt.title('10 Robots',fontsize=15,fontweight='bold')
plt.grid()
plt.savefig(ctime(TIME()).replace(':','_')+'.png')
plt.show()
print('hi')