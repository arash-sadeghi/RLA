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

# FinalTime=116000*10
# pointN=FinalTime//(1000*5)
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
        datas.append(np.load(_,allow_pickle=True))
        ''' storing number of datas for each iteration for each file '''
        datasLen.append(np.shape(datas[count])[-1]) 

'''this piece of code takes the file name and what 
ever is after character x puts it as the legend for that file'''
label=list(map(lambda x: os.path.splitext(x)[0],allFiles))
for c,v in enumerate(label):
    label[c]=v[v.find('x')+1:]

for count,file_ in enumerate(allFiles):
    '''
    structure hieracrchy of datas:
    files -> iteration -> ROBN ->
    '''
    data=datas[count][0,:,0] # file,iteration,sample,robot
    
    # '''accumulate data'''
    # for i in range(len(data)):
    #     if i==0:
    #         data[i]=data[i]
    #     else:
    #         data[i]=data[i-1]+data[i]

    window=1000 # average each 'window' data and represent it as one point
    pointN=len(data)//window
    averagedData=np.zeros((1,len(data)//window))[0]
    j=0
    for i in range(pointN):
        averagedData[i]=np.mean(data[j:j+window])
        j+=window

    plt.plot(averagedData,color=palete[count],label=label[count]) # plot for only robot 0

# ''' indicator of when the env has changed '''
# ''' np.max(plt.yticks(fontsize=12)[0]) gives the maximum y tick'''
# plt.vlines(len(averagedData)//2,-0.1,np.max(plt.yticks()[0]),color="black",linestyles='--',label='when env has changed',linewidth=2)

# ''' 10 cheated'''
# devisionScale=window*10
# xt=plt.xticks()[0][1:-1]
# xt=np.linspace(int(min(xt)),int(max(xt)),int(max(xt)//10))
# # plt.xticks(plt.xticks()[0][1:-1],(plt.xticks()[0][1:-1]//10).astype(int),fontsize=12)
# plt.xticks(xt,(xt//10).astype(int),fontsize=12)
devisionScale=100000
''' cheated by knowing the last tick value '''
# xt=np.linspace(0,1200000,13)
xt=plt.xticks()[0][-2]
xt=np.arange(0,xt+10,10)
plt.xticks(xt,(xt//10).astype(int))

''' indicator of when the env has changed 
np.max(plt.yticks(fontsize=12)[0]) gives the maximum y tick
cheated knowing the mid time 
10 is the sampling period'''
plt.vlines(580/10,-0.1,np.max(plt.yticks()[0]),color="black",linestyles='--',label='when env has changed',linewidth=2)


plt.yticks(fontsize=12)
plt.legend(fontsize=12.5,loc=1)
plt.xlabel('Time [s] / '+str(devisionScale),fontsize=15,fontweight='bold')
plt.ylabel('reward',fontsize=15,fontweight='bold')
plt.title('10 Robots',fontsize=15,fontweight='bold')
plt.grid()
plt.savefig(ctime(TIME()).replace(':','_')+'.png')
plt.show()
print('hi')