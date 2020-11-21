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
 
for count,file_ in enumerate(allFiles):
    '''
    structure hieracrchy of datas:
    files -> iteration -> ROBN ->
    '''
    data=datas[count][0,:,0] # file,iteration,sample,robot
    
    '''accumulate data'''
    for i in range(len(data)):
        if i==0:
            data[i]=data[i]
        else:
            data[i]=data[i-1]+data[i]

    window=10 # average each 'window' data and represent it as one point
    pointN=len(data)//window
    averagedData=np.zeros((1,len(data)//window))[0]
    j=0
    for i in range(pointN):
        averagedData[i]=np.mean(data[j:j+window])
        j+=window

    plt.plot(averagedData,color=palete[count],label=file_+' robot 0') # plot for only robot 0

'''
for count,file_ in enumerate(allFiles):
    data=datas[count]
    averagedData=np.zeros((len(data),pointN))
    window=np.shape(data)[1]//pointN # 1000 is the number of points that i want to see in plot
    for k in range(len(data)):
        j=0
        for i in range(pointN):
            averagedData[k,i]=np.mean(data[k,j:j+window])
            j+=window
    
    averagedDataMean=np.zeros((1,pointN+1))[0]
    averagedDataQ1=np.zeros((1,pointN+1))[0]
    averagedDataQ2=np.zeros((1,pointN+1))[0]

    for i in range(pointN):
        averagedDataMean[1+i]=np.percentile(averagedData[:,i],50)
        averagedDataQ1[1+i]=np.percentile(averagedData[:,i],25)
        averagedDataQ2[1+i]=np.percentile(averagedData[:,i],75)

    x=np.arange(0,len(averagedDataMean))*((samplingPeriod*datasLen[count])/len(averagedDataMean))
    # plt.plot(x,averagedDataMean,color=palete[count],label=label[count])
    plt.plot(x,averagedDataMean,color=palete[count],label=file_)


    plt.fill_between(x,averagedDataMean,averagedDataQ2,color=palete[count],alpha=0.25)
    plt.fill_between(x,averagedDataQ1,averagedDataMean,color=palete[count],alpha=0.25)

'''


devisionScale=100
plt.yticks(fontsize=12)
# plt.ylim(-0.1,1.1)
# plt.xlim(-0.1,)
plt.legend(fontsize=12.5,loc=1)
plt.xlabel('Time [s] / '+str(devisionScale),fontsize=15,fontweight='bold')
plt.ylabel('reward',fontsize=15,fontweight='bold')
plt.title('10 Robots',fontsize=15,fontweight='bold')
plt.grid()
plt.savefig(ctime(TIME()).replace(':','_')+'.png')
plt.show()
print('hi')