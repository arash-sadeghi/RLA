import numpy as np
import matplotlib.pyplot as plt
import os
from time import time as TIME
from time import ctime
scriptLoc=__file__
''' with this segment code is callable from any folder '''
for i in range(len(scriptLoc)):
    # if '/' in scriptLoc[-i-2:-i]: # in running
    if '\\' in scriptLoc[-i-2:-i]: # in debuging

        scriptLoc=scriptLoc[0:-i-2]
        break
print('[+] code path',scriptLoc)
os.chdir(scriptLoc)
''' done '''

samplingPeriodSmall=10
FinalTime=116000
Datalen=FinalTime//samplingPeriodSmall
# pointN=1000//(1*15)
pointN=11600//100
itNum=5
ticknum=11

allFiles=os.listdir()
# label=['BEECLUST','RL without comminucation']
palete=['r','b','g','purple','pink']
tobeDeleted=[]
for files in allFiles:
    if os.path.splitext(files)[1]!='.npy' or not ('rewards' in os.path.splitext(files)[0]):
        tobeDeleted.append(files)

for i in tobeDeleted:
    allFiles.remove(i)

label=list(map(lambda x: os.path.splitext(x)[0],allFiles))
for count,file_ in enumerate(allFiles):
    with open(file_,'rb') as Reward:
        data=np.load(Reward,allow_pickle=True)
    averagedData=np.zeros((len(data),pointN))
    window=np.shape(data)[1]//pointN # 1000 is the number of points that i want to see in plot
    for k in range(len(data)):
        j=0
        for i in range(pointN):
            averagedData[k,i]=np.mean(data[k,j:j+window])
            j+=window
    
    ''' +1 is for injecting 0 in the beggining of array '''
    averagedDataMean=np.zeros((1,pointN+1))[0]
    averagedDataQ1=np.zeros((1,pointN+1))[0]
    averagedDataQ2=np.zeros((1,pointN+1))[0]

    for i in range(pointN):
        averagedDataMean[1+i]=np.percentile(averagedData[:,i],50)
        averagedDataQ1[1+i]=np.percentile(averagedData[:,i],25)
        averagedDataQ2[1+i]=np.percentile(averagedData[:,i],75)

    x=np.arange(0,len(averagedDataMean))
    plt.plot(x,averagedDataMean,color=palete[count],label=label[count])

    plt.fill_between(x,averagedDataMean,averagedDataQ2,color=palete[count],alpha=0.25)
    plt.fill_between(x,averagedDataQ1,averagedDataMean,color=palete[count],alpha=0.25)

# desiredTicks=np.arange(0,FinalTime+20000,20000,dtype='int32')//100
# orginalTicks=np.linspace(0,x[-1],len(desiredTicks))

# orginalTicks=np.linspace(0,x[-1],len(desiredTicks))
# desiredTicks=orginalTicks[0:10000]

desiredTicks=np.arange(0,x[-1]+20,20,dtype='int32')

plt.xticks(desiredTicks,desiredTicks*10,fontsize=12)

# plt.xticks(orginalTicks,desiredTicks,fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0,1)
plt.xlim(0,)
plt.legend(fontsize=15)
plt.xlabel('Time [s] / 100',fontsize=15,fontweight='bold')
plt.ylabel('Normalized aggregation size',fontsize=15,fontweight='bold')
plt.title('10 Robots with '+str(itNum)+' time repetition',fontsize=15,fontweight='bold')
plt.grid()
plt.savefig(ctime(TIME()).replace(':','_')+'.png')
plt.show()