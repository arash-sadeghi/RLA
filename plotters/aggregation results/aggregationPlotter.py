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

FinalTime=116000*10
pointN=FinalTime//(1000*5)
itNum=5 # for title only
samplingPeriod=10 # for arranging x axis only

allFiles=os.listdir()
# label=['BEECLUST','RL without comminucation']
palete=['r','b','g','purple','orange','grey']
tobeDeleted=[]
for files in allFiles:
    if os.path.splitext(files)[1]!='.npy':

        tobeDeleted.append(files)
for i in tobeDeleted:
    allFiles.remove(i)

'''this piece of code takes the file name and what 
ever is after character x puts it as the legend for that file'''
label=list(map(lambda x: os.path.splitext(x)[0],allFiles))
for c,v in enumerate(label):
    label[c]=v[v.find('x')+1:]

# label=['alpha 0.5 eps damp 0.9',\
#        'alpha 0.1 eps damp 0.999',\
#        'alpha 0.5 eps damp 0.999',\
#        'alpha 1 eps damp 0.9',\
#         'alpha 1 eps damp 0.999',\
#         'alpha 0.5 eps damp 0.999 static']

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
    averagedData=np.zeros((len(data),pointN))
    window=np.shape(data)[1]//pointN # 1000 is the number of points that i want to see in plot
    ''' averaging in time '''
    for k in range(len(data)):
        j=0
        for i in range(pointN):
            averagedData[k,i]=np.mean(data[k,j:j+window])
            j+=window
    
    ''' +1 is for injecting 0 in the beggining of array '''
    averagedDataMean=np.zeros((1,pointN+1))[0]
    averagedDataQ1=np.zeros((1,pointN+1))[0]
    averagedDataQ2=np.zeros((1,pointN+1))[0]
    ''' averaging and Q1 and Q3 for shades '''
    for i in range(pointN):
        averagedDataMean[1+i]=np.percentile(averagedData[:,i],50)
        averagedDataQ1[1+i]=np.percentile(averagedData[:,i],25)
        averagedDataQ2[1+i]=np.percentile(averagedData[:,i],75)

    x=np.arange(0,len(averagedDataMean))*((samplingPeriod*datasLen[count])/len(averagedDataMean))
    # plt.plot(x,averagedDataMean,color=palete[count],label=label[count])
    plt.plot(x,averagedDataMean,color=palete[count],label=label[count])


    plt.fill_between(x,averagedDataMean,averagedDataQ2,color=palete[count],alpha=0.25)
    plt.fill_between(x,averagedDataQ1,averagedDataMean,color=palete[count],alpha=0.25)

''' cheated by knowing the final time '''
devisionScale=100000
xt=np.linspace(0,1200000,13)
plt.xticks(xt,(xt//devisionScale).astype(int))

''' indicator of when the env has changed 
np.max(plt.yticks(fontsize=12)[0]) gives the maximum y tick
cheated knowing the mid time 
10 is the sampling period'''
plt.vlines(580000,-0.1,np.max(plt.yticks()[0]),color="black",linestyles='--',label='when env has changed',linewidth=2)


plt.yticks(fontsize=12)
plt.ylim(0,1)
plt.xlim(0,)
plt.legend(fontsize=12.5,loc=4)
plt.xlabel('Time [s] / '+str(devisionScale),fontsize=15,fontweight='bold')
plt.ylabel('Normalized aggregation size',fontsize=15,fontweight='bold')
plt.title('10 Robots with '+str(itNum)+' time repetition',fontsize=15,fontweight='bold')
plt.grid()
plt.savefig(ctime(TIME()).replace(':','_')+'.png')
plt.show()
print('hi')