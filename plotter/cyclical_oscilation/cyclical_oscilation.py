import numpy as np
import matplotlib.pyplot as plt
import os
from time import time as TIME
from time import ctime
from lmfit import Model

'''
this code will plot all .npy files in its own location
so do not put any .npy file in its location if you dont
want it to be plotted
'''
# plt.rcParams['axes.grid'] = True
import matplotlib as mpl
mpl.rcParams['font.family'] = ['serif']
fig, ax = plt.subplots(1, 1,figsize=(15.5/2,5),sharex='col',sharey='row') 
# fig, ax = plt.subplots(2, 1,figsize=(12,12),sharex='col',sharey='row') 

#--------------------------------------------------------------------------------------
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
#--------------------------------------------------------------------------------------
FinalTime=1160000
'''pointN: number of points to be shown in figure '''
pointN=FinalTime//(1000*5)
# pointN=FinalTime//100
datasLen=int(1e+5)
itNum=5 # for title only
samplingPeriod=10 # for arranging x axis only

allFiles=os.listdir()
palete=['r','b','g','purple','orange','grey','pink','yellow','cyan',]
tobeDeleted=[]

''' remove files which are not npy type '''
for files in allFiles:
    if os.path.splitext(files)[1]!='.npy':
        tobeDeleted.append(files)
for i in tobeDeleted:
    allFiles.remove(i)

allFiles.sort()
marker=["o","v","^","s","p","d"]
NAS=[0 for _ in range(len(allFiles))]
NAS_L=[0 for _ in range(len(allFiles))]
for c,v in enumerate(allFiles):
    with open(v,'rb') as _:
        NAS[c]=np.load(_)
        L=os.path.splitext(v)[0]
        NAS_L[c]=L[L.find('x')+1:]
        NAS_L[c]=NAS_L[c].replace("  noise 15","")
        NAS_L[c]="LBA-RL"+NAS_L[c]

#------------------------------------------------------------------------
def plotter(arr,sca,lable_):
    for count,data in enumerate(arr):
        plt.sca(ax)
        # data=datas[count]
        if data.shape[1]>1e+5:
            ''' BEECLUST file is 1160000 s '''
            data=data[:,0:datasLen]
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

        x=np.arange(0,len(averagedDataMean))*((samplingPeriod*datasLen)/len(averagedDataMean))

        plt.plot(x,averagedDataMean,color=palete[count],label=lable_[count]) 
        plt.plot(x,[np.mean(averagedDataMean[-10:-1]) for _ in range(len(averagedDataMean))],color=palete[count]) 

        plt.fill_between(x,averagedDataMean,averagedDataQ2,color=palete[count],alpha=0.2)
        plt.fill_between(x,averagedDataQ1,averagedDataMean,color=palete[count],alpha=0.2)

    ''' cheated by knowing the final time '''
    devisionScale=100000
    xt=np.linspace(0,1200000,13)
    plt.xticks(xt,(xt//devisionScale).astype(int))
    plt.vlines(500000,-0.1,1,color="black",linestyles='--',label=r'$t_{change}$',linewidth=2)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlim(0,int(1e+6))

    # plt.title('(a) Noiseless Setup $\mathbf{\sigma_n=0\degree}$',fontsize=15,fontweight='bold')
    plt.ylim(0,1)
    plt.xlabel('Time [s] / '+str(devisionScale//10),fontsize=15,fontweight='bold')
    plt.ylabel('Normalized Aggregation Size',fontsize=15,fontweight='bold')
    plt.legend(fontsize=14.3,loc=3)

#------------------------------------------------------------------------

plotter(NAS,0,NAS_L)
# lines_labels = ax.get_legend_handles_labels()
# fig.legend(lines_labels[0], lines_labels[1],ncol=5,loc=1,fontsize=13,bbox_to_anchor=(1, 1))

plt.tight_layout()

plt.savefig('cyclical_oscilation.png')
plt.savefig('/home/arash/Dropbox/cyclical_oscilation.png')
plt.show()
print('hi')
