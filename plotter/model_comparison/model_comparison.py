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
fig, ax = plt.subplots(1, 2,figsize=(15.5,8),sharex='col',sharey='row') 
# fig, ax = plt.subplots(1, 2,figsize=(14,6),sharex='col',sharey='row') 

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


marker=["o","v","^","s","p","d"]
n_NAS=[0 for _ in range(4)]
n_NAS_L=[0 for _ in range(4)]

# n_R=[0 for _ in range(4)]
# n_R_L=[0 for _ in range(4)]

NAS=[0 for _ in range(4)]
NAS_L=[0 for _ in range(4)]

# R=[0 for _ in range(4)]
# R_L=[0 for _ in range(4)]

for c,v in enumerate(allFiles):
    if "BEECLUST" in v:
        indx=0
    elif ("LBA" in v) and (not "LBA-RL" in v) :
        indx=1
    elif "VDBE" in v:
        indx=2
    elif "cyclical" in v:
        indx=3




    if "-n-" in v:
        if "NAS" in v:
            with open(v,'rb') as _:
                n_NAS[indx]=np.load(_)
                L=os.path.splitext(v)[0]
                n_NAS_L[indx]=L[L.find('x')+1:]

        elif "reward" in v:
            pass
    else:
        if "NAS" in v:
            with open(v,'rb') as _:
                NAS[indx]=np.load(_)
                L=os.path.splitext(v)[0]
                NAS_L[indx]=L[L.find('x')+1:]
        elif "reward" in v:
            pass
#------------------------------------------------------------------------
def plotter(arr,sca,lable_):
    for count,data in enumerate(arr):
        plt.sca(ax[sca])
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
    if sca==0:
        # plt.title('(a) Noiseless Setup $\mathbf{\sigma_n=0\degree}$',fontsize=15,fontweight='bold')
        plt.ylim(0,1)
        plt.xlabel('Time [s] / '+str(devisionScale//10),fontsize=15,fontweight='bold')
        plt.ylabel('Normalized Aggregation Size',fontsize=15,fontweight='bold')

    elif sca==1:
        # plt.title('(b) Noisy Setup $\mathbf{\sigma_n=15\degree}$',fontsize=15,fontweight='bold')
        plt.ylim(0,1)

        plt.xlabel('Time [s] / '+str(devisionScale//10),fontsize=15,fontweight='bold')
        # L=plt.legend(fontsize=11.5,loc=4)

#------------------------------------------------------------------------

plotter(NAS,0,NAS_L)
plotter(n_NAS,1,n_NAS_L)
lines_labels = ax[1].get_legend_handles_labels()
fig.legend(lines_labels[0], lines_labels[1],ncol=5,loc=1,fontsize=15.65,bbox_to_anchor=(0.95, 1))
fig.text(0.05-0.01, 0.08-0.05, '(a)', ha='center',fontsize=17,fontweight="bold")
fig.text(0.535-0.01, 0.08-0.05, '(b)', ha='center',fontsize=17,fontweight="bold")

plt.tight_layout()
plt.subplots_adjust(wspace=0.08,top=0.92)

plt.savefig('model_comparison.png')
plt.savefig('/home/arash/Dropbox/model_comparison.png')
plt.show()
print('hi')
