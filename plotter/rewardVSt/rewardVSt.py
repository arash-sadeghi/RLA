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
fig, ax = plt.subplots(1, 1,figsize=(15.5/2,6),sharex='col',sharey='row') 
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
#------------------------------------------------------------------------
def plotter():
    allFiles=os.listdir()
    palete=['r','b','g','purple','pink']
    tobeDeleted=[]
    for files in allFiles:
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
        data=datas[count][:,:,0] # file,iteration,sample,robot
        
        '''accumulate data'''
        window=1000 # average each 'window' data and represent it as one point
        pointN=datas[count].shape[1]//window
        averagedData=np.zeros((datas[count].shape[0],pointN))
        j=0
        for i in range(pointN):
            averagedData[:,i]=np.mean(data[:,j:j+window],axis=1)
            j+=window
        '''' boxplot between iteration '''
        Q1=np.zeros(pointN)
        Q2=np.zeros(pointN)
        Q3=np.zeros(pointN)

        for i in range(pointN):
            Q1[i]=np.percentile(averagedData[:,i],25)
            Q2[i]=np.percentile(averagedData[:,i],50)
            Q3[i]=np.percentile(averagedData[:,i],75)

        x=np.arange(0,pointN)
        plt.plot(x,Q2,color=palete[count],label=label[count]) # plot for only robot 0
        plt.fill_between(x,Q2,Q1,color=palete[count],alpha=0.2)
        plt.fill_between(x,Q3,Q2,color=palete[count],alpha=0.2)

    ''' cheated by knowing the final time '''
    devisionScale=100000
    # plt.vlines(50,-0.1,150,color="black",linestyles='--',label='when environment has changed',linewidth=2)
    plt.vlines(50,-0.1,150,color="black",linestyles='--',label=r'$t_{change}$',linewidth=2)

    plt.yticks(fontsize=12)
    plt.xticks([0,10,20,30,40,50,60,70,80,90,100],[0,1,2,3,4,5,6,7,8,9,10],fontsize=12)
    plt.xlim(0,int(100))

    plt.ylim(0,150)
    plt.xlabel('Time [s] / '+str(devisionScale),fontsize=17,fontweight='bold')
    plt.ylabel('Reward',fontsize=17,fontweight='bold')
    # plt.legend(fontsize=15,loc=2)
    lines_labels = ax.get_legend_handles_labels()
    fig.legend(lines_labels[0], lines_labels[1],loc=1,ncol=2,fontsize=17,bbox_to_anchor=(0.88, 1))

#------------------------------------------------------------------------

plotter()
plt.tight_layout()
plt.subplots_adjust(top=0.83)
plt.savefig('rewardVSt.png')
plt.savefig('/home/arash/Dropbox/rewardVSt.png')
plt.show()
print('hi')
