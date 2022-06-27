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
fig, ax = plt.subplots(1, 1,figsize=(13.15,8.5),sharex='col',sharey='row') 
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
        data=datas[count][0,:,0] # file,iteration,sample,robot
        
        x=np.arange(0,data.shape[0])
        if "VDBE" in file_:
            for i in range(1,data.shape[1]):
                # plt.plot(x,data[:,i],color=palete[count],label=label[count]+" state "+str(i)) # plot for only robot 0
                plt.plot(x,data[:,i],label=label[count]+" state "+str(i)) # plot for only robot 0

        else:
            # plt.plot(x,data,color=palete[count],label=label[count]) # plot for only robot 0
            plt.plot(x,data,label=label[count]) # plot for only robot 0

    ''' cheated by knowing the final time '''
    devisionScale=100000
    plt.vlines(5e+4,-0.1,150,color="black",linestyles='--',label=r'$t_{change}$',linewidth=2)
    plt.yticks(fontsize=16.5)
    plt.xticks(np.arange(0,int(1e+5)+int(1e+4),int(1e+4)),np.arange(0,int(1e+5)+int(1e+4),int(1e+4))//int(1e+4),fontsize=16.5)
    plt.xlim(0,int(1e+5))

    plt.ylim(0,1)
    plt.xlabel('Time [s] / '+str(devisionScale),fontsize=20,fontweight='bold')
    plt.ylabel(r'$\mathbf{\varepsilon}$',fontsize=26,fontweight='bold')
    # plt.legend(fontsize=11.5,loc=2)
    lines_labels = ax.get_legend_handles_labels()
    fig.legend(lines_labels[0], lines_labels[1],loc=1,ncol=3,fontsize=17,bbox_to_anchor=(0.88, 1))


#------------------------------------------------------------------------

plotter()
# lines_labels = ax.get_legend_handles_labels()
# fig.legend(lines_labels[0], lines_labels[1],ncol=5,loc=1,fontsize=13,bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.subplots_adjust(top=0.85)

plt.savefig('epsilonVSt.png')
plt.savefig('/home/arash/Dropbox/epsilonVSt.png')
plt.show()
print('hi')
