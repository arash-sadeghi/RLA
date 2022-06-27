import numpy as np
import matplotlib.pyplot as plt
import os
from time import time as TIME
from time import ctime
import matplotlib as mpl
mpl.rcParams['font.family'] = ['serif']
fig, ax = plt.subplots(1, 2,figsize=(15.5,8),sharex='col',sharey='row') 

class plotter:
    def __init__(self):
        pass


#! path to folder containing .npy s to be plotted
input_folder_paths="/home/arash/Desktop/workdir/emre_thesis/RLA/MAIN_SIMULATOR_FILES/output/Initial_comparison/NAS"
output_path="/home/arash/Desktop/workdir/emre_thesis/RLA/MAIN_SIMULATOR_FILES/output/Initial_comparison/NAS"
FinalTime=1160000
'''pointN: number of points to be shown in figure '''
pointN=FinalTime//(1000*5)
# pointN=FinalTime//100
datasLen=int(1e+5)
samplingPeriod=10 # for arranging x axis only

palete=['r','b','g','purple','orange','grey','pink','yellow','cyan',]
marker=["o","v","^","s","p","d"]

#! get complete path
allFiles=list(map(lambda x : os.path.join(input_folder_paths , x) , os.listdir(input_folder_paths)))
arrays=[]
labels=[]
''' remove files which are not npy type '''
for file in allFiles:
    #! only plot .npy files
    if os.path.splitext(file)[1]=='.npy':
        arrays.append( np.load(file) )
        labels.append(file)

#------------------------------------------------------------------------
def plotter(arr,sca,lable_):
    for count,data in enumerate(arr):
        plt.sca(ax[sca])
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
for i in range( len(arrays) ):
    plotter(arrays[i],0,labels[i])

lines_labels = ax[1].get_legend_handles_labels()
fig.legend(lines_labels[0], lines_labels[1],ncol=5,loc=1,fontsize=15.65,bbox_to_anchor=(0.95, 1))

fig.text(0.05-0.01, 0.08-0.05, '(a)', ha='center',fontsize=17,fontweight="bold")

plt.tight_layout()
plt.subplots_adjust(wspace=0.08,top=0.92)

plt.savefig(os.path.join(output_path , 'model_comparison.png'))
plt.show()
print('hi')
