from typing import Final
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time as TIME
from time import ctime
import matplotlib as mpl
mpl.rcParams['font.family'] = ['serif']
fig, ax = plt.subplots(1, 1,figsize=(15.5,8),sharex='col',sharey='row') 

class plotter:
    def __init__(self):
        pass


#! path to folder containing .npy s to be plotted
input_folder_paths="/home/arash/Desktop/workdir/emre_thesis/RLA/MAIN_SIMULATOR_FILES/output/CASES/res"
output_path=input_folder_paths
pointN_percent=0.01 # you will have pointN_percent*data point number of points in your graph
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
def plotter(arr,lable_,color):
    global pointN_percent
    pointN = int(pointN_percent * arr.shape[1])
    averagedData=np.zeros(( arr.shape[0] ,pointN))
    plt.sca(ax)
    window=np.shape(arr)[1]//pointN # 1000 is the number of points that i want to see in plot

    ''' averaging in time '''
    j=0
    for i in range( averagedData.shape[1] ):
        averagedData[:,i] = np.mean(arr[:,j:j+window],axis=1)
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

    plt.plot(x,averagedDataMean,color=color,label=lable_) 
    plt.fill_between(x,averagedDataMean,averagedDataQ2,color=color,alpha=0.2)
    plt.fill_between(x,averagedDataQ1,averagedDataMean,color=color,alpha=0.2)
    return x
#------------------------------------------------------------------------
for i in range( len(arrays) ):
    label = os.path.basename(labels[i]).split('.')[0].split('x')[1]
    x_axis = plotter(arrays[i],label,palete[i])

lines_labels = ax.get_legend_handles_labels()
fig.legend(lines_labels[0], lines_labels[1],ncol=5,loc=1,fontsize=15.65,bbox_to_anchor=(0.95, 1))

''' cheated by knowing the final time '''
devisionScale=1
FinalTime = 50_000
xt=np.linspace(0,FinalTime,13)
# plt.xticks(xt,(xt//devisionScale).astype(int))
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
# plt.xlim(0,int(1e+6))

# plt.title('(a) Noiseless Setup $\mathbf{\sigma_n=0\degree}$',fontsize=15,fontweight='bold')
# plt.ylim(0,1)
# plt.xlabel('Time [s] / '+str(devisionScale//10),fontsize=15,fontweight='bold')
plt.xlabel('Time [s]',fontsize=15,fontweight='bold')
plt.ylabel('Normalized Aggregation Size',fontsize=15,fontweight='bold')


plt.tight_layout()
plt.subplots_adjust(wspace=0.08,top=0.92)

plt.savefig(os.path.join(output_path , 'model_comparison.png'))
plt.show()
print('hi')
