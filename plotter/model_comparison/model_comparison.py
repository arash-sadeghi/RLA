from codecs import raw_unicode_escape_decode
from typing import Final
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time as TIME
from time import ctime
import matplotlib as mpl
import pandas as pd
mpl.rcParams['font.family'] = ['serif']
fig, ax = plt.subplots(1, 1,figsize=(15.5,8),sharex='col',sharey='row') 

class plotter:
    def __init__(self):
        pass


#! path to folder containing .npy s to be plotted
input_folder_paths="./data"
output_path=input_folder_paths
# pointN_percent=0.00001 # you will have pointN_percent*data point number of points in your graph
pointN_percent=0.0001 # you will have pointN_percent*data point number of points in your graph
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

def moving_average_pd(arr , window_size):
    # Convert array of integers to pandas series
    numbers_series = pd.Series(arr)
    
    # Get the window of series
    # of observations of specified window size
    windows = numbers_series.rolling(window_size)
    
    # Create a series of moving
    # averages of each window
    moving_averages = windows.mean()
    
    # Convert pandas series back to list
    moving_averages_list = moving_averages.tolist()
    
    # Remove null entries from the list
    final_list = moving_averages_list[window_size - 1:]
  
    return np.array(final_list)
#------------------------------------------------------------------------
def plotter(arr,lable_,color):
    global pointN_percent
    """ 
    row_num:
    var is for determining how many rows should the averaged data containt.
    usually it should be equalt to the number of iterations (rows) in the file. Exception
    is when data is single rowed, meaning that file contains data only for one iteration.
    data_num:
    number of data points inside the file.
    """
    if len(arr.shape) == 0:
        raise NameError("[-] array is empty")
    elif len(arr.shape) == 1:
        print("[+] array is one dimentional")
        pointN = int(pointN_percent * arr.shape[0])
        row_num = 1 # this var is for determining how many rows should the 
        data_num = arr.shape[0]
        arr = np.reshape(arr,(1,arr.shape[0])) # for making array compatible with the rest of function
    else:
        row_num = arr.shape[0]
        data_num = arr.shape[1]
        print("[+] iterations are in rows")

    pointN = int(pointN_percent * data_num)
    
    averagedData=np.zeros(( row_num ,pointN))
    plt.sca(ax)
    window=data_num//pointN # 1000 is the number of points that i want to see in plot

    ''' averaging in time '''
    tmp = moving_average_pd(arr[0,:],100_000)
    plt.plot(tmp);plt.show()    
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

    plt.plot(x,averagedDataMean,color=color,label=lable_) 
    df['brandA'].rolling(window =20).mean().plot()

    plt.fill_between(x,averagedDataMean,averagedDataQ2,color=color,alpha=0.2)
    plt.fill_between(x,averagedDataQ1,averagedDataMean,color=color,alpha=0.2)
    return x
#------------------------------------------------------------------------
for i in range( len(arrays) ):
    label = os.path.basename(labels[i]).split('.')[0]
    if 'x' in  label:
        label = label.split('x')[1]
        print(f"[+] comment divide char x was used")
    else:
        print(f"[+] comment divide char was not used")

    x_axis = plotter(arrays[i],label,palete[i])

lines_labels = ax.get_legend_handles_labels()
fig.legend(lines_labels[0], lines_labels[1],ncol=5,loc=1,fontsize=15.65,bbox_to_anchor=(0.95, 1))

''' cheated by knowing the final time '''
devisionScale=10000
FinalTime = 100000
xt=np.linspace(0,FinalTime,13)
# plt.xticks(xt,(xt//devisionScale).astype(int))
plt.vlines(x_axis[-1]//2,-0.1,1,color="black",linestyles='--',label=r'$t_{change}$',linewidth=2)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlim(0,int(1e+6))

# plt.title('(a) Noiseless Setup $\mathbf{\sigma_n=0\degree}$',fontsize=15,fontweight='bold')
plt.ylim(0,1)
# plt.xlabel('Time [s] / '+str(devisionScale//10),fontsize=15,fontweight='bold')
plt.xlabel('Time [s]',fontsize=15,fontweight='bold')
plt.ylabel('Normalized Aggregation Size',fontsize=15,fontweight='bold')


plt.tight_layout()
plt.subplots_adjust(wspace=0.08,top=0.92)

plt.savefig(os.path.join(output_path , 'model_comparison.png'))
plt.show()
print('hi')
