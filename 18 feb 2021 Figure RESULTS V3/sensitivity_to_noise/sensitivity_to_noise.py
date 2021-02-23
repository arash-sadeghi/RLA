import numpy as np
import matplotlib.pyplot as plt
import os
from time import time as TIME
from time import ctime
from termcolor import colored

'''
this code will plot all .npy files in its own location
so do not put any .npy file in its location if you dont
want it to be plotted
'''

import matplotlib as mpl
mpl.rcParams['font.family'] = ['serif']
fig, ax = plt.subplots(1, 2,figsize=(14,5),sharex='col',sharey='row') 


def goToScriptDir():
    ''' with this segment code is callable from any folder '''
    scriptLoc=__file__
    for i in range(len(scriptLoc)):
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

FinalTime=1160000
'''pointN: number of points to be shown in figure '''
pointN=FinalTime//(1000*5)
# pointN=FinalTime

itNum=5 # for title only
samplingPeriod=10 # for arranging x axis only

allFiles=os.listdir()
# label=['BEECLUST','RL without comminucation']
palete=['r','b','g','purple','orange','grey','pink','yellow','cyan',]
tobeDeleted=[]
for files in allFiles:
    if os.path.splitext(files)[1]!='.npy':

        tobeDeleted.append(files)
for i in tobeDeleted:
    allFiles.remove(i)

allFiles.sort()

'''this piece of code takes the file name and what 
ever is after character x puts it as the legend for that file'''
label=list(map(lambda x: os.path.splitext(x)[0],allFiles))
for c,v in enumerate(label):
    label[c]=v[v.find('x')+1:]
''' determining max data len and storing files in a list '''
datas=[]
datasLen=[]
for count,file_ in enumerate(allFiles):
    with open(file_,'rb') as _:
        datas.append(np.load(_))
        ''' storing number of datas for each iteration for each file '''
        datasLen.append(np.shape(datas[count])[-1]) 
 

marker=["o","v","^","s","p","d"]

ycs=[]
ycd=[]
yVs=[]
yVd=[]
yLs=[]
yLd=[]
# yBs=[0.38 for _ in range(6)]
# yBd=[0.38 for _ in range(6)]
yBs=[]
yBd=[]
noises=[0,5,15,90,135,180]
print(colored(str(allFiles),"yellow"))
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

    if "cyclical" in file_:
        ycs.append([\
            np.mean(averagedDataQ1[pointN//2-100:pointN//2]),\
            np.mean(averagedDataMean[pointN//2-100:pointN//2]),\
            np.mean(averagedDataQ2[pointN//2-100:pointN//2])\
            ])
        ycd.append([\
            np.mean(averagedDataQ1[pointN-100:pointN]),\
            np.mean(averagedDataMean[pointN-100:pointN]),\
            np.mean(averagedDataQ2[pointN-100:pointN])\
            ])
    elif "VDBE" in file_: 
        ''' amp ends with p and might get condfused with previous case be carefull'''
        yVs.append([\
            np.mean(averagedDataQ1[pointN//2-100:pointN//2]),\
            np.mean(averagedDataMean[pointN//2-100:pointN//2]),\
            np.mean(averagedDataQ2[pointN//2-100:pointN//2])\
            ])
        yVd.append([\
            np.mean(averagedDataQ1[pointN-100:pointN]),\
            np.mean(averagedDataMean[pointN-100:pointN]),\
            np.mean(averagedDataQ2[pointN-100:pointN])\
            ])
    elif "LBA" in file_:
        yLs.append([\
            np.mean(averagedDataQ1[pointN//2-100:pointN//2]),\
            np.mean(averagedDataMean[pointN//2-100:pointN//2]),\
            np.mean(averagedDataQ2[pointN//2-100:pointN//2])\
            ])
        yLd.append([\
            np.mean(averagedDataQ1[pointN-100:pointN]),\
            np.mean(averagedDataMean[pointN-100:pointN]),\
            np.mean(averagedDataQ2[pointN-100:pointN])\
            ])

    elif "BEECLUST" in file_:
        yBs.append([\
            np.mean(averagedDataQ1[pointN//2-100:pointN//2]),\
            np.mean(averagedDataMean[pointN//2-100:pointN//2]),\
            np.mean(averagedDataQ2[pointN//2-100:pointN//2])\
            ])
        yBd.append([\
            np.mean(averagedDataQ1[pointN-100:pointN]),\
            np.mean(averagedDataMean[pointN-100:pointN]),\
            np.mean(averagedDataQ2[pointN-100:pointN])\
            ])

    else:
        print(colored("[-] file not found","red"),file_)
count=0
ycs=np.array(ycs)
ycd=np.array(ycd)

yVs=np.array(yVs)
yVd=np.array(yVd)

yLs=np.array(yLs)
yLd=np.array(yLd)

yBs=np.array(yBs)
yBd=np.array(yBd)

font=18 ##############################################################################

fig.text(0.05, 0.08, '(a)', ha='center',fontsize=font+3,fontweight="bold")
fig.text(0.535, 0.08, '(b)', ha='center',fontsize=font+3,fontweight="bold")

############################################################################################################################
x=np.arange(0,ycs.shape[0])
plt.sca(ax[0])
color="Blue"
plt.plot(x,ycs[:,1],label='LBA-RL cyclical $p$=100, $A$=1',marker='v',markersize=12,linewidth=3,color='red') 
plt.fill_between(x,ycs[:,0],ycs[:,1],color='red',alpha=0.2)
plt.fill_between(x,ycs[:,1],ycs[:,2],color='red',alpha=0.2)

plt.plot(x,yVs[:,1],label='LBA-RL VDBE $\sigma$=1',marker='v',markersize=12,linewidth=3,color='blue') 
plt.fill_between(x,yVs[:,0],yVs[:,1],color='blue',alpha=0.2)
plt.fill_between(x,yVs[:,1],yVs[:,2],color='blue',alpha=0.2)

plt.plot(x,yLs[:,1],label=r'LBA $\tau_e$=4',marker='v',markersize=12,linewidth=3,color='green') 
plt.fill_between(x,yLs[:,0],yLs[:,1],color='green',alpha=0.2)
plt.fill_between(x,yLs[:,1],yLs[:,2],color='green',alpha=0.2)

plt.plot(yBs[:,1],label='BEECLUST',marker='v',markersize=12,linewidth=3,color='purple') 
plt.fill_between(x,yBs[:,0],yBs[:,1],color='purple',alpha=0.2)
plt.fill_between(x,yBs[:,1],yBs[:,2],color='purple',alpha=0.2)

# plt.title('(a)',fontsize=font+1,fontweight='bold')
plt.yticks(fontsize=font)
plt.xticks([0,1,2,3,4,5],[0,5,15,90,135,180],fontsize=font)
plt.ylabel('Normalized Aggregation Size',fontsize=font+1,fontweight='bold')
plt.xlabel(r'$\mathbf{\sigma_n\degree}$',fontsize=font+1,fontweight='bold')
# ############################################################################################################################
plt.sca(ax[1])
color="Blue"
plt.plot(x,ycd[:,1],label='LBA-RL cyclical $p$=100, $A$=1',marker='v',markersize=12,linewidth=3,color='red') 
plt.fill_between(x,ycd[:,0],ycd[:,1],color='red',alpha=0.2)
plt.fill_between(x,ycd[:,1],ycd[:,2],color='red',alpha=0.2)

plt.plot(x,yVd[:,1],label='LBA-RL VDBE $\sigma$=1',marker='v',markersize=12,linewidth=3,color='blue') 
plt.fill_between(x,yVd[:,0],yVd[:,1],color='blue',alpha=0.2)
plt.fill_between(x,yVd[:,1],yVd[:,2],color='blue',alpha=0.2)

plt.plot(x,yLd[:,1],label=r'LBA $\tau_e$=4',marker='v',markersize=12,linewidth=3,color='green') 
plt.fill_between(x,yLd[:,0],yLd[:,1],color='green',alpha=0.2)
plt.fill_between(x,yLd[:,1],yLd[:,2],color='green',alpha=0.2)

plt.plot(yBd[:,1],label='BEECLUST',marker='v',markersize=12,linewidth=3,color='purple') 
plt.fill_between(x,yBd[:,0],yBd[:,1],color='purple',alpha=0.2)
plt.fill_between(x,yBd[:,1],yBd[:,2],color='purple',alpha=0.2)

plt.yticks(fontsize=font)
plt.xticks([0,1,2,3,4,5],[0,5,15,90,135,180],fontsize=font)
plt.xlabel(r'$\mathbf{\sigma_n\degree}$',fontsize=font+1,fontweight='bold')
# legends=plt.legend(fontsize=font-2,loc=1)
# ############################################################################################################################
plt.setp(ax, ylim=(0.3,0.7))
# plt.setp(legends.texts,family='Times New Roman')
# fig.suptitle('sensitivity of aggregation methods to noise',fontsize=14,fontweight='bold')
# lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
# lines, labels = [sum(zip(*lines_labels), []) ]
lines_labels = ax[0].get_legend_handles_labels()
fig.legend(lines_labels[0], lines_labels[1],ncol=4,loc=1,fontsize=font-0.75,bbox_to_anchor=(0.995, 1))
plt.tight_layout()
plt.subplots_adjust(wspace=0.08,top=0.874)
plt.savefig('/home/arash/Dropbox/sensitivity_to_noise.png')
plt.savefig('sensitivity_to_noise.png')

plt.show()
print('hi')