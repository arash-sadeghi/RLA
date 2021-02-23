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

'''
plt.rcParams['axes.grid'] = True
will work only at the beggining
plt.grid() wont work
'''
# plt.rcParams['axes.grid'] = True
import matplotlib as mpl
mpl.rcParams['font.family'] = ['serif']
fig, ax = plt.subplots(2, 3,figsize=(13,7.5),sharex='col',sharey='row') 

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
    label[c]="LBA-RL "+v[v.find('x')+1:]
''' determining max data len and storing files in a list '''
datas=[]
datasLen=[]
for count,file_ in enumerate(allFiles):
    with open(file_,'rb') as _:
        datas.append(np.load(_))
        ''' storing number of datas for each iteration for each file '''
        datasLen.append(np.shape(datas[count])[-1]) 
 

marker=["o","v","^","s","p","d"]

yfs=[]
yfd=[]
yAs=[]
yAd=[]
yVs=[]
yVd=[]

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

    if " p=" in file_ or " P=" in file_:
        yfs.append(\
            [\
            np.mean(averagedDataQ1[pointN//2-100:pointN//2]),\
            np.mean(averagedDataMean[pointN//2-100:pointN//2]),\
            np.mean(averagedDataQ2[pointN//2-100:pointN//2])])
        
        yfd.append(\
            [\
            np.mean(averagedDataQ1[pointN-100:pointN]),\
            np.mean(averagedDataMean[pointN-100:pointN]),\
            np.mean(averagedDataQ2[pointN-100:pointN])])
    elif "amplitude" in file_ or "amp" in file_: 
        ''' amp ends with p and might get condfused with previous case be carefull'''
        yAs.append(\
            [\
            np.mean(averagedDataQ1[pointN//2-100:pointN//2]),\
            np.mean(averagedDataMean[pointN//2-100:pointN//2]),\
            np.mean(averagedDataQ2[pointN//2-100:pointN//2])])
        
        yAd.append(\
            [\
            np.mean(averagedDataQ1[pointN-100:pointN]),\
            np.mean(averagedDataMean[pointN-100:pointN]),\
            np.mean(averagedDataQ2[pointN-100:pointN])])
    elif "VDBE" in file_:
        yVs.append(\
            [\
            np.mean(averagedDataQ1[pointN//2-100:pointN//2]),\
            np.mean(averagedDataMean[pointN//2-100:pointN//2]),\
            np.mean(averagedDataQ2[pointN//2-100:pointN//2])])
        
        yVd.append(\
            [\
            np.mean(averagedDataQ1[pointN-100:pointN]),\
            np.mean(averagedDataMean[pointN-100:pointN]),\
            np.mean(averagedDataQ2[pointN-100:pointN])\
            ])

    else:
        print(colored("[-] file not found","red"),file_)
count=0
font=16
color="Blue"
############################################################################################################################
plt.sca(ax[0,0])
yfs=np.array(yfs)
x=np.arange(0,yfs.shape[0])
plt.plot(x,yfs[:,1],color=color,label=label[count],marker='v',markersize=15,linewidth=3) 
plt.fill_between(x,yfs[:,0],yfs[:,1],color=color,alpha=0.2)
plt.fill_between(x,yfs[:,1],yfs[:,2],color=color,alpha=0.2)
plt.yticks(fontsize=font-3)
# plt.ylabel('Normalized Aggregation Size',fontsize=font,fontweight='bold')
############################################################################################################################
plt.sca(ax[1,0])
yfd=np.array(yfd)
x=np.arange(0,yfd.shape[0])
plt.plot(yfd[:,1],color=color,label=label[count],marker='v',markersize=15,linewidth=3) 
plt.fill_between(x,yfd[:,0],yfd[:,1],color=color,alpha=0.2)
plt.fill_between(x,yfd[:,1],yfd[:,2],color=color,alpha=0.2)
plt.yticks(fontsize=font-3)
plt.xlabel(r"cyclical $p$",fontsize=font+3,fontweight='bold')
# plt.ylabel('Normalized Aggregation Size',fontsize=font,fontweight='bold')
plt.xticks([0,1,2,3,4,5,6,7,8],[0.01,0.1,1,10,50,100,150,200,500],fontsize=font-3.5)
############################################################################################################################
plt.sca(ax[0,1])
yAs=np.array(yAs)
x=np.arange(0,yAs.shape[0])
plt.plot(yAs[:,1],color=color,label=label[count],marker='v',markersize=15,linewidth=3) 
plt.fill_between(x,yAs[:,0],yAs[:,1],color=color,alpha=0.2)
plt.fill_between(x,yAs[:,1],yAs[:,2],color=color,alpha=0.2)
############################################################################################################################
plt.sca(ax[1,1])
yAd=np.array(yAd)
x=np.arange(0,yAd.shape[0])
plt.plot(yAd[:,1],color=color,label=label[count],marker='v',markersize=15,linewidth=3) 
plt.fill_between(x,yAd[:,0],yAd[:,1],color=color,alpha=0.2)
plt.fill_between(x,yAd[:,1],yAd[:,2],color=color,alpha=0.2)
plt.xlabel(r'cyclical $A$',fontsize=font+3,fontweight='bold')
plt.xticks([0,1,2,3,4,5,6,7],[0.12,0.25,0.37,0.5,0.62,0.75,0.87,1.0],fontsize=font-3.5)
############################################################################################################################
plt.sca(ax[0,2])
yVs=np.array(yVs)
x=np.arange(0,yVs.shape[0])
plt.plot(yVs[:,1],color=color,label=label[count],marker='v',markersize=15,linewidth=3) 
plt.fill_between(x,yVs[:,0],yVs[:,1],color=color,alpha=0.2)
plt.fill_between(x,yVs[:,1],yVs[:,2],color=color,alpha=0.2)
############################################################################################################################
plt.sca(ax[1,2])
yVd=np.array(yVd)
x=np.arange(0,yVd.shape[0])
plt.plot(yVd[:,1],color=color,label=label[count],marker='v',markersize=15,linewidth=3) 
plt.fill_between(x,yVd[:,0],yVd[:,1],color=color,alpha=0.2)
plt.fill_between(x,yVd[:,1],yVd[:,2],color=color,alpha=0.2)
plt.xlabel(r'VDBE $\sigma$',fontsize=font+3,fontweight='bold')
plt.xticks([0,1,2,3,4,5,6,7],[0.01,0.05,0.1,0.5,1,5,10,50],fontsize=font-3)
############################################################################################################################
fig.text(0.01,0.5,'Normalized Aggregation Size',fontsize=font+3,fontweight='bold',rotation="vertical",va="center")
plt.setp(ax, ylim=(0.2,0.75))
# fig.suptitle(r'sensitivity of $\mathbf{\varepsilon}$ schedules to their model parameters with noise $\mathbf{\sigma_n=15\degree}$',fontsize=14,fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(left=0.064)
plt.savefig('/home/arash/Dropbox/sensitivity_to_model_parameter.png')
plt.savefig('sensitivity_to_model_parameter.png')

plt.show()
print('hi')