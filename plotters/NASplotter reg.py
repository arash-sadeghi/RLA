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
plt.rcParams['axes.grid'] = True

fig, ax = plt.subplots(2, 2,figsize=(7,7),sharex='col',sharey='row') 

def tnh(x,a=0.5,b=1):
    return a*np.tanh(b*x)
    # return a*np.log(b*x)
    # return a*np.sqrt(b*x)
    # return a*x+b




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
 

marker=["o","v","^","s","p","d"]

yfs=[]
yfd=[]
yAs=[]
yAd=[]
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

    if "p=" in file_ or "P=" in file_:
        yfs.append(np.mean(averagedDataMean[pointN//2-100:pointN//2]))
        yfd.append(np.mean(averagedDataMean[pointN-100:pointN]))
    elif "amplitude" in file_:
        yAs.append(np.mean(averagedDataMean[pointN//2-100:pointN//2]))
        yAd.append(np.mean(averagedDataMean[pointN-100:pointN]))
    else:
        print(colored("[-] file not found","red"),file_)
count=0
plt.sca(ax[0,0])
color="Blue"
font=11.5
plt.plot(yfs,color=color,label=label[count],marker='v',markersize=15,linewidth=3) 
plt.yticks(fontsize=font)
plt.xticks(fontsize=font)
plt.ylabel('NAS',fontsize=13,fontweight='bold')
plt.title('SS before change of environment',fontsize=font,fontweight='bold')
############################################################################################################################
plt.sca(ax[1,0])
# color="black"
plt.plot(yfd,color=color,label=label[count],marker='v',markersize=15,linewidth=3) 
plt.yticks(fontsize=font)
plt.xticks(fontsize=font)
plt.xlabel("p",fontsize=13,fontweight='bold')
plt.ylabel('NAS',fontsize=13,fontweight='bold')
plt.title('SS after change of environment',fontsize=font,fontweight='bold')
############################################################################################################################
plt.sca(ax[0,1])
# color="black"
plt.plot(yAs,color=color,label=label[count],marker='v',markersize=15,linewidth=3) 
plt.yticks(fontsize=font)
plt.xticks(fontsize=font)
plt.title('SS before change of environment',fontsize=font,fontweight='bold')
############################################################################################################################
plt.sca(ax[1,1])
# color="black"
plt.plot(yAd,color=color,label=label[count],marker='v',markersize=15,linewidth=3) 
plt.yticks(fontsize=font)
plt.xticks(fontsize=font)
plt.xlabel('A',fontsize=13,fontweight='bold')
plt.title('SS after change of environment',fontsize=font,fontweight='bold')

# plt.fill_between(x,averagedDataMean,averagedDataQ2,color=palete[count],alpha=0.2)
# plt.fill_between(x,averagedDataQ1,averagedDataMean,color=palete[count],alpha=0.2)

############################################################################################################################

plt.setp(ax, ylim=(0,1))
# fig.text(0.05,0.5, 'Normalized Aggregation Size',fontsize=12,fontweight='bold', va='center', rotation='vertical')
# fig.text(0.5,0.9, 'sensitivity of cyclical method to its model parameters',fontsize=12,fontweight='bold', ha='center', rotation='horizontal')

# fig.suptitle('10 Robots with '+str(itNum)+r' time repetition, alpha 0.1, arena size=2.82$\times$5.65 $\mathbf{m^2}$',fontsize=font,fontweight='bold')
fig.suptitle('sensitivity of cyclical method to its model parameters',fontsize=14,fontweight='bold')
# plt.subplots_adjust(left=0.125, right=0.99, top=0.921, bottom=0.07,wspace=0.1,hspace=0.15)

plt.tight_layout()
plt.savefig(ctime(TIME()).replace(':','_')+'.png')
plt.show()
print('hi')