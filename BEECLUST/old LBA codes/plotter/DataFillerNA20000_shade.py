import matplotlib.pyplot as plt 
import numpy as np
from statistics import median
# import seaborn as sns
# import pandas as pd
import os
# from shutil import rmtree

######################################################################################## 10 roe4ots ###########################################################################################
def ten(FolderName,numel):
    lst=[]

    os.chdir(FolderName)
    expdir=sorted(os.listdir())
    print(sorted(expdir))

    for c,v in enumerate(expdir):
        if os.path.isdir(v):
            # os.chdir(v+"\\controllers\\SupervisorUltra")
            os.chdir(v)

            casedirx=sorted([ name for name in os.listdir() if os.path.isdir(name) ]) #os.listdir()
            data=[]
            casedir=[]
            for _ in range(len(casedirx)):
                if casedirx[_][-4:] == 'True':
                    casedir.append(casedirx[_])
            # casedir=[casedir[_] if casedir[_][-7:-4] == 'rue' for _ in range(len(casedir))]
            # print(">><<",casedir,casedirx[_-1][-4:])
            for cc,vv in enumerate(casedir):
                # print(">>",vv)
                os.chdir(vv)
                datadir=os.listdir()
                if len(datadir) >= 2:
                    for val in datadir:
                        if val[-7:-4] == 'rue':
                            txtf=val
                    # txtf= datadir[0] if datadir[0][-7:-4] == 'rue' else datadir[1]
                    # print(os.getcwd(),txtf)
                    f=open(txtf,'r')
                    # print(txtf)
                    slstr=f.read()
                    f.close()
                    sl=slstr.split(",")
                    
                    for ccc,vvv in enumerate(sl):   
                        dig=''
                        for i in vvv:
                            if i.isdigit()==True:
                                dig=dig+i
                        if dig != '' : sl[ccc]=float(dig)
                    # sl.insert(0,0)
                    # sl.insert(0,0)
                    del sl[numel:]
                    data.append(np.array(sl))
                    
                    os.chdir("..")
                else :
                    os.chdir("..")
                    # rmtree(vv) #####
                
            lst.append(np.array(data))


            # os.chdir("..");os.chdir("..");os.chdir("..")
            # os.chdir("..");os.chdir("..")

            os.chdir("..")

    # print(lst[0])
    lst=np.array(lst)
    # print(lst[0,0],type(lst[0,0]),np.shape(lst[0,0]))
    # exit()

    rownum=min(min([len(_) for _ in lst]),20)
    os.chdir("..")
    return lst,rownum
######################################################################################## 10 roe4ots ###########################################################################################
