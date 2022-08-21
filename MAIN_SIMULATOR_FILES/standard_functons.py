from sys import warnoptions
from warnings import simplefilter
import random as rnd
from termcolor import colored 
from psutil import disk_usage
from time import time as TIME
import os
import numpy as np
# ------------------------------------------------------------------------------------------------------------------------------

def set_seed(seed):
    if seed == "random":
        seed=int(TIME()%1000)
        print(colored("[+] seed: ","red"),seed)
    else:
        print(colored("\n\n[!]>>>>>>>>>>>> SEED GIVEN. NOT RANDOM<<<<<<<<<<<<<<\n\n","red"))
    
    rnd.seed(seed)
    return seed

# ------------------------------------------------------------------------------------------------------------------------------
def warningSupress():
    if not warnoptions:
        simplefilter("ignore")
# ------------------------------------------------------------------------------------------------------------------------------
def checkHealth():
    MinDiskCap=30
    # health=shutil.disk_usage('/')
    # if health[-1]/(2**30)<=5:
    #     raise NameError('[-] disk is getting full')
    # disk='/media/arash/7bee828d-5758-452e-956d-aca000de2c81'
    disk=os.getcwd()

    try:
        hdd=disk_usage(disk)
        total,used,free=hdd.total / (2**30),hdd.used / (2**30),hdd.free / (2**30)
        if free<MinDiskCap:
                print (colored('[-] disk is almost full',"red"))
                exit(1)

    except Exception as E:
        print(colored("[+] Error not in "+disk+" ERROR:"+E,'red'))

    s=__file__
    s=s.replace(__name__,"")
    s=s.replace('.py',"")
    if '\\' in s:
        s=s.replace('\\','/')
    hdd2=disk_usage(s)
    total2,used2,free2=hdd2.total / (2**30),hdd2.used / (2**30),hdd2.free / (2**30)

    if free2<MinDiskCap:
            print (colored('[-] disk is almost full',"red"))
            exit(1)
    print(colored('\t[+] Disk health checked. free2: ','green'),str(int(free2)),' GB')
# ------------------------------------------------------------------------------------------------------------------------------
def m2px(inp):
    return int(inp*512/2)
# ------------------------------------------------------------------------------------------------------------------------------
def px2m(inp):
    return inp*2/512
# ------------------------------------------------------------------------------------------------------------------------------
def RotStandard(inp):
    while inp<0: inp+=360
    while inp>360: inp-=360
    return inp
# ------------------------------------------------------------------------------------------------------------------------------
def dist(delta):
    if np.size(delta)>2:
        # return np.sqrt(np.square(delta[:,0])+np.square(delta[:,1]))
        return np.linalg.norm(delta,axis=1)
    else:
        return np.linalg.norm(delta)
        # return np.sqrt(np.square(delta[0])+np.square(delta[1]))
#...............................................................................................................................
def clearTerminal(): 
    ''' check and make call for specific operating system '''
    # call('clear' if os.name =='posix' else 'cls')
    if os.name=='nt':os.system('cls')
    else:os.system('clear')

#...............................................................................................................................
def str2bool(string):
    if string == "True":
        return True
    elif string == "False":
        return False
    else:
        raise NameError("[-] invalid Flag value")
#...............................................................................................................................
def param_str2val(params):

    params.flags.dynamic = str2bool(params.flags.dynamic)
    params.flags.showFrames = str2bool(params.flags.showFrames)
    params.flags.record = str2bool(params.flags.record)
    params.flags.globalQ = str2bool(params.flags.globalQ)
    params.flags.communicate = str2bool(params.flags.communicate)
    params.flags.save_csv = str2bool(params.flags.save_csv)
    params.flags.save_tables_videos = str2bool(params.flags.save_tables_videos)

    params.vals.Lx = float(params.vals.Lx)
    params.vals.Ly = float(params.vals.Ly)
    params.vals.cueRaduis = float(params.vals.cueRaduis)
    params.vals.visibleRaduis = float(params.vals.visibleRaduis)
    params.vals.iteration = int(params.vals.iteration)
    params.vals.samplingPeriodSmall = int(params.vals.samplingPeriodSmall)
    params.vals.FinalTime = int(params.vals.FinalTime)
    params.vals.HalfTime = int(params.vals.HalfTime)
    params.vals.ROBN = int(params.vals.ROBN)
    params.vals.PRMparameter = float(params.vals.PRMparameter)
    params.vals.noise = float(params.vals.noise)
    params.vals.seed_value = int(params.vals.seed_value) if not params.vals.seed_value == "random" else params.vals.seed_value

    return params