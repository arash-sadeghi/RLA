import numpy as np
import os
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
# print(os.listdir())
# select=int(input('which one?'))
# f=os.listdir()[select]
f="0 1160001 s Thu Dec  3 23_38_33 2020 Qtable x alpha 0.5 damp ratio 0.999 image checking of robots.npy"
with open(f,'rb') as f_:
    ar=np.load(f_)
np.savetxt(f+".csv", np.round(ar[0,0],2), delimiter=",")