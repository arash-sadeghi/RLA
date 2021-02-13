""" import os
dirs=os.listdir()
for dir in dirs:
    if "%" in dir:
        print("EXECUTING",dir)
        os.system("python3 "+dir+"/MAIN.py") """
import os

def DirLocManage(returnchar=False):
    ''' with this segment code is callable from any folder '''
    if os.name=='nt':
        dirChangeCharacter='\\'
    else:
        dirChangeCharacter='/'
    if returnchar==False:
        scriptLoc=__file__
        for i in range(len(scriptLoc)):
            # if '/' in scriptLoc[-i-2:-i]: # in running
            if dirChangeCharacter in scriptLoc[-i-2:-i]: # in debuging
                scriptLoc=scriptLoc[0:-i-2]
                break
        # print('[+] code path',scriptLoc)
        os.chdir(scriptLoc)
    return dirChangeCharacter
# DirLocManage()  
sigs=[50,150,100,200,500]
for sig in sigs:
    print("------------------EXECUTING sigma",sig)
    os.system("python3 MAIN.py "+str(sig))
