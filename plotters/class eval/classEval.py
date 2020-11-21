import pickle
import os
import numpy as np

'''
this code will plot all .npy files in its own location
so do not put any .npy file in its location if you dont
want it to be plotted
'''

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


allFiles=os.listdir()
tobeDeleted=[]
for files in allFiles:
    if os.path.splitext(files)[1]!='.hi':
        tobeDeleted.append(files)

for i in tobeDeleted:
    allFiles.remove(i)
print('[+] allFiles',allFiles)

obj=[]
for fname in allFiles:
    with open(fname,'rb') as f:
        obj.append(pickle.load(f))

print('\n[+] adaptive eps ',np.mean(list(map(lambda x: np.mean(x.epsilon),obj[0].swarm))))
print('[+] classic eps',np.mean(list(map(lambda x: x.RLparams['epsilon'],obj[1].swarm))))
print('\n[+] adaptive action coverage ',np.mean(list(map(lambda x: (x.QtableCheck==1).sum()/np.size(x.Qtable),obj[0].swarm))))
print('[+] classic action coverage',np.mean(list(map(lambda x: (x.QtableCheck==1).sum()/np.size(x.Qtable),obj[1].swarm))))

'''
old action coverage 0.61
my action coverage 0.68
old final eps 0.52
my final eps 0.89
'''
# np.round(obj[0].swarm[0].epsilon+0.1,4)

print('hi')