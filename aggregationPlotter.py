import numpy as np
import matplotlib.pyplot as plt
import os

folder='Sat Oct 10 21_08_23 2020'
file_='30 results Sat Oct 10 21_08_23 2020.npy'

os.chdir(folder)
with open(file_,'rb') as Reward:
    # a=np.load(Reward)[0:290000]
    # a=np.load(Reward)[0:500000]
    a=np.load(Reward)[0:116000]

    print(np.shape(a))

pointN=1000//20
window=len(a)//pointN # 1000 is the number of points that i want to see in plot
print('window',window)
b=np.zeros((1,pointN))[0]
j=0
for i in range(pointN):
    b[i]=np.mean(a[j:j+window])
    j+=window

# plt.plot(a,color='red')

x=np.arange(0,len(b))
plt.plot(x,b,color='blue',marker='s')

plt.ylim(0,1)
plt.xlabel('Time [s] / '+str(pointN),fontsize=15,fontweight='bold')
plt.ylabel('Reward',fontsize=15,fontweight='bold')
plt.grid()
# plt.xticks(fontsize=10,fontweight='bold')
# plt.yticks(fontsize=10,fontweight='bold')
plt.show()