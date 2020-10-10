import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('Fri Oct  9 23_32_58 2020')
# print(os.listdir())
with open('15 rewards Fri Oct  9 23_32_58 2020 .npy','rb') as Reward:
    a=np.load(Reward)

pointN=1000//2
window=len(a)//pointN # 1000 is the number of points that i want to see in plot
b=np.zeros((1,pointN))[0]
j=0
for i in range(pointN):
    b[i]=np.mean(a[j:j+window])
    j+=window

plt.plot(b)
plt.xlabel('Time [s] / '+str(pointN),fontsize=15,fontweight='bold')
plt.ylabel('Reward',fontsize=15,fontweight='bold')
plt.xticks(fontsize=10,fontweight='bold')
plt.yticks(fontsize=10,fontweight='bold')
plt.show()