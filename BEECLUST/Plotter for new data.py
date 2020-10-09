import matplotlib.pyplot as plt 
import numpy as np
from statistics import median
endTime=5000
TimeAverageInterval=20
colors=['red']
f=open('results.npy','rb')
data=np.load(f)
f.close()

dataShape=np.shape(data)
dataTimeAveraged=np.zeros((dataShape[0],dataShape[1]//TimeAverageInterval))

for row in range(dataShape[0]):
    # for col in range(dataShape[1]-1): # -1
    #     dataTimeAveraged[row,col//TimeAverageInterval]=np.mean(data[row,col:col+TimeAverageInterval])
    col1=0
    col2=0
    while col2 < dataShape[1]-1:
        dataTimeAveraged[row,col1]=np.mean(data[row,col2:col2+TimeAverageInterval])
        col1+=1
        col2+=TimeAverageInterval

q1=np.zeros((1,np.shape(dataTimeAveraged)[1]))[0]
q2=np.zeros((1,np.shape(dataTimeAveraged)[1]))[0]
q3=np.zeros((1,np.shape(dataTimeAveraged)[1]))[0]

for col in range(np.shape(dataTimeAveraged)[1]):
    q1[col]=np.percentile(dataTimeAveraged[:,col],25)
    q2[col]=np.percentile(dataTimeAveraged[:,col],50)
    q3[col]=np.percentile(dataTimeAveraged[:,col],75)
X=np.linspace(0,endTime,len(q1),True)
plt.plot(X,q2,color=colors[0],label='BEECLUST with new OOP simulator')
plt.fill_between(X,q2,q1,color=colors[0],alpha=0.25)
plt.fill_between(X,q3,q2,color=colors[0],alpha=0.25)

plt.ylim(0,1)
plt.xlim(X[0],X[-1])
plt.xticks(fontsize=15,fontweight='bold')
plt.yticks(fontsize=15,fontweight='bold')
plt.xlabel('Time [s]',fontsize=15,fontweight='bold')
plt.ylabel('Normalized Aggregation size',fontsize=15,fontweight='bold')
plt.legend(fontsize=15)
plt.savefig('results.png')
plt.show()
print("hi")