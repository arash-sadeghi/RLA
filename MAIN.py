from header import SUPERVISOR
import numpy as np
from time import time as TIME
from time import ctime
if __name__ == "__main__":
    iteration=20
    FinalTime=5000
    samplingPeriod=5
    vizFlag=True
    epoch=FinalTime//samplingPeriod+1
    results=np.zeros((iteration,epoch))
    for it in range(iteration):
        print("[+] iteration: ", it)
        col=0
        t=0
        sup=SUPERVISOR(FinalTime=5000,samplingPeriod=5)
        sup.generateRobots()
        while sup.getTime()<=FinalTime:
            if vizFlag: sup.visualize()
            sup.checkCollision()
            sup.getGroundSensors()
            sup.aggregateSwarm()
            sup.moveAll()
            if sup.getTime()%samplingPeriod==0 and sup.getTime()-t>1 :
                results[it,col]=sup.getStatus()
                col+=1
                t=sup.getTime()
                if col%(epoch//5)==0:
                   print("      [+] ", sup.getTime())

        del sup
    with open('results'+ctime(TIME()).replace(':','_')+'.npy','wb') as f:
        np.save(f,results)