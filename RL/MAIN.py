from header import SUPERVISOR
import numpy as np
from time import time as TIME
from time import ctime
if __name__ == "__main__":
    iteration=20
    FinalTime=5000
    samplingPeriod=5000
    vizFlag= not True
    epoch=FinalTime//samplingPeriod+1
    results=np.zeros((iteration,epoch))
    codeBeginTime=ctime(TIME()).replace(':','_')
    saved=0
    for it in range(iteration):
        print("[+] iteration: ", it)
        col=0
        t=0
        sup=SUPERVISOR(FinalTime=5000,samplingPeriod=5)
        sup.generateRobots()
        sup.moveAll()
        while sup.getTime()<=FinalTime or True: # alert
            sup.checkCollision()
            sup.getGroundSensors()
            sup.aggregateSwarm()
            sup.getQRs()
            sup.swarmRL()
            sup.moveAll()
            # if vizFlag or saved>3: sup.visualize()
            if vizFlag: sup.visualize()
            if sup.getTime()%samplingPeriod==0 and sup.getTime()-t>1 :
                # results[it,col]=sup.getStatus()
                # col+=1
                t=sup.getTime()
                # if col%(epoch//5)==0:
                #    print("      [+] ", sup.getTime())
                #    with open('Qtable.npy','wb') as Qtable:
                #        np.save(Qtable,sup.swarm[0].Qtable)
                print("      [+] ", sup.getTime(), "epsilon of robot 0", sup.swarm[0].RLparams["epsilon"])
                with open(str(saved)+' Qtable '+codeBeginTime+' .npy','wb') as Qtable:
                    np.save(Qtable,sup.swarm[0].Qtable)
                    saved+=1

        del sup
    with open('results'+ctime(TIME()).replace(':','_')+'.npy','wb') as f:
        np.save(f,results)