from header import SUPERVISOR
import numpy as np
from time import time as TIME
from time import ctime
import os
if __name__ == "__main__":
    iteration=20
    FinalTime=5000
    samplingPeriod=50000
    vizFlag= not True
    epoch=FinalTime//samplingPeriod+1
    # results=np.zeros((iteration,epoch))
    results=[]
    codeBeginTime=ctime(TIME()).replace(':','_')
    saved=0
    folder=os.makedirs(codeBeginTime)
    # os.chdir(codeBeginTime)
    for it in range(iteration):
        print("[+] iteration: ", it)
        col=0
        t=0
        sup=SUPERVISOR(FinalTime=5000,samplingPeriod=5)
        sup.generateRobots()
        sup.moveAll()
        os.chdir(codeBeginTime)
        while sup.getTime()<=FinalTime or True: # alert
            sup.checkCollision()
            sup.getGroundSensors()
            sup.aggregateSwarm()
            sup.getQRs()
            sup.swarmRL()
            sup.moveAll()
            if vizFlag: sup.visualize()
            # if vizFlag or saved >= 8 or True: sup.visualize()

            if sup.getTime()%10==0 and sup.getTime()-t>1 :
                results.append(sup.getStatus())

            if sup.getTime()%samplingPeriod==0 and sup.getTime()-t>1 :
                # results.append(sup.getStatus())
                col+=1
                t=sup.getTime()
                saved+=1
                with open(str(saved)+' results '+codeBeginTime+'.npy','wb') as f:
                    np.save(f,np.array(results))
                # if col%(epoch//5)==0:
                #    print("      [+] ", sup.getTime())
                #    with open('Qtable.npy','wb') as Qtable:
                #        np.save(Qtable,sup.swarm[0].Qtable)
                print("      [+] ", sup.getTime(), "epsilon of robot 0", sup.swarm[0].RLparams["epsilon"])
                with open(str(saved)+' Qtable '+codeBeginTime+' .npy','wb') as Qtable:
                    np.save(Qtable,sup.swarm[0].Qtable)
                with open(str(saved)+' rewards '+codeBeginTime+' .npy','wb') as reward:
                    np.save(reward,np.array(sup.swarm[0].rewardMemory))
                
                trainingCheck=[]
                for _ in sup.swarm[0].Qtable:
                    trainingCheck.append(any(_>0))
                print(trainingCheck)



        del sup
    with open('results'+ctime(TIME()).replace(':','_')+'.npy','wb') as f:
        np.save(f,results)