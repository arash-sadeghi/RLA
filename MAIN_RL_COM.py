from header import *
import numpy as np
import signal
from varname import nameof

if os.name=='nt':
    dirChangeCharacter='\\'
else:
    dirChangeCharacter='/'

def keyboardInterruptHandler(signal, frame):
    QtableMem[it,:,:,:]=sup.getQtables()
    rewardMem[it]=sup.getReward()
    data2Bsaved=[np.array(results),log,QtableMem,np.array(rewardMem)]
    data2BsavedStr=["results","log","Qtable","rewards"]
    fileName=codeBeginTime+dirChangeCharacter+ctime(TIME()).replace(':','_')+' halfDone '
    for i in range(len(data2Bsaved)):
        with open(fileName+' '+data2BsavedStr[i]+'.npy','wb') as f:
            np.save(f,data2Bsaved[i])

    print('[+] half data saved')
    ans=input('quit? [y/n]')
    if ans=='y': exit(0)
signal.signal(signal.SIGINT, keyboardInterruptHandler)                  

if __name__ == "__main__":
    iteration=20//4
    samplingPeriodSmall=10
    FinalTime=20 #116000
    samplingPeriod=20000#50000
    ROBN=10
    vizFlag=not  True
    globalQ=True
    record=False
    sampledDataNum=FinalTime//samplingPeriodSmall
    results=[]
    method='RL'
    codeBeginTime=ctime(TIME()).replace(':','_')+' '+method
    saved=0
    folder=os.makedirs(codeBeginTime)
    print('[+] '+method)
    QtableMem=np.zeros((iteration,ROBN,7,44)) ###
    log=np.zeros((iteration,sampledDataNum,ROBN,3))
    rewardMem=[[] for _ in range(ROBN)]



    for it in range(iteration):
        print("     [+] iteration: ", it)
        col=0
        t=0
        tt=0
        results_=[]
        sup=SUPERVISOR(ROBN=ROBN,codeBeginTime=codeBeginTime,vizFlag=vizFlag,globalQ=globalQ,record=record)
        sup.generateRobots()
        sup.moveAll()
        sampled=0
        while sup.getTime()<=FinalTime:

            sup.checkCollision()
            sup.getGroundSensors()
            sup.aggregateSwarm()
            if method=='RL':
                sup.getQRs()
                sup.swarmRL()
                if globalQ: sup.talk()

            sup.moveAll()
            sup.visualize()

            if sup.getTime()%samplingPeriodSmall==0 and sup.getTime()-t>1:
                results_.append(sup.getStatus())
                log[it,sampled,:,:]=sup.getlog()
                t=sup.getTime()
                sampled+=1


            # if sup.getTime()%samplingPeriod==0 and sup.getTime()-tt>1 :
            #     tt=sup.getTime()
            #     with open(str(it)+' Qtable '+codeBeginTime+' .npy','wb') as Qtable:
            #         np.save(Qtable,sup.swarm[0].Qtable)
            #     with open(str(saved)+' rewards '+codeBeginTime+' .npy','wb') as reward:
            #         np.save(reward,np.array(sup.swarm[0].rewardMemory))
                
            #     trainingCheck=[]
            #     for _ in sup.swarm[0].Qtable:
            #         trainingCheck.append(any(_>0))
            #     print(trainingCheck)

            signal.signal(signal.SIGINT, keyboardInterruptHandler)                  

        
        results.append(results_)
        QtableMem[it,:,:,:]=sup.getQtables()
        rewardMem[it]=sup.getReward()
        if record: sup.video.release()
        del sup

    os.chdir(codeBeginTime)
    data2Bsaved=[np.array(results),log,QtableMem,np.array(rewardMem)]
    data2BsavedStr=["results","log","Qtable","rewards"]
    for i in range(len(data2Bsaved)):
        with open(data2BsavedStr[i]+' '+ctime(TIME()).replace(':','_')+'.npy','wb') as f:
            np.save(f,data2Bsaved[i])
    print('[+] goodbye')

