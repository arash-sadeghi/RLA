from header import *
import numpy as np
import signal

if os.name=='nt':
    dirChangeCharacter='\\'
else:
    dirChangeCharacter='/'

def keyboardInterruptHandler(signal, frame):
    global results
    with open(codeBeginTime+dirChangeCharacter+'results halfdone '+ctime(TIME()).replace(':','_')+'.npy','wb') as f:
        np.save(f,np.array(results))
        print('[+] half data saved')
    ans=input('quit? [y/n]')
    if ans=='y': exit(0)
signal.signal(signal.SIGINT, keyboardInterruptHandler)                  

if __name__ == "__main__":
    iteration=20//4
    samplingPeriodSmall=10
    FinalTime=116000
    samplingPeriod=50000
    vizFlag= not True
    epoch=FinalTime//samplingPeriod+1
    results=[]
    method='RL'
    codeBeginTime=ctime(TIME()).replace(':','_')+' '+method
    saved=0
    folder=os.makedirs(codeBeginTime)
    print('[+] '+method)
    for it in range(iteration):
        print("     [+] iteration: ", it)
        col=0
        t=0
        results_=[]
        sup=SUPERVISOR(FinalTime=5000,samplingPeriod=5,codeBeginTime=codeBeginTime)
        sup.generateRobots()
        sup.moveAll()
        while sup.getTime()<=FinalTime:

            sup.checkCollision()
            sup.getGroundSensors()
            sup.aggregateSwarm()
            if method=='RL':
                sup.getQRs()
                sup.swarmRL()

            sup.moveAll()
            sup.visualize(vizFlag)

            if sup.getTime()%samplingPeriodSmall==0 and sup.getTime()-t>1 :
                results_.append(sup.getStatus())
                t=sup.getTime()


            # if sup.getTime()%samplingPeriod==0 and sup.getTime()-t>1 :
            #     col+=1
            #     t=sup.getTime()
            #     saved+=1
            #     with open(str(saved)+' results '+codeBeginTime+'.npy','wb') as f:
            #         np.save(f,np.array(results))
            #     print("      [+] ", sup.getTime(), "epsilon of robot 0", sup.swarm[0].RLparams["epsilon"])
            #     with open(str(saved)+' Qtable '+codeBeginTime+' .npy','wb') as Qtable:
            #         np.save(Qtable,sup.swarm[0].Qtable)
            #     with open(str(saved)+' rewards '+codeBeginTime+' .npy','wb') as reward:
            #         np.save(reward,np.array(sup.swarm[0].rewardMemory))
                
            #     trainingCheck=[]
            #     for _ in sup.swarm[0].Qtable:
            #         trainingCheck.append(any(_>0))
            #     print(trainingCheck)
            signal.signal(signal.SIGINT, keyboardInterruptHandler)                  

        
        results.append(results_)



        sup.video.release()
        del sup

    os.chdir(codeBeginTime)
    with open('results '+ctime(TIME()).replace(':','_')+'.npy','wb') as f:
        np.save(f,np.array(results))
        print('[+] goodbye')

