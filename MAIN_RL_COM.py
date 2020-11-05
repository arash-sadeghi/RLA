from header import *

def saveData():
    QtableMem[it,:,:,:]=sup.getQtables()
    rewardMem[it]=sup.getReward()
    data2Bsaved=[np.array(results),log,QtableMem,np.array(rewardMem)]
    data2BsavedStr=["results","log","Qtable","rewards"]
    fileName=codeBeginTime+dirChangeCharacter+'process data'+dirChangeCharacter+str(it)+' '+str(sup.getTime())+' s '+ctime(TIME()).replace(':','_')+' halfDone '
    for i in range(len(data2Bsaved)):
        with open(fileName+data2BsavedStr[i]+'.npy','wb') as f:
            np.save(f,data2Bsaved[i])

def keyboardInterruptHandler(signal, frame):
    saveData()
    print('[+] half data saved')
    ans=input('quit? [y/n]')
    if ans=='y': exit(0)
signal.signal(signal.SIGINT, keyboardInterruptHandler)                  


if __name__ == "__main__":
    dirChangeCharacter=DirLocManage()
    Lx=2
    Ly=4
    cueRaduis=0.7
    visibleRaduis=0.3
    iteration=20//4
    samplingPeriodSmall=10
    FinalTime=116000
    samplingPeriod=116000//100
    ROBN=10
    vizFlag= True
    globalQ=not True
    communicate=not True
    if globalQ and communicate:
        raise NameError('[-] what do you want?')
    record=False
    sampledDataNum=FinalTime//samplingPeriodSmall
    results=[]
    method='RL'
    codeBeginTime=ctime(TIME()).replace(':','_')+' '+method
    saved=0
    os.makedirs(codeBeginTime)
    os.makedirs(codeBeginTime+dirChangeCharacter+'process data')
    print('[+] '+method)
    QtableMem=np.zeros((iteration,ROBN,7,44)) ###
    log=np.zeros((iteration,sampledDataNum,ROBN,3))
    rewardMem=[[] for _ in range(iteration)]



    for it in range(iteration):
        print("     [+] iteration: ", it)
        t=0;tt=0;sampled=0
        results_=[]
        sup=SUPERVISOR(ROBN,codeBeginTime,vizFlag,globalQ,record,Lx,Ly,cueRaduis,visibleRaduis)
        sup.generateRobots()
        sup.moveAll() # to make initilazation happen
        while sup.getTime()<=FinalTime:
            checkHealth()
            sup.checkCollision()
            sup.getGroundSensors()
            sup.aggregateSwarm()
            if method=='RL':
                sup.getQRs()
                sup.swarmRL()
                if communicate==True:
                    sup.talk()
            sup.moveAll()
            sup.visualize()

            if sup.getTime()%samplingPeriodSmall==0 and sup.getTime()-t>1:
                results_.append(sup.getStatus())
                log[it,sampled,:,:]=sup.getlog()
                sampled+=1
                t=sup.getTime()

            if sup.getTime()%samplingPeriod==0 and sup.getTime()-tt>1 :
                tt=sup.getTime()
                saveData()
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

