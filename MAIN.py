from HEADER import *
#...............................................................................................................................
def saveData():
    data2BsavedStr=["NAS","log","Qtable","rewards","eps","alpha"]
    dataType='process data'
    QtableMem[it,:,:,:]=sup.getQtables()
    data2Bsaved=[NAS,log,QtableMem,reward,eps,alpha]
    fileName=codeBeginTime+dirChangeCharacter+dataType+dirChangeCharacter+str(it)+' '+paramReductionMethod+' '+str(sup.getTime())+' s '+ctime(TIME()).replace(':','_')+' '
    for i in range(len(data2Bsaved)):
        with open(fileName+data2BsavedStr[i]+'.npy','wb') as f:
            np.save(f,data2Bsaved[i])
    with open(fileName+' sup class.sup', 'wb') as supSaver:
        pickle.dump(sup, supSaver)
#...............................................................................................................................
def keyboardInterruptHandler(signal, frame):
    saveData()
    print('[+] half data saved')
    ans=input('\t[+] continue/quit? [c/q]')
    if ans=='q': exit(0)
signal.signal(signal.SIGINT, keyboardInterruptHandler)                  
#...............................................................................................................................
def clearTerminal(): 
    # check and make call for specific operating system 
    # call('clear' if os.name =='posix' else 'cls')
    if os.name=='nt':os.system('cls')
    else:os.system('clear')
#...............................................................................................................................

if __name__ == "__main__" or True:
    t1_=TIME()
    # clearTerminal()
    ''' call wsential functions '''
    warningSupress()
    dirChangeCharacter=DirLocManage()

    ''' parameter value assigning '''
    Lx=2
    Ly=4
    cueRaduis=0.7
    visibleRaduis=0.3
    iteration=5 
    samplingPeriodSmall=10
    FinalTime=116000*10#3 
    HalfTime=FinalTime//2
    dynamic=True
    samplingPeriod=FinalTime//5 #100 causes in 2500 files 100*5*5
    ROBN=10#10
    paramReductionMethod='classic' # possible values= 'adaptive' , 'classic' , 'adaptive united'
    print(colored('[+] adaptive shut with modofication','green'))
    vizFlag=not True
    globalQ=not True
    communicate=not True
    if globalQ and communicate:
        raise NameError('[-] what do you want?')
    record=False
    method='RL'
    comment='_'
    codeBeginTime=ctime(TIME()).replace(':','_')+'_'+method+'_'+comment

    ''' preparing dirs '''
    os.makedirs(codeBeginTime)
    os.makedirs(codeBeginTime+dirChangeCharacter+'process data')

    ''' for saving image of mats '''
    os.makedirs(codeBeginTime+dirChangeCharacter+'ims') 
    imsName=["delta","deltaDot","DELTA","epsilon","QtableCheck"]
    for _ in imsName:
        if _=="DELTA" and os.name=='nt': _+='_' # in windows this wierd thing happens
        os.makedirs(codeBeginTime+dirChangeCharacter+'ims'+dirChangeCharacter+_)

    ''' save parameters into a file '''
    paramDict={'Lx':Lx , 'Ly':Ly , 'cueRaduis':cueRaduis , 'visibleRaduis':visibleRaduis , 'iteration':iteration , 'samplingPeriodSmall':samplingPeriodSmall , \
        'FinalTime':FinalTime , 'HalfTime':HalfTime , 'dynamic':dynamic , 'samplingPeriod':samplingPeriod , 'ROBN':ROBN , 'paramReductionMethod':paramReductionMethod , 'vizFlag':vizFlag , 'globalQ':globalQ , \
            'communicate':communicate , 'record':record , 'method':method}
    with open(codeBeginTime+dirChangeCharacter+'params.txt','w') as paramfile :
        paramfile.write(str(paramDict))


    ''' initilization '''
    sampledDataNum=FinalTime//samplingPeriodSmall
    saved=0
    print(colored('[+] '+method,'green'))
    print(colored('[+] press ctrl+c for saving data asynchronously','green'))
    QtableMem=np.zeros((iteration,ROBN,7,44)) ###
    log=np.zeros((iteration,sampledDataNum,ROBN,3))
    eps=np.zeros((iteration,sampledDataNum,ROBN))
    alpha=np.zeros((iteration,sampledDataNum,ROBN))
    reward=np.zeros((iteration,sampledDataNum,ROBN))
    NAS=np.zeros((iteration,sampledDataNum))



    for it in range(iteration):
        print(colored("\t[+] iteration: ",'blue'), it)
        t=0;tt=0;sampled=0
        sup=SUPERVISOR(ROBN,codeBeginTime,vizFlag,globalQ,record,Lx,Ly,cueRaduis,visibleRaduis,paramReductionMethod)
        sup.generateRobots()
        sup.moveAll() # to make initilazation happen
        GroundChanged=False # to make sure ground is changed only once in each iteration
        checkHealth()
        while sup.getTime()<=FinalTime:
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
                NAS[it,sampled]=sup.getNAS()
                log[it,sampled,:,:]=sup.getLog()
                eps[it,sampled,:]=sup.getEps()
                alpha[it,sampled,:]=sup.getAlpha()
                reward[it,sampled,:]=sup.getReward()
                if it==0 and sampled % 100==0:
                    deltaDotTemp=np.copy(sup.swarm[0].deltaDot)
                    deltaDotTemp[deltaDotTemp>=0]=0 # positives will be white
                    deltaDotTemp[deltaDotTemp<0]=255 # negative will be black
                    imsMat=[sup.swarm[0].delta*255,deltaDotTemp,sup.swarm[0].DELTA*255,sup.swarm[0].epsilon*255,sup.swarm[0].QtableCheck*255]
                    for count_,v in enumerate(imsMat):
                        cv.imwrite(codeBeginTime+dirChangeCharacter+'ims'+dirChangeCharacter+imsName[count_]+dirChangeCharacter+str(sup.getTime())+'.png',255-v)
                ''' dont forget the effect of state 0'''
                sampled+=1
                t=sup.getTime()

            if abs(HalfTime-sup.getTime())<1 and GroundChanged==False:
                GroundChanged=True
                print(colored('\t[+] half time reached','green'))
                if dynamic:
                    sup.changeGround()

        
        QtableMem[it,:,:,:]=sup.getQtables()
        if record: sup.video.release()
        saveData()
        del sup
    print('duration',TIME()-t1_)
    print('[+] goodbye')


