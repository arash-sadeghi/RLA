from HEADER import *
#...............................................................................................................................
def saveData(caller=None):
    print(colored("[+] data saved ",'yellow'))
    data2BsavedStr=["NAS","log","Qtable","rewards","eps","alpha"]
    dataType='process data'
    QtableMem[it,:,:,:]=sup.getQtables()
    data2Bsaved=[NAS,log,QtableMem,reward,eps,alpha]
    fileName=codeBeginTime+dirChangeCharacter+dataType+dirChangeCharacter+str(it)+' '+str(sup.getTime())+' s '+ctime(TIME()).replace(':','_')+' '
    for i in range(len(data2Bsaved)):
        with open(fileName+data2BsavedStr[i]+commentDividerChar+comment+'.npy','wb') as f:
            np.save(f,data2Bsaved[i])

    if caller=='itDone':

        with open(fileName+commentDividerChar+comment+'.sup', 'wb') as supSaver:
            try:
                '''
                del sup.video
                this is not needed since in iterations other than it=0,
                sup wont record and thus wont have video recorder
                '''
                del sup.pos_getter
                del sup.rot_getter
                pickle.dump(sup, supSaver)
            except Exception as E: 
                print(colored('[-] error in saving class: '+str(E),'red'))
    elif caller=='interupt':
        with open(fileName+commentDividerChar+comment+'.sup', 'wb') as supSaver:
            try:
                pickle.dump(sup, supSaver)
            except Exception as E: 
                print(colored('[-] error in saving class: '+str(E),'red'))

#...............................................................................................................................
def keyboardInterruptHandler(signal, frame):
    saveData('interupt')
    print(colored('[+] half data saved. Time: '+str(sup.getTime()),'red'))
    ans=input('\t[+] continue/quit? [c/q]')
    if ans=='q': exit(0)
signal.signal(signal.SIGINT, keyboardInterruptHandler)                  
#...............................................................................................................................
def clearTerminal(): 
    ''' check and make call for specific operating system '''
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
    dynamic= not True
    samplingPeriod=FinalTime//5 #100 causes in 2500 files 100*5*5
    ROBN=10#10
    paramReductionMethod='classic' # possible values= 'adaptive' , 'classic' , 'adaptive united'
    commentDividerChar=' x '
    vizFlag=not True
    globalQ=not True
    communicate=not True
    record=not True
    method='RL'
    comment='test2 with alpha 0.5 eps damp 0.999 static' 
    save_csv=True
    ''' <><><<><><><><><><><<><><><<> '''
    print(colored('[+] '+comment,'green'))
    codeBeginTime=ctime(TIME()).replace(':','_')+'_'+method+'_'+comment
    if globalQ and communicate:
        raise NameError('[-] what do you want?')

    ''' preparing dirs '''
    os.makedirs(codeBeginTime)
    os.makedirs(codeBeginTime+dirChangeCharacter+'process data')

    imsName=["delta","deltaDot","DELTA","epsilon","QtableCheck","QtableRob0"]

    ''' save parameters into a file '''
    paramDict={'Lx':Lx , 'Ly':Ly , 'cueRaduis':cueRaduis , 'visibleRaduis':visibleRaduis , 'iteration':iteration , 'samplingPeriodSmall':samplingPeriodSmall , \
        'FinalTime':FinalTime , 'HalfTime':HalfTime , 'dynamic':dynamic , 'samplingPeriod':samplingPeriod , 'ROBN':ROBN , 'paramReductionMethod':paramReductionMethod , 'vizFlag':vizFlag , 'globalQ':globalQ , \
            'communicate':communicate , 'record':record , 'method':method}
    with open(codeBeginTime+dirChangeCharacter+'params.txt','w') as paramfile :
        paramfile.write(str(paramDict))

    ''' for saving csvs '''
    os.makedirs(codeBeginTime+dirChangeCharacter+'csvs') 


    ''' initilization '''
    sampledDataNum=FinalTime//samplingPeriodSmall
    saved=0
    print(colored('[+] '+method,'green'))
    print(colored('[+] press ctrl+c for saving data asynchronously','green'))
    QtableMem=np.zeros((iteration,ROBN,7,44)) ##### caviat
    log=np.zeros((iteration,sampledDataNum,ROBN,3))
    eps=np.zeros((iteration,sampledDataNum,ROBN))
    alpha=np.zeros((iteration,sampledDataNum,ROBN))
    reward=np.zeros((iteration,sampledDataNum,ROBN))
    NAS=np.zeros((iteration,sampledDataNum))
    strip=np.arange(0,44) ##### caviat: table dimentions pre known
    strip[strip%2==0]=0
    strip[strip%2==1]=255
    tableImSize=(7*20,44*20)[::-1] ##### caviat: table dimentions pre known

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    FPS=1
    videoList=[]
    for _ in range(len(imsName)):
        videoList.append(cv.VideoWriter(codeBeginTime+DirLocManage(returnchar=True)+imsName[_]+'.mp4',fourcc, FPS, tableImSize,True))


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
            sup.aggregateSwarm()
            if method=='RL':
                sup.getQRs()
                sup.swarmRL()
                if communicate==True:
                    sup.talk()
            sup.moveAll()

            if abs(HalfTime-sup.getTime())<1 and GroundChanged==False:
                GroundChanged=True
                print(colored('\t[+] half time reached','green'))
                if dynamic:
                    sup.changeGround()

            # sup.visualize() # for debug
            if sup.getTime()%samplingPeriodSmall==0 and sup.getTime()-t>1:
                if it==0: # save these in the first iteration only
                    sup.visualize() # moved for less file size
                    if sampled % 100==0:
                        deltaDotTemp=np.copy(sup.swarm[0].deltaDot)
                        deltaDotTemp[deltaDotTemp>=0]=0 # positives will be white
                        deltaDotTemp[deltaDotTemp<0]=255 # negative will be black
                        QtableRob0=sup.getQtables()[0]
                        if save_csv:
                            np.savetxt(codeBeginTime+dirChangeCharacter+'csvs'+dirChangeCharacter+str(sup.getTime())+".csv", np.round(QtableRob0,2), delimiter=",")
                        imsMat=[sup.swarm[0].delta*255,deltaDotTemp,sup.swarm[0].DELTA*255,sup.swarm[0].epsilon*255,sup.swarm[0].QtableCheck*255,QtableRob0]

                        for count_,v in enumerate(imsMat):
                            v=np.minimum(v,np.ones(v.shape)*255)
                            v[0]=strip
                            im=255-v
                            im.astype(int)
                            canvas=np.zeros((im.shape[0],im.shape[1],3))
                            canvas[:,:,0]=im
                            canvas[:,:,1]=im
                            canvas[:,:,2]=im
                            canvas=cv.resize(canvas,tableImSize)
                            videoList[count_].write(np.uint8(canvas))

                    elif abs(FinalTime-sup.getTime())<1:
                        for _ in range(len(videoList)):
                            videoList[_].release()
                        record=False
                        if hasattr(sup,'video'): sup.video.release()
                        SAR=np.stack(sup.swarm[0].SAR)
                        with open(codeBeginTime+dirChangeCharacter+'SAR_robot0.npy','wb') as SAR_f:
                            np.save(SAR_f,SAR)
                        np.savetxt(codeBeginTime+dirChangeCharacter+'SAR_robot0.csv',np.round(SAR), delimiter=",")

                NAS[it,sampled]=sup.getNAS()
                log[it,sampled,:,:]=sup.getLog()
                eps[it,sampled,:]=sup.getEps()
                alpha[it,sampled,:]=sup.getAlpha()
                reward[it,sampled,:]=sup.getReward()
                sampled+=1
                t=sup.getTime()



        
        QtableMem[it,:,:,:]=sup.getQtables()
        if it==iteration-1: saveData("itDone") # -1 is for that it at max will be iteration-1
        del sup
    print('duration',TIME()-t1_)
    print('[+] goodbye')


