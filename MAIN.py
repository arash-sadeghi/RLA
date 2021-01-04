from HEADER import *
#...............................................................................................................................
def saveData(caller=None):
    print(colored("\n\t\t[+] data saved ",'yellow'))
    data2BsavedStr=["NAS","log","Qtable","rewards","eps","alpha"]
    dataType='process data'
    QtableMem[it,:,:,:]=sup.getQtables()
    data2Bsaved=[NAS,log,QtableMem,reward,eps,alpha]
    fileName=codeBeginTime+dirChangeCharacter+dataType+dirChangeCharacter+str(it)+' '+str(sup.getTime())+' s '+ctime(TIME()).replace(':','_')+' '
    for i in range(len(data2Bsaved)):
        with open(fileName+data2BsavedStr[i]+commentDividerChar+comment+'.npy','wb') as f:
            np.save(f,data2Bsaved[i])

    if caller=='iterationDone':

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
                print(colored('\t\t[-] error in saving class: '+str(E),'red'))
    elif caller=='interupt':
        with open(fileName+commentDividerChar+comment+'.sup', 'wb') as supSaver:
            try:
                pickle.dump(sup, supSaver)
            except Exception as E: 
                print(colored('\t\t[-] error in saving class: '+str(E),'red'))

#...............................................................................................................................
def keyboardInterruptHandler(signal, frame):
    saveData('interupt')
    print(colored('\t\t[+] half data saved. Time: '+str(sup.getTime())+', ratio '+str(int(sup.getTime()/FinalTime*100))+'%','green'))
    ans=input(colored('\t\t[+] continue/quit? [c/q]','green'))
    if ans=='q': exit(0)
    else:print(colored('\t\t[+] continuing','green'))
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
    iteration=1#5
    samplingPeriodSmall=10
    FinalTime=1160000
    HalfTime=FinalTime//2
    dynamic= not True
    samplingPeriod=FinalTime//5 #100 causes in 2500 files 100*5*5
    ROBN=10

    '''paramReductionMethod: possible values= 'adaptive' , 'classic' , 'adaptive united' , 'VDBE' '''
    paramReductionMethod='classic'#'VDBE' 
    
    '''commentDividerChar: plotter code will take legend what ever is after this char'''
    commentDividerChar=' x '
    
    '''vizFlag: whether the visualization computation will happen or not. if it is True,
    cod will either record arena or show you scene '''
    vizFlag=True
    
    '''globalQ: whether all robots will share one Q-table '''
    globalQ=not True
    
    '''communicate: flag for local communication '''
    communicate=not True
    
    '''record: if set True, you will get video of first iteration '''
    record=True
    
    '''Method: RL , BEECLUST '''
    method='RL'
    
    '''comment: comment to apear in file name '''
    comment='alpha check'#'alpha 0.1 VDBE sigma 1' 
    
    '''save_csv: whether tables will be saved as csv or not '''
    save_csv=True

    print(colored('[+] '+comment,'green'))
    codeBeginTime=ctime(TIME()).replace(':','_')+'_'+method+'_'+comment
    if globalQ and communicate:
        '''local and global communication cant be toghether '''
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

    ''' for saving csvs which is Q-table of robot 0 for iteration 0 '''
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
        iteration_duration=TIME()
        print(colored("\t[+] iteration: ",'blue'), it)
        t=0;tt=0;sampled=0
        sup=SUPERVISOR(ROBN,codeBeginTime,vizFlag,globalQ,record,Lx,Ly,cueRaduis,visibleRaduis,paramReductionMethod)
        sup.generateRobots()
        sup.moveAll() # to make initilazation happen
        GroundChanged=False # to make sure ground is changed only once in each iteration
        checkHealth()
        while sup.getTime()<=FinalTime:
            ''' start of main loop '''
            sup.checkCollision()
            sup.aggregateSwarm()
            if method=='RL':
                sup.getQRs()
                sup.swarmRL()
                if communicate==True:
                    sup.talk()
            sup.moveAll()
            ''' end of main loop. rest is logging '''

            '''check half time and change ground if env is dynamic'''
            if abs(HalfTime-sup.getTime())<1 and GroundChanged==False:
                GroundChanged=True
                print(colored('\t[+] half time reached','green'))
                if dynamic:
                    sup.changeGround()

            # sup.visualize(True) # for debug
            ''' main logger '''
            if sup.getTime()%samplingPeriodSmall==0 and sup.getTime()-t>1:
                '''logs specail for first iteration'''
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
                        '''iteration 0 is about to end. so release the video and turn of record
                        flag so we dont have any video attribute afterwards and save SARs as npy ans csv '''
                        for _ in range(len(videoList)):
                            videoList[_].release()
                        record=False
                        if hasattr(sup,'video'): sup.video.release()
                        SAR=np.stack(sup.swarm[0].SAR)
                        with open(codeBeginTime+dirChangeCharacter+'SAR_robot0.npy','wb') as SAR_f:
                            np.save(SAR_f,SAR)
                        np.savetxt(codeBeginTime+dirChangeCharacter+'SAR_robot0.csv',np.round(SAR), delimiter=",")

                ''' in every iteration, log the vital performance indexes with frequency of samplingPeriodSmall''' 
                NAS[it,sampled]=sup.getNAS()
                log[it,sampled,:,:]=sup.getLog()
                eps[it,sampled,:]=sup.getEps()
                alpha[it,sampled,:]=sup.getAlpha()
                reward[it,sampled,:]=sup.getReward()
                sampled+=1
                t=sup.getTime()



        
        QtableMem[it,:,:,:]=sup.getQtables()
        '''V: -1 is for that 'it' at max will be iteration-1 and after that code will exit the loop'''
        if it==iteration-1: saveData("iterationDone") 
        del sup
        print(colored("\t[+] iteration duration: ",'blue'),int(TIME()-iteration_duration))
    print(colored('[+] duration','green'),int(TIME()-t1_))
    print(colored('[+] goodbye',"green"))


