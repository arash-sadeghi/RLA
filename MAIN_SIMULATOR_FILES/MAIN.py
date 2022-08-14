from HEADER import SUPERVISOR
from standard_functons import set_seed, warningSupress, checkHealth, param_str2val
from termcolor import colored
import json
import signal # used in main
import pickle # used in main
import os
from munch import DefaultMunch
from time import time as TIME
from time import ctime
import numpy as np
import cv2 as cv
#...............................................................................................................................
def saveData(caller=None):
    print(colored("\n\t\t[+] data saved ",'yellow'))
    dataType='process data'
    if params.vals.method=="RL":
        QtableMem[it,:,:,:]=sup.getQtables()
    data2BsavedStr=["NAS","log","Qtable","rewards","eps","alpha"]
    data2Bsaved=[NAS,log,QtableMem,reward,eps,alpha]
    fileName=os.path.join(codeBeginTime , dataType , f"seed_{seed}_it_{str(it)}_sim_time_{str(sup.getTime())}_date_{ctime(TIME()).replace(':','_').replace(' ','_')}")
    for i in range(len(data2Bsaved)):
        with open(fileName+data2BsavedStr[i]+params.vals.commentDividerChar+comment+'.npy','wb') as f:
            np.save(f,data2Bsaved[i])

    if caller=='iterationDone':

        with open(fileName+params.vals.commentDividerChar+comment+'.sup', 'wb') as supSaver:
            try:
                if hasattr(sup,'video'):
                    del sup.video
                del sup.NASfunction
                pickle.dump(sup, supSaver)
            except Exception as E: 
                print(colored('\t\t[-] error in saving class: '+str(E),'red'))
    elif caller=='interupt':
        ''' you should not delete any thing here, for code will continue after here'''
        with open(fileName+params.vals.commentDividerChar+comment+'.sup', 'wb') as supSaver:
            try:
                pickle.dump(sup, supSaver)
            except Exception as E: 
                print(colored('\t\t[-] error in saving class: '+str(E),'red'))
#...............................................................................................................................
def keyboardInterruptHandler(signal, frame):
    saveData('interupt')
    print(colored('\t\t[+] half data saved. Time: '+str(sup.getTime())+', ratio '+str(int(sup.getTime()/params.vals.FinalTime*100))+'%','green'))
    ans=input(colored('\t\t[+] continue/quit? [c/q]','green'))
    if ans=='q': exit(0)
    else:print(colored('\t\t[+] continuing','green'))
signal.signal(signal.SIGINT, keyboardInterruptHandler)                  
#...............................................................................................................................
def LOG():
    global sup,it,sampled,codeBeginTime,videoList,record,showFrames,SAR,t
    global NAS,log,eps,alpha,reward 

    sup.visualize()
    if sup.getTime()% params.vals.samplingPeriodSmall==0 and sup.getTime()-t>1:
        '''logs specail for first iteration'''
        if params.vals.method=='RL':
            if it==0: # save these in the first iteration only
                if sampled % 100==0 :
                    # sup.visualize() # moved for less file size >>>>>>>>>>>>>> alert
                    '''save csvs '''
                    if params.flags.save_csv:
                        QtableRob0=sup.getQtables()[0]
                        np.savetxt(
                            os.path.join(codeBeginTime,'csvs',str(sup.getTime())+".csv"),
                            np.round(QtableRob0,2),
                            delimiter=",")

                    '''save tables videos '''
                    if params.flags.save_tables_videos:
                        imsMat=[sup.swarm[0].epsilon*255,sup.swarm[0].QtableCheck*255,QtableRob0]
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

                elif abs(params.vals.FinalTime-sup.getTime())<1:
                    '''iteration 0 is about to end. so release the video and turn of record
                    flag so we dont have any video attribute afterwards and save SARs as npy ans csv '''
                    if params.vals.save_tables_videos:
                        for _ in range(len(videoList)):
                            videoList[_].release()
                    record=False
                    showFrames=False
                    if hasattr(sup,'video'): sup.video.release()
                    ''' save SAR 
                    SAR=np.stack(sup.swarm[0].SAR)
                    with open(codeBeginTime+dirChangeCharacter+'SAR_robot0.npy','wb') as SAR_f:
                        np.save(SAR_f,SAR)
                    np.savetxt(codeBeginTime+dirChangeCharacter+'SAR_robot0.csv',np.round(SAR), delimiter=",")
                    '''
            NAS[it,sampled]=sup.getNAS()
            # log[it,sampled,:,:]=sup.getLog()
            eps[it,sampled,:]=sup.getEps()
            # alpha[it,sampled,:]=sup.getAlpha()
            reward[it,sampled,:]=sup.getReward()
        elif params.vals.method=='BEECLUST' or params.vals.method=='LBA':
            if it==0: # save these in the first iteration only
                sup.visualize() 
                if abs(params.vals.FinalTime-sup.getTime())<1:
                    '''iteration 0 is about to end. so release the video and turn of record
                    flag so we dont have any video attribute afterwards and save SARs as npy ans csv '''
                    record=False
                    showFrames=False
                    if hasattr(sup,'video'): sup.video.release()
            NAS[it,sampled]=sup.getNAS()

            if abs(params.vals.FinalTime-sup.getTime())<1:
                '''iteration 0 is about to end. so release the video and turn of record
                flag so we dont have any video attribute afterwards and save SARs as npy ans csv '''
                record=False
                showFrames=False
                if hasattr(sup,'video'): sup.video.release()
        else:
            raise NameError("[-] METHOD NOT RECOGNIZED")
        sampled+=1
        t=sup.getTime()
#...............................................................................................................................

if __name__ == "__main__":
    print(colored("VVVVVVVVVVVVVVVVVV STARTED VVVVVVVVVVVVVVVVVV","yellow"))
    # print(colored("[!] be carefull avout sup.visualuz","red"))
    # print(colored("[!] action space changed","red"))
    
    #! make sure code starts from its own local location. last part is scripts own name that is why .. is for
    os.chdir(  os.path.dirname(__file__))
    with open('params.JSON') as json_file: 
        params_json = json.load(json_file) 
    params_obj_str = DefaultMunch.fromDict(params_json) #! convert dict to obj
    
    global params
    params = param_str2val(params_obj_str)
    
    global comment
    if params.vals.method=="RL":
        comment=params.vals.paramReductionMethod+' '+str(params.vals.PRMparameter)+' '+params.vals.comment+' noise '+str(params.vals.noise)
    else:
        comment=params.vals.method+' '+params.vals.comment+' noise '+str(params.vals.noise)


    '''initiate seed'''
    seed=set_seed(params.vals.seed_value)
    params.vals.seed_value=seed

    t1_=TIME()

    warningSupress()
    
    ''' parameter value assigning '''
    print(colored('[+] '+comment,'green'))
    print(colored('[+] paramReductionMethod','green'),params.vals.paramReductionMethod,params.vals.PRMparameter)

    # LOGthrd=threading.Thread(target=LOG)
    #! put all data in output_base_path
    output_base_path = "output"
    
    #! check if folder already exists
    if os.path.exists(output_base_path) == False:
        os.makedirs(output_base_path)
    
    codeBeginTime=os.path.join(output_base_path , ctime(TIME()).replace(':','_').replace(' ','_')+'_'+params.vals.method+'_'+comment)

    ''' preparing dirs '''
    os.makedirs(os.path.join(codeBeginTime,'process data') , exist_ok=True)

    ''' save parameters into a file '''
    with open( os.path.join(codeBeginTime,'params.txt'),'w' ) as paramfile :
        paramfile.write(str(params))

    ''' for saving csvs which is Q-table of robot 0 for iteration 0 '''
    if params.flags.save_csv: os.makedirs(os.path.join(codeBeginTime,'csvs')) 


    ''' initilization '''
    sampledDataNum=params.vals.FinalTime//params.vals.samplingPeriodSmall
    saved=0
    print(colored('[+] '+params.vals.method,'green'))
    print(colored('[+] press ctrl+c for saving data asynchronously','green'))
    QtableMem=np.zeros((params.vals.iteration,params.vals.ROBN,7,44)) ##### caviat
    # QtableMem=np.zeros((iteration,ROBN,7,6)) ##### caviat
    
    log=np.zeros((params.vals.iteration,sampledDataNum,params.vals.ROBN,3))
    if params.vals.paramReductionMethod=='classical' or params.vals.paramReductionMethod=='cyclical':
        eps=np.zeros((params.vals.iteration,sampledDataNum,params.vals.ROBN))##### caviat
    elif params.vals.paramReductionMethod=='VDBE':
        eps=np.zeros((params.vals.iteration,sampledDataNum,params.vals.ROBN,7))##### caviat

    alpha=np.zeros((params.vals.iteration,sampledDataNum,params.vals.ROBN))
    reward=np.zeros((params.vals.iteration,sampledDataNum,params.vals.ROBN))
    NAS=np.zeros((params.vals.iteration,sampledDataNum))
    NASw=np.zeros((params.vals.iteration,sampledDataNum))

    strip=np.arange(0,6) ##### caviat: table dimentions pre known
    strip[strip%2==0]=0
    strip[strip%2==1]=255
    tableImSize=(7*20,6*20)[::-1] ##### caviat: table dimentions pre known


    if params.flags.save_tables_videos:
        imsName=["epsilon","QtableCheck","QtableRob0"]
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        FPS=10
        videoList=[]
        for _ in range(len(imsName)):
            videoList.append(cv.VideoWriter(os.path.join(codeBeginTime,imsName[_]+'.mp4'),fourcc, FPS, tableImSize,True))


    for it in range(params.vals.iteration):
        iteration_duration=TIME()
        print(colored("\t[+] iteration: ",'blue'), it)
        t=0;tt=0;sampled=0
        sup=SUPERVISOR(params.vals.ROBN , codeBeginTime , params.flags.showFrames , params.flags.globalQ , params.flags.record , params.vals.Lx , params.vals.Ly , params.vals.cueRaduis , params.vals.visibleRaduis , params.vals.paramReductionMethod , params.vals.PRMparameter , params.vals.noise ,params.vals.method)
        sup.generateRobots()
        sup.moveAll() # to make initilazation happen
        GroundChanged=False # to make sure ground is changed only once in each iteration
        checkHealth()
        while sup.getTime()<=params.vals.FinalTime:
            ''' start of main loop '''
            sup.checkCollision()
            sup.aggregateSwarm()
            if params.vals.method=='RL':
                sup.getQRs()
                sup.swarmRL()
                if params.flags.communicate:
                    sup.talk()
            elif params.vals.method=='LBA':
                sup.getQRs()
                sup.LBA()
            sup.moveAll()
            if abs(params.vals.HalfTime-sup.getTime())<1 and GroundChanged==False:
                GroundChanged=True
                print(colored('\t[+] half time reached','green'))
                if params.vals.dynamic:
                    sup.changeGround()
            LOG()
        if params.vals.method=="RL":
            QtableMem[it,:,:,:]=sup.getQtables()
        '''V: -1 is for that 'it' at max will be iteration-1 and after that code will exit the loop'''
        if it==params.vals.iteration-1: saveData("iterationDone") 
        del sup
        print(colored("\t[+] iteration duration: ",'blue'),int(TIME()-iteration_duration))
    print(colored('[+] duration','green'),int(TIME()-t1_))
    print(colored('[+] goodbye  ^^',"green"))
