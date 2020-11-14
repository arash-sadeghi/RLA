import cv2 as cv
import numpy as np
import random as rnd
from math import sin,cos,sqrt,atan2,exp
from itertools import product
from time import time as TIME
from time import ctime
import os
import shutil
import signal
from varname import nameof
import pickle
from termcolor import colored as c
################################################################################################################################
################################################################################################################################
################################################################################################################################
def checkHealth():
    health=shutil.disk_usage('/')
    if health[-1]/(2**30)<=5:
        raise NameError('[-] disk is getting full')
# ------------------------------------------------------------------------------------------------------------------------------
def m2px(inp):
    return int(inp*512/2)
# ------------------------------------------------------------------------------------------------------------------------------
def RotStandard(inp):
    if inp<0: inp+=360
    if inp>360: inp-=360
    return inp
# ------------------------------------------------------------------------------------------------------------------------------
def dist(x,y):
    return sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)
# ------------------------------------------------------------------------------------------------------------------------------
def TableCompare(table1,table2):
    return np.round((table1+table2)/2,3)
# ------------------------------------------------------------------------------------------------------------------------------
def DirLocManage():
    ''' with this segment code is callable from any folder '''
    if os.name=='nt':
        dirChangeCharacter='\\'
    else:
        dirChangeCharacter='/'
    scriptLoc=__file__
    for i in range(len(scriptLoc)):
        # if '/' in scriptLoc[-i-2:-i]: # in running
        if dirChangeCharacter in scriptLoc[-i-2:-i]: # in debuging
            scriptLoc=scriptLoc[0:-i-2]
            break
    print('[+] code path',scriptLoc)
    os.chdir(scriptLoc)
    return dirChangeCharacter
    ''' done '''
# ------------------------------------------------------------------------------------------------------------------------------
def sigmoid(x):
    return 1/(1+exp(-10*x+5))
# ------------------------------------------------------------------------------------------------------------------------------
def saturate(x):
    if x>=1: return 1
    elif x<=0: return 0
    else: return x
################################################################################################################################
################################################################################################################################
################################################################################################################################
class SUPERVISOR:
    def __init__(self,ROBN,codeBeginTime,vizFlag,globalQ=False,record=False,Lx=2,Ly=4,cueRaduis=0.7,visibleRaduis=0.3):
        self.Lx=m2px(Lx)
        self.Ly=m2px(Ly)
        self.cueRaduis=m2px(cueRaduis)
        self.visibleRaduis=m2px(visibleRaduis)
        self.ground=self.generateBackground(self.Lx,self.Ly,self.cueRaduis,self.visibleRaduis)
        self.sharedParams()
        self.vizFlag=vizFlag
        self.fps=int(self.timeStep*1000)//50
        self.ROBN=ROBN
        self.collisionDist=m2px(0.05)
        self.swarm=[0]*self.ROBN
        self.wallNum=4
        self.time=0
        self.flagsR=[[False for _ in range(self.ROBN)] for _i in range(self.ROBN)] # [[False]*self.ROBN]*self.ROBN >>this one has address problem<<
        self.counterR= [[0 for _ in range(self.ROBN)] for _i in range(self.ROBN)]# [[0]*self.ROBN]*self.ROBN >>this one has address problem<<
        self.collisionDelay=1
        self.detectRad=0.06
        self.robotRad=0.06
        self.robotSenseRad=m2px(self.robotRad+self.detectRad)
        self.Wmax=120 if not self.debug else 0
        if self.debug: print('\t[+] Debugging mode: Wmax=0')
        self.log=0
        self.record=record
        videoRecordTime=ctime(TIME()).replace(':','_')
        capture_rate=5
        FPS=20
        size=(self.Xlen,self.Ylen)
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        if self.record:
            if os.name=='nt':
                self.video = cv.VideoWriter(codeBeginTime+'\\'+videoRecordTime+'.mp4',fourcc, FPS, size,True)
            else:
                self.video = cv.VideoWriter(codeBeginTime+'/'+videoRecordTime+'.mp4',fourcc, FPS, size,True)
        
        self.globalQ=globalQ
# ...............................................................................................................................
    def sharedParams(self):
        """
        These parameters are constant
        Other wise it will cause conflict between
        supervisor and robots
        """
        self.paramReductionMethod='my'
        self.Xlen=np.shape(self.ground)[0]  
        self.Ylen=np.shape(self.ground)[1]  
        self.cueRadius=m2px(0.7)   
        self.EpsilonDampRatio=0.999 #######

        angles=np.arange(0,180+18,18)
        maxlen=int(16*512/9.2) # 14
        lens=[maxlen//4,2*maxlen//4,3*maxlen//4,3*maxlen//4]
        self.actionSpace=list(product(lens,angles))

        self.numberOfStates=7
        self.NumberOfActions=len(self.actionSpace)

        self.RLparams={"epsilon":0.9,"alpha":0.9/2,"sensitivity":1,"maxdiff":255}
        self.velocity=14
        self.timeStep=512*0.001
        self.printFlag=False
        self.debug=not True ##########
        self.QRloc={'QR1':(self.Ylen,self.Xlen//4),'QR2':(self.Ylen,self.Xlen//4*2),\
            'QR3':(self.Ylen,self.Xlen//4*3),'QR4':(0,self.Xlen//4*3),'QR5':(0,self.Xlen//4*2),'QR6':(0,self.Xlen//4)}
        self.QRdetectableArea=m2px(0.3)      
#...............................................................................................................................
    def generateBackground(self,Lx,Ly,cueRaduis,visibleRaduis):
        Lxpx=Lx
        Lypx=Ly
        R=cueRaduis
        def gauss(x):
            a = 1.0 # amplititude of peak
            b = Lxpx/2.0 # center of peak
            c = Lxpx/7# standard deviation
            return a*exp(-((x-b)**2)/(2*(c**2)))
        im=np.zeros((Lxpx,Lypx))
        for i in range(0,R):
            cv.circle(im,(int((Lypx/4)),int(Lxpx/2)),i,gauss(Lxpx/2-i),2)
        im=cv.rotate(im, cv.ROTATE_90_CLOCKWISE)
        QRlocs=[(Lxpx,Lypx//4),(Lxpx,Lypx//2),(Lxpx,3*Lypx//4),(0,3*Lypx//4),(0,Lypx//2),(0,Lypx//4)]
        for i in QRlocs:
            cv.circle(im,i,10,(255,255,255),-1)
            cv.circle(im,i,visibleRaduis,(255,255,255),1)
        for i in range(0,Lypx):
            for j in range(0,Lxpx):
                im[i,j]=im[i,j]*255
                im[i,j]=255-im[i,j]
        cv.imwrite("BackgroundGeneratedBySim.png",im)
        return cv.imread("BackgroundGeneratedBySim.png")
# ...............................................................................................................................
    def generateRobots(self):
        margin=m2px(self.robotRad)*5
        possibleY=np.linspace(0+margin,self.Xlen-margin,int(self.Xlen//2*self.robotSenseRad))
        possibleX=np.linspace(0+margin,self.Ylen-margin,int(self.Ylen//2*self.robotSenseRad))
        possibleRot=np.linspace(0,360,10)
        for i in range(self.ROBN):
            self.swarm[i]=ROBOT(str(i),self.ground)
            self.swarm[i].position=np.array([rnd.sample(list(possibleX),1)[0],rnd.sample(list(possibleY),1)[0]])
            self.swarm[i].rotation2B=rnd.sample(list(possibleRot),1)[0]
            self.swarm[i].ground=self.ground # sharing ground image among robots

            # sharing the Qtable addres for all robots
            if self.globalQ:
                if i==0: tmp=self.swarm[i].Qtable
                else: self.swarm[i].Qtable=tmp
#...............................................................................................................................
    def visualize(self):
        background=np.copy(self.ground)
        for i in range(self.ROBN):

            if self.swarm[i].inAction== True:
                I=tuple(map(lambda x: int(round(x)),self.swarm[i].initialPos))
                E=tuple(map(lambda x: int(round(x)),self.swarm[i].desiredPos))
                C=tuple(map(lambda x: int(round(x)),self.swarm[i].position))
                Q=tuple(map(lambda x: int(round(x)),self.swarm[i].QRloc[self.swarm[i].lastdetectedQR]))
                cv.arrowedLine(background,Q,E,(0,0,255),2) # action space vector
                # cv.arrowedLine(background,I,C,(255,0,0),2) # motion trajectory
                cv.arrowedLine(background,I,E,(0,255,0),2) # vector that must be traversed
                cv.arrowedLine(background,I,Q,(100,100,0),2) # sudo vec
            
            vizPos=[]
            for ii in range(len(self.swarm[i].position)): vizPos.append(int(self.swarm[i].position[ii]))

            RobotColor=(255,100,100)
            debugRobotColor=(0,255,0)
            if self.swarm[i].ExploreExploit=='Exploit' and self.swarm[i].inAction== True:
                RobotColor=(100,255,100) # color of robot will be green if exploits. otherwise blue again

            if self.debug and i==0:
                cv.circle(background,tuple(vizPos),self.robotSenseRad,debugRobotColor,1)
                cv.circle(background,tuple(vizPos),m2px(self.robotRad),debugRobotColor,-1)
            else:
                cv.circle(background,tuple(vizPos),self.robotSenseRad,RobotColor,1)
                cv.circle(background,tuple(vizPos),m2px(self.robotRad),RobotColor,-1)

            direction=np.array([int(m2px(self.robotRad)*sin(np.radians(self.swarm[i].rotation))) \
                , int(m2px(self.robotRad)*cos(np.radians(self.swarm[i].rotation)))])
            cv.putText(background,str(i),tuple([vizPos[0]-10,vizPos[1]+10]),cv.FONT_HERSHEY_SIMPLEX\
                ,0.75,(0,0,0),3 )
            cv.line(background,tuple(vizPos),tuple(np.array(vizPos)+direction),(0,0,200),3)

        cv.putText(background,str(int(self.time))+' s',(20,20),cv.FONT_HERSHEY_SIMPLEX,0.75,(0,100,0),3 )

        # AllEpsilons=np.array([self.swarm[_].RLparams['epsilon'] for _ in range(self.ROBN)])
        AllEpsilons=np.array([np.mean(self.swarm[_].epsilon) for _ in range(self.ROBN)])

        EpsilonAverage=round(np.mean(AllEpsilons),3)
        cv.putText(background,'eps: '+str(EpsilonAverage),(20,50),cv.FONT_HERSHEY_SIMPLEX,0.75,(0,100,0),3 )

        if self.vizFlag:
            cv.imshow("background",background)
            cv.waitKey(self.fps)
        if self.record: self.video.write(background)
#...............................................................................................................................
    def moveAll(self):
        for i in range(self.ROBN):
            if self.swarm[i].delayFlag==False:
                self.swarm[i].move()
            else :
                self.swarm[i].waitingTime-=self.timeStep
                if self.swarm[i].waitingTime<=0:
                    self.swarm[i].delayFlag=False
                    self.checkCollision(specific=True,robotNum=i)            
                    self.swarm[i].move()

        self.time+=self.timeStep
#...............................................................................................................................
    def checkCollision(self,specific=False,robotNum=None):
        if specific==False:
            for i in range(0,self.ROBN):
                Xcond1=self.swarm[i].position[0]>=self.Ylen-self.robotSenseRad 
                Xcond2=self.swarm[i].position[0]<=0+self.robotSenseRad
                Ycond1=self.swarm[i].position[1]>=self.Xlen-self.robotSenseRad 
                Ycond2=self.swarm[i].position[1]<=0+self.robotSenseRad

                if Ycond1 or Xcond1 or Ycond2 or Xcond2:
                    if Xcond1:# >|
                        if 0<=self.swarm[i].rotation<=180:
                            self.swarm[i].rotation2B=rnd.randint(180,360)
                    if Xcond2:
                        if 180<=self.swarm[i].rotation<=360:
                            self.swarm[i].rotation2B=rnd.randint(0,180)# |<
                    if Ycond1:
                        if 270<=self.swarm[i].rotation<=360 or 0<=self.swarm[i].rotation<=90:
                            self.swarm[i].rotation2B=rnd.randint(90,270)# _
                    if Ycond2:
                        if 90<=self.swarm[i].rotation<=270:
                            self.swarm[i].rotation2B=rnd.randint(270,360+90)# -

                    self.swarm[i].rotation2B=RotStandard(self.swarm[i].rotation2B)
                    if self.swarm[i].inAction==True:
                        self.swarm[i].actAndReward(-1) ########3
                        # self.swarm[i].actAndReward(0)
         
            for i in range(0,self.ROBN):
                for j in range(0,self.ROBN):
                    if j!=i:
                        if dist(self.swarm[j].position,self.swarm[i].position)<=self.robotSenseRad+self.collisionDist and self.flagsR[i][j]==False: 
                            collisionAngle=RotStandard(np.degrees(atan2( self.swarm[j].position[0]-self.swarm[i].position[0] , self.swarm[j].position[1]-self.swarm[i].position[1] )))
                            self.swarm[i].rotation2B=collisionAngle+rnd.randint(90,270) 
                            self.swarm[i].rotation2B=RotStandard(self.swarm[i].rotation2B)
                            self.flagsR[i][j]=True
                            self.swarm[i].inAction=False
                            break

                        elif self.flagsR[i][j]==True:
                            self.counterR[i][j]+=1
                            if self.counterR[i][j]>=self.collisionDelay:
                                self.counterR[i][j]=0
                                self.flagsR[i][j]=False
                        elif dist(self.swarm[j].position,self.swarm[i].position)>=self.robotSenseRad*2 :
                            self.flagsR[i][j]=False
                        
        else:
            i=robotNum
            for j in range(0,self.ROBN):
                if j!=i:
                    if dist(self.swarm[j].position,self.swarm[i].position)<=self.robotSenseRad+self.collisionDist and self.flagsR[i][j]==False: 
                        self.swarm[i].rotation+=rnd.randint(90,270)
                        collisionAngle=RotStandard(np.degrees(atan2( self.swarm[j].position[0]-self.swarm[i].position[0] , self.swarm[j].position[1]-self.swarm[i].position[1] )))
                        self.swarm[i].rotation=collisionAngle+rnd.randint(90,270) 
                        self.swarm[i].rotation=RotStandard(self.swarm[i].rotation)
                        self.flagsR[i][j]=False
                        break
#...............................................................................................................................
    def getGroundSensors(self):
        for i in range(self.ROBN):
            self.swarm[i].groundSense()
#...............................................................................................................................
    def aggregateSwarm(self):
        for i in range(self.ROBN):
            self.swarm[i].aggregate(self.flagsR,self.Wmax)
#...............................................................................................................................
    def getTime(self):
        return int(self.time)
#...............................................................................................................................
    def getStatus(self):
        inCue=0
        for i in range(self.ROBN):
            if self.swarm[i].groundSensorValue>0:
                inCue+=1
        return inCue/self.ROBN
#...............................................................................................................................
    def getQRs(self):
        for i in range(self.ROBN):
            self.swarm[i].detectQR()
#...............................................................................................................................
    def swarmRL(self):
        for i in range(self.ROBN):
            if self.swarm[i].detectedQR != 'QR0' or self.swarm[i].inAction==True:
                self.swarm[i].RL()
#...............................................................................................................................
    def talk(self): ##### print must be fixed ####### epsilon sharing policy must be changed
        for i in range(self.ROBN):
            if any(self.flagsR[i]):
                tobesharedRobots=np.where(self.flagsR[i])[0]
                for j in tobesharedRobots:
                    temp=TableCompare(self.swarm[i].Qtable,self.swarm[j].Qtable)
                    if self.vizFlag : print('table shared among ',i,j,np.all(self.swarm[i].Qtable==self.swarm[j].Qtable),end=' ')
                    self.swarm[i].Qtable=np.copy(temp)
                    self.swarm[j].Qtable=np.copy(temp)
                    if self.vizFlag : print(np.all(self.swarm[i].Qtable==self.swarm[j].Qtable),\
                        self.swarm[i].RLparams['epsilon'],self.swarm[j].RLparams['epsilon'])
                    
                    temp=min(self.swarm[i].RLparams['epsilon'],self.swarm[j].RLparams['epsilon'])
                    self.swarm[i].RLparams['epsilon']=temp
                    self.swarm[j].RLparams['epsilon']=temp
                    if self.vizFlag : print('eps after',self.swarm[i].RLparams['epsilon'],self.swarm[j].RLparams['epsilon'])
        if self.vizFlag : print("------------------------------------------------------")
#...............................................................................................................................
    def getlog(self):
        location=[]
        for i in range(self.ROBN):
            location.append(np.append(self.swarm[i].position,self.swarm[i].rotation))
        return np.array(location)
#...............................................................................................................................
    def getQtables(self):
        Qtables=[]
        for i in range(self.ROBN):
            Qtables.append(self.swarm[i].Qtable)
        return np.array(Qtables)
#...............................................................................................................................
    def getReward(self):
        Rewards=[]
        for i in range(self.ROBN):
            Rewards.append(self.swarm[i].rewardMemory)
        return Rewards
#...............................................................................................................................
    def geteps(self):
        return np.array([np.mean(self.swarm[_].epsilon) for _ in range(self.ROBN)]) 
#...............................................................................................................................
    def getalpha(self):
        return np.array([np.mean(self.swarm[_].alpha) for _ in range(self.ROBN)]) 

################################################################################################################################
################################################################################################################################
################################################################################################################################

class ROBOT(SUPERVISOR):
    def __init__(self,name,ground):
        self.ground=ground
        super().sharedParams()
        self.ground=0 # initilizing for robots
        self.rotation=0
        self.rotation2B=0
        self.position=[0,0]
        self.groundSensorValue=0
        self.robotName=name
        self.waitingTime=0
        self.delayFlag=False
        self.detectedQR=' '
        self.lastdetectedQR=' '
        self.Qtable=np.zeros((self.numberOfStates,self.NumberOfActions))
        self.QtableCheck=np.zeros((self.numberOfStates,self.NumberOfActions))
        self.exploredAmount=0
        self.inAction=False
        self.desiredPos=0
        self.action=0
        self.state=0
        self.initialPos=0
        self.rewardMemory=[]
        self.sudoVec=0
        self.ExploreExploit=''
        self.actionIndx=0

        self.delta=np.zeros(np.shape(self.Qtable))
        self.deltaDot=np.zeros(np.shape(self.Qtable))
        self.prevdelta=np.zeros(np.shape(self.Qtable))
        self.DELTA=np.zeros(np.shape(self.Qtable))
        self.deltaDot=np.zeros(np.shape(self.Qtable))

        ''' start from full eaxploration and complete learning '''
        self.epsilon=np.zeros(np.shape(self.Qtable))+self.RLparams['epsilon']
        self.alpha=np.zeros(np.shape(self.Qtable))+self.RLparams['alpha']

        if self.debug and self.robotName=='0': self.printFlag=True
#...............................................................................................................................  
    def move(self):
            self.rotation=self.rotation2B
            self.position[0]=self.position[0]+self.velocity*sin(np.radians(self.rotation))*self.timeStep
            self.position[1]=self.position[1]+self.velocity*cos(np.radians(self.rotation))*self.timeStep

            self.position[0]=min(self.position[0],self.Ylen-1)
            self.position[1]=min(self.position[1],self.Xlen-1)

            self.position[0]=max(self.position[0],0+1)
            self.position[1]=max(self.position[1],0+1)
#...............................................................................................................................
    def groundSense(self):
        temp=self.ground[int(round(self.position[1])),int(round(self.position[0]))]
        if dist(self.position,[self.Xlen//4,self.Ylen//2])<=self.cueRadius:
            self.groundSensorValue=255-temp[0]
        else: self.groundSensorValue=0
#...............................................................................................................................
    def aggregate(self,flagsR,Wmax):
        if any(flagsR[int(self.robotName)]) and self.groundSensorValue>0 and not self.delayFlag :
            self.waitingTime=Wmax*((self.groundSensorValue**2)/((self.groundSensorValue**2) + 5000))
            self.delayFlag=True
#...............................................................................................................................
    def detectQR(self):
        for QRpos in self.QRloc:
            if dist(self.position,self.QRloc[QRpos])<=self.QRdetectableArea :
                self.detectedQR=QRpos
                break
            else:
                self.detectedQR='QR0'
#...............................................................................................................................
    def RL(self):
        if self.inAction==False :
            self.state=int(self.detectedQR[-1])
            eps=np.mean(self.epsilon[self.state]) #########
            if rnd.random()<= eps: ###################
                self.action=rnd.sample(self.actionSpace,1)[0]
                self.ExploreExploit='Explore'
            else:
                actionIndx=np.argmax(self.Qtable[self.state,:])
                self.action=self.actionSpace[actionIndx]
                self.ExploreExploit='Exploit'

            self.actAndReward()
        else:
            self.actAndReward()
#...............................................................................................................................
    def actAndReward(self,rewardInp=None):
        if self.inAction==False:
            angle=self.action[1]
            length=self.action[0]
            if self.state<=3: angle=180+angle

            actionXY=np.array([length*sin(np.radians(angle)),length*cos(np.radians(angle))])
            self.sudoVec= np.array(self.QRloc[self.detectedQR])-self.position
            actionXY_SudoVec=actionXY+self.sudoVec
            angle=np.degrees(atan2(actionXY_SudoVec[0],actionXY_SudoVec[1]))
            length=sqrt(actionXY_SudoVec[0]**2+actionXY_SudoVec[1]**2)
            self.rotation2B=angle
            self.inAction=True
            self.initialPos=np.copy(self.position)
            self.reward=0
            self.desiredPos=self.initialPos+actionXY_SudoVec
            self.lastdetectedQR=self.detectedQR
        elif (self.inAction==True and dist(self.position,self.desiredPos)<=20) or rewardInp!=None:
            ''' elif goal reached or a reward is forced '''
            self.inAction=False
            if rewardInp==None:
                self.groundSense()
                self.reward=self.groundSensorValue
                # if self.reward==0:self.reward-=1 #######
            else: self.reward=rewardInp
            # if self.printFlag: print("\t [+] REWARD::",self.reward)
            self.actionIndx=self.actionSpace.index(self.action)
            x=self.state
            y=self.actionIndx

            ''' parameter adaptation '''
            self.delta[x,y]=abs(self.reward-self.Qtable[x,y])
            self.deltaDot[x,y]=self.delta[x,y]-self.prevdelta[x,y]
            self.deltaDot[x,y]=np.sign(self.deltaDot[x,y])

            self.deltaDot[x,y] = self.deltaDot[x,y] if self.deltaDot[x,y]!=0 else 1
            self.DELTA[x,y]=self.deltaDot[x,y]*self.delta[x,y]/self.RLparams['maxdiff']
            if self.printFlag:
                print('\n\t[+] index {} reward {} Qtable {} delta {} prevdelta {} deltaDot {} DELTA {}'\
                    .format([x,y],self.reward,self.Qtable[x,y],\
                        self.delta[x,y],self.prevdelta[x,y],self.deltaDot[x,y],self.DELTA[x,y]))
                print('\t[+]')
            self.prevdelta[x,y]=self.delta[x,y]
            self.updateRLparameters()

            ''' update rule '''
            # self.Qtable[x,y]+=self.RLparams['alpha']*(self.reward-self.Qtable[x,y]) ########################
            self.Qtable[x,y]+=self.alpha[x,y]*(self.reward-self.Qtable[x,y]) ########################

            self.QtableCheck[x,y]=1
            self.exploredAmount=(self.QtableCheck==1).sum()/np.size(self.Qtable[1:]) # discarding data of state 0
            self.rewardMemory.append(self.reward)

#...............................................................................................................................
    def updateRLparameters(self):
        if self.paramReductionMethod=='old':
            self.RLparams["epsilon"]*=self.EpsilonDampRatio ####
        elif self.paramReductionMethod=='my':
            if self.printFlag: print('\t[+] before update: eps {} alpha {}'.format(self.epsilon[self.state,self.actionIndx],self.alpha[self.state,self.actionIndx]))
            ''' sensitivity must be multiplied not divided '''
            self.epsilon[self.state,self.actionIndx]=saturate(self.epsilon[self.state,self.actionIndx]+self.DELTA[self.state,self.actionIndx]*self.RLparams["sensitivity"])
            self.alpha[self.state,self.actionIndx]=self.epsilon[self.state,self.actionIndx]/2 ####
            if self.printFlag: print('\t[+] after update: eps {} alpha {}'.format(self.epsilon[self.state,self.actionIndx],self.alpha[self.state,self.actionIndx]))
