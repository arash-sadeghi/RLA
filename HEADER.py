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
from termcolor import colored 
from subprocess import call 
from itertools import combinations  as comb
################################################################################################################################
################################################################################################################################
################################################################################################################################
def warningSupress():
    from sys import warnoptions
    from warnings import simplefilter
    if not warnoptions:
        simplefilter("ignore")
# ------------------------------------------------------------------------------------------------------------------------------
def checkHealth():
    health=shutil.disk_usage('/')
    if health[-1]/(2**30)<=5:
        raise NameError('[-] disk is getting full')
# ------------------------------------------------------------------------------------------------------------------------------
def m2px(inp):
    return int(inp*512/2)
# ------------------------------------------------------------------------------------------------------------------------------
def px2m(inp):
    return inp*2/512
# ------------------------------------------------------------------------------------------------------------------------------
def RotStandard(inp):
    if inp<0: inp+=360
    if inp>360: inp-=360
    return inp
# ------------------------------------------------------------------------------------------------------------------------------
''' optimized '''
def dist(delta):
    if np.size(delta)>2:
        return np.sqrt(np.square(delta[:,0])+np.square(delta[:,1]))
    else:
        return np.sqrt(np.square(delta[0])+np.square(delta[1]))
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
# ------------------------------------------------------------------------------------------------------------------------------
def quadratic(x):
    return saturate(x**2)
################################################################################################################################
################################################################################################################################
################################################################################################################################
class SUPERVISOR:
    def __init__(self,ROBN,codeBeginTime,vizFlag,globalQ=False,record=False,Lx=2,Ly=4,cueRaduis=0.7,visibleRaduis=0.3,paramReductionMethod='adaptive'):
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
        self.flagsR=np.zeros((self.ROBN,self.ROBN))
        self.counterR=np.zeros((self.ROBN,self.ROBN))
        self.collisionDelay=1
        self.detectRad=0.06
        self.robotRad=0.06
        self.robotSenseRad=m2px(self.robotRad+self.detectRad)
        self.collisionDetectDist=self.robotSenseRad+self.collisionDist
        self.Wmax=120 if not self.debug else 0
        self.paramReductionMethod=paramReductionMethod
        print(colored('\t[+] paramReductionMethod','red'),self.paramReductionMethod)
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
        self.globalQ=globalQ
        self.allnodes=np.array(list(comb(np.arange(0,self.ROBN),2)))
        self.crowdThresh= 5

# sharedParams ...............................................................................................................................
    def sharedParams(self):
        """
        These parameters are constant
        Other wise it will cause conflict between
        supervisor and robots
        """
        self.Xlen=np.shape(self.ground)[0]  
        self.Ylen=np.shape(self.ground)[1]  
        self.cueRadius=m2px(0.7)   
        self.EpsilonDampRatio=0.999 #######

        angles=np.arange(0,180+18,18)
        maxlen=int(16*512/9.2) # 14
        lens=[maxlen//4,2*maxlen//4,3*maxlen//4,4*maxlen//4]
        self.actionSpace=list(product(lens,angles))

        self.numberOfStates=7
        self.NumberOfActions=len(self.actionSpace)

        self.RLparams={"epsilon":0.9,"alpha":0.9/2,"sensitivity":10,"maxdiff":255}
        self.velocity=14
        self.timeStep=512*0.001
        self.printFlag=not True # print flag for robot 0 # caviat
        self.debug=not True ##########
        self.QRloc={'QR1':(self.Ylen,self.Xlen//4),'QR2':(self.Ylen,self.Xlen//4*2),\
            'QR3':(self.Ylen,self.Xlen//4*3),'QR4':(0,self.Xlen//4*3),'QR5':(0,self.Xlen//4*2),'QR6':(0,self.Xlen//4)}
        self.QRdetectableArea=m2px(0.3)      
# generateBackground ...............................................................................................................................
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
        im=255-255*im
        cv.imwrite("BackgroundGeneratedBySim.png",im)
        return cv.imread("BackgroundGeneratedBySim.png")
# generateRobots ...............................................................................................................................
    def generateRobots(self):
        margin=m2px(self.robotRad)*5
        possibleY=np.linspace(0+margin,self.Xlen-margin,int(self.Xlen//2*self.robotSenseRad))
        possibleX=np.linspace(0+margin,self.Ylen-margin,int(self.Ylen//2*self.robotSenseRad))
        possibleRot=np.linspace(0,360,10)
        for i in range(self.ROBN):
            self.swarm[i]=ROBOT(self,str(i))
            self.swarm[i].position=np.array([rnd.sample(list(possibleX),1)[0],rnd.sample(list(possibleY),1)[0]])
            self.swarm[i].rotation2B=rnd.sample(list(possibleRot),1)[0]
            self.swarm[i].ground=self.ground # sharing ground image among robots

            # sharing the Qtable addres for all robots
            if self.globalQ:
                if i==0: tmp=self.swarm[i].Qtable
                else: self.swarm[i].Qtable=tmp
# visualize ...............................................................................................................................
    def visualize(self):
        if self.vizFlag:
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

            cv.imshow("background",background)
            cv.waitKey(self.fps)
        if self.record: self.video.write(background)
# moveAll ...............................................................................................................................
    def moveAll(self):
        for i in range(self.ROBN):
            if self.swarm[i].delayFlag==False:
                self.swarm[i].move()
            else :
                self.swarm[i].waitingTime-=self.timeStep
                if self.swarm[i].waitingTime<=0:
                    self.swarm[i].delayFlag=False
                    # if not self.crowded(i):
                    #     self.swarm[i].crowd+=1
                    # if self.swarm[i].crowd>self.crowdThresh:
                    #     self.swarm[i].crowd=0
                    #     self.checkCollision(specific=True,robotNum=i)            
                    #     self.swarm[i].move()
                    self.checkCollision(specific=True,robotNum=i)            
                    self.swarm[i].move()

        self.time+=self.timeStep
# crowded ...............................................................................................................................
    def crowded(self,i):
            temp=np.zeros((self.ROBN,2),dtype=int)
            temp[:,0]+=i
            temp[:,1]+=np.arange(0,self.ROBN)
            dists=np.array(list(map(lambda x: dist(self.swarm[x[0]].position-self.swarm[x[1]].position),temp)))
            indexes=np.where(dists<=self.collisionDetectDist)
            if np.size(indexes)>0:
                indexes=indexes[0]
                colliders=temp[indexes]
                colliders1d=np.reshape(colliders,(1,np.size(colliders)))[0]
                unique, counts = np.unique(colliders1d, return_counts=True)
                counts=np.array(list(counts))
                unique=np.array(list(unique))
                if np.any(counts>3): return True
                else: return False
            else : return False
# avoid ...............................................................................................................................
    def avoid(self,cols,specific=False):
        if specific==False:
            c=np.reshape(cols,(np.size(cols)//2,2))
            for pairs in c:
                i,j=pairs
                collisionAngle=RotStandard(np.degrees(atan2( self.swarm[j].position[0]-self.swarm[i].position[0] \
                    , self.swarm[j].position[1]-self.swarm[i].position[1] )))
                self.swarm[i].rotation2B=collisionAngle+rnd.randint(90,270) 
                self.swarm[i].rotation2B=RotStandard(self.swarm[i].rotation2B)
                self.flagsR[i,j]=1
                self.swarm[i].inAction=False
                i,j=j,i
                collisionAngle=RotStandard(np.degrees(atan2( self.swarm[j].position[0]-self.swarm[i].position[0] \
                    , self.swarm[j].position[1]-self.swarm[i].position[1] )))
                self.swarm[i].rotation2B=collisionAngle+rnd.randint(90,270) 
                self.swarm[i].rotation2B=RotStandard(self.swarm[i].rotation2B)
                self.flagsR[i,j]=1
                self.swarm[i].inAction=False
        else:
            c=np.reshape(cols,(np.size(cols)//2,2))
            for pairs in c:
                i,j=pairs
                collisionAngle=RotStandard(np.degrees(atan2( self.swarm[j].position[0]-self.swarm[i].position[0] \
                    , self.swarm[j].position[1]-self.swarm[i].position[1] )))
                self.swarm[i].rotation2B=collisionAngle+rnd.randint(90,270) 
                self.swarm[i].rotation2B=RotStandard(self.swarm[i].rotation2B)
                self.flagsR[i,j]=0

# checkCollision ...............................................................................................................................
    def checkCollision(self,specific=False,robotNum=None):
        ''' optimized '''
        ''' flagsR is always numerical not boolien'''
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
                    elif Xcond2:
                        if 180<=self.swarm[i].rotation<=360:
                            self.swarm[i].rotation2B=rnd.randint(0,180)# |<
                    elif Ycond1:
                        if 270<=self.swarm[i].rotation<=360 or 0<=self.swarm[i].rotation<=90:
                            self.swarm[i].rotation2B=rnd.randint(90,270)# _
                    elif Ycond2:
                        if 90<=self.swarm[i].rotation<=270:
                            self.swarm[i].rotation2B=rnd.randint(270,360+90)# -

                    self.swarm[i].rotation2B=RotStandard(self.swarm[i].rotation2B)
                    if self.swarm[i].inAction==True:
                        self.swarm[i].actAndReward(-1) ########3
                        # self.swarm[i].actAndReward(0)
            #.........................
            # func=np.vectorize(lambda x: dist(self.swarm[x[0]].position-self.swarm[x[1]].position))
            

            dists=np.array(list(map(lambda x: dist(self.swarm[x[0]].position-self.swarm[x[1]].position),self.allnodes)))
            indexes=np.where(dists<=self.collisionDetectDist)
            if np.size(indexes)>0:
                indexes=indexes[0]
                colliders=self.allnodes[indexes]
                flags=self.flagsR[colliders[:,0],colliders[:,1]]==0 # i want collider to choose from flagsR
                colliders=colliders[flags] # i want flags to choose form collider_filterer
                self.avoid(colliders)
                ''' if collided with more than one robot '''
                # colliders1d=np.reshape(colliders,(1,np.size(colliders)))[0]
                # unique, counts = numpy.unique(colliders1d, return_counts=True)
                # # counts=np.array(list(counts))
                # # unique=np.array(list(unique))
                # # ultraColRobots=unique[counts>1]



            cond=self.flagsR==1
            if np.size(indexes)>0: # if any collision has happend dont touch theri matrices
                cond[colliders[:,0],colliders[:,1]]=False # dont touch the recently decided ones            
                cond[colliders[:,1],colliders[:,0]]=False # also reverse of tuple
            self.counterR[cond]+=1
            self.flagsR[self.counterR>=self.collisionDelay]=0
            self.counterR[self.counterR>=self.collisionDelay]=0
            #.........................
        else:
            temp=np.zeros((self.ROBN-1,2),dtype=int)
            temp[:,0]+=robotNum
            yaxis=np.arange(0,self.ROBN)
            temp[:,1]+=yaxis[yaxis!=robotNum]
            dists=np.array(list(map(lambda x: dist(self.swarm[x[0]].position-self.swarm[x[1]].position),temp)))
            indexes=np.where(dists<=self.collisionDetectDist)
            if np.size(indexes)>0:
                indexes=indexes[0]
                colliders=temp[indexes]
                flags=self.flagsR[colliders[:,0],colliders[:,1]]==0 # i want collider to choose from flagsR
                colliders=colliders[flags] # i want flags to choose form collider_filterer
                self.avoid(colliders,specific)
# getGroundSensors ...............................................................................................................................
    def getGroundSensors(self):
        for i in range(self.ROBN):
            self.swarm[i].groundSense()
# aggregateSwarm ...............................................................................................................................
    def aggregateSwarm(self):
        for i in range(self.ROBN):
            self.swarm[i].aggregate()
# getTime ...............................................................................................................................
    def getTime(self):
        return int(self.time)
# getNAS ...............................................................................................................................
    def getNAS(self):
        f=np.vectorize(lambda x: x.groundSensorValue)
        inCue=np.count_nonzero(f(self.swarm))
        return inCue/self.ROBN
# getQRs ...............................................................................................................................
    def getQRs(self):
        for i in range(self.ROBN):
            self.swarm[i].detectQR()
# swarmRL ...............................................................................................................................
    def swarmRL(self):
        for i in range(self.ROBN):
            if self.swarm[i].detectedQR != 'QR0' or self.swarm[i].inAction==True:
                self.swarm[i].RL()
# talk ...............................................................................................................................
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
# getLog ...............................................................................................................................
    def getLog(self):
        location=[]
        for i in range(self.ROBN):
            location.append(np.append(self.swarm[i].position,self.swarm[i].rotation))
        return np.array(location)
# getQtables ...............................................................................................................................
    def getQtables(self):
        Qtables=[]
        for i in range(self.ROBN):
            Qtables.append(self.swarm[i].Qtable)
        return np.array(Qtables)
# getReward ...............................................................................................................................
    def getReward(self):
            return np.array([self.swarm[_].rewardMemory[-1] if len(self.swarm[_].rewardMemory)>0 else 0\
                 for _ in range(self.ROBN)]) 
# getEps ...............................................................................................................................
    def getEps(self):
        if self.paramReductionMethod=='classic':
            return np.array([self.swarm[_].RLparams['epsilon'] for _ in range(self.ROBN)]) 
        else:
            return np.array([np.mean(self.swarm[_].epsilon[1:]) for _ in range(self.ROBN)]) #[1:] is because discarding state 0
# getAlpha ...............................................................................................................................
    def getAlpha(self):
        return np.array([np.mean(self.swarm[_].alpha) for _ in range(self.ROBN)]) 
# changeGround ...............................................................................................................................
    def changeGround(self):
        ''' this code will ruin address exchange'''
        self.ground=cv.rotate(self.ground,cv.ROTATE_180)
        for i in range(self.ROBN):
            self.swarm[i].ground=self.ground
        ''' now address equality returned '''
################################################################################################################################
################################################################################################################################
################################################################################################################################
class ROBOT(SUPERVISOR):
    def __init__(self,SUPERVISOR,name):
        self.SUPERVISOR=SUPERVISOR # address of supervisor hass passed here
        self.ground=self.SUPERVISOR.ground # grounds will also have same address since it is array
        self.robotName=name
        self.paramReductionMethod=self.SUPERVISOR.paramReductionMethod
        super().sharedParams()
        self.ground=0 # initilizing for robots
        self.rotation=0
        self.rotation2B=0
        self.position=[0,0]
        self.groundSensorValue=0
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


        ''' start from full eaxploration and complete learning '''
        self.epsilon=np.zeros(np.shape(self.Qtable))+self.RLparams['epsilon']
        self.alpha=np.zeros(np.shape(self.Qtable))+self.RLparams['alpha']

        x=np.arange(0,len(self.delta[0]))
        self.delta[0,x%2==0]=255
        self.deltaDot[0,x%2==0]=255
        self.DELTA[0,x%2==0]=255
        self.epsilon[0,x%2==0]=255

        self.printFlag=True if self.robotName=='0' and self.printFlag else False # only robot 0 will talk 
# move ...............................................................................................................................  
    def move(self):
            self.rotation=self.rotation2B
            self.position[0]=self.position[0]+self.velocity*sin(np.radians(self.rotation))*self.timeStep
            self.position[1]=self.position[1]+self.velocity*cos(np.radians(self.rotation))*self.timeStep

            self.position[0]=min(self.position[0],self.Ylen-1)
            self.position[1]=min(self.position[1],self.Xlen-1)

            self.position[0]=max(self.position[0],0+1)
            self.position[1]=max(self.position[1],0+1)
# groundSense ...............................................................................................................................
    def groundSense(self):
        temp=self.ground[int(round(self.position[1])),int(round(self.position[0]))]
        ''' if dist(self.position,[self.Xlen//4,self.Ylen//2])<=self.cueRadius: '''
        ''' for dynamic env, x condition is deleted ''' ####
        if self.position[0]<= self.Ylen-self.SUPERVISOR.visibleRaduis and self.position[0]>= self.SUPERVISOR.visibleRaduis:

            self.groundSensorValue=255-temp[0]
        else: self.groundSensorValue=0
# aggregate ...............................................................................................................................
    def aggregate(self):
        if any(self.SUPERVISOR.flagsR[int(self.robotName)]) and self.groundSensorValue>0 and not self.delayFlag :
            self.waitingTime=self.SUPERVISOR.Wmax*((self.groundSensorValue**2)/((self.groundSensorValue**2) + 5000))
            self.delayFlag=True
# detectQR ...............................................................................................................................
    def detectQR(self):
        for QRpos in self.QRloc:
            if dist(self.position-np.array(self.QRloc[QRpos]))<=self.QRdetectableArea :
                self.detectedQR=QRpos
                break
            else:
                self.detectedQR='QR0'
# RL ...............................................................................................................................
    def RL(self):
        if self.inAction==False :
            self.state=int(self.detectedQR[-1])
            if self.paramReductionMethod=='classic':
                eps=self.RLparams['epsilon']
            elif self.paramReductionMethod=='adaptive' or self.paramReductionMethod=='adaptive shut':
                eps=np.mean(self.epsilon[self.state]) #########
            elif self.paramReductionMethod=='adaptive united':
                eps=np.mean(self.epsilon[1:]) # shared eps. state 0 discarded
            else: raise NameError('[-] method could not be found')

            
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
# actAndReward ...............................................................................................................................
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
        elif (self.inAction==True and dist(self.position-self.desiredPos)<=20) or rewardInp!=None:
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
            # self.deltaDot[x,y]=np.sign(self.deltaDot[x,y])
            # self.deltaDot[x,y] = self.deltaDot[x,y] if self.deltaDot[x,y]!=0 else 1
            self.DELTA[x,y]=self.delta[x,y]/self.RLparams['maxdiff']
            if self.printFlag and self.paramReductionMethod!='classic':
                print(colored('\t[+] time {} index {} reward {} Qtable {} delta {} prevdelta {} deltaDot {} DELTA {}','white')\
                    .format(self.SUPERVISOR.getTime(),[x,y],self.reward,self.Qtable[x,y],\
                        self.delta[x,y],self.prevdelta[x,y],self.deltaDot[x,y],self.DELTA[x,y]))
            self.prevdelta[x,y]=self.delta[x,y]
            self.updateRLparameters()

            ''' update rule '''
            if self.paramReductionMethod=='classic':
                self.Qtable[x,y]+=self.RLparams['alpha']*(self.reward-self.Qtable[x,y]) ########################
            elif self.paramReductionMethod=='adaptive':
                self.Qtable[x,y]+=self.alpha[x,y]*(self.reward-self.Qtable[x,y]) ########################
            elif self.paramReductionMethod=='adaptive united' or self.paramReductionMethod=='adaptive shut':
                # self.Qtable[x,y]+=np.mean(self.alpha[x,y])*(self.reward-self.Qtable[x,y]) ########################
                self.Qtable[x,y]+=self.alpha[x,y]*(self.reward-self.Qtable[x,y]) ########################



            self.QtableCheck[x,y]=1
            self.exploredAmount=(self.QtableCheck==1).sum()/np.size(self.Qtable[1:]) # discarding data of state 0


            self.rewardMemory.append(self.reward)
# updateRLparameters ...............................................................................................................................
    def updateRLparameters(self):
        if self.paramReductionMethod=='classic':
            self.RLparams["epsilon"]*=self.EpsilonDampRatio ####
        else :
            if self.paramReductionMethod=='adaptive':
                beforeEps=self.epsilon[self.state,self.actionIndx]
                beforeAlpha=self.alpha[self.state,self.actionIndx]
            elif self.paramReductionMethod=='adaptive united':
                beforeEps=np.mean(self.epsilon)
                beforeAlpha=np.mean(self.alpha)
            elif self.paramReductionMethod=='adaptive shut':
                beforeEps=np.mean(self.epsilon[self.state])
                beforeAlpha=np.mean(self.alpha[self.state])

            ''' sensitivity must be multiplied not divided '''

            if self.paramReductionMethod=='adaptive' or self.paramReductionMethod=='adaptive united':
                self.epsilon[self.state,self.actionIndx]=saturate(self.DELTA[self.state,self.actionIndx]*self.RLparams["sensitivity"])
                self.alpha[self.state,self.actionIndx]=self.epsilon[self.state,self.actionIndx]/2 ####
            elif self.paramReductionMethod=='adaptive shut':
                # if np.any(self.deltaDot[self.state]>0): # does delta of any action for a specific state increases?
                if self.deltaDot[self.state,self.actionIndx]>0: # does delta of any action for a specific state increases?
                    # self.epsilon[self.state]=1 # all values of epsilon for all actions for this specific state will be one 
                    # self.alpha[self.state]=0.5 # same scenario for alpha
                    self.epsilon[self.state,self.actionIndx]=1 # all values of epsilon for all actions for this specific state will be one 
                    self.alpha[self.state,self.actionIndx]=0.5 # same scenario for alpha

                    shut=True
                else:
                    self.epsilon[self.state,self.actionIndx]=saturate(self.DELTA[self.state,self.actionIndx]*self.RLparams["sensitivity"])
                    self.alpha[self.state,self.actionIndx]=self.epsilon[self.state,self.actionIndx]/2 ####
                    shut=False
                    if self.printFlag:  # for debugging
                        print()
            if self.printFlag:
                if self.paramReductionMethod=='adaptive':
                    print('\t[+] eps: {}->{} alpha: {}->{}\n'\
                    .format(round(beforeEps,3),round(self.epsilon[self.state,self.actionIndx],3),round(beforeAlpha,3),round(self.alpha[self.state,self.actionIndx],3)))
                elif self.paramReductionMethod=='adaptive united':
                    print('\t[+] eps: {}->{} alpha: {}->{}\n'\
                    .format(round(beforeEps,3),round(np.mean(self.epsilon),3),round(beforeAlpha,3),round(np.mean(self.alpha),3)))
                elif self.paramReductionMethod=='adaptive shut':
                    print('\t[+] eps: {}->{} alpha: {}->{} shut {}\n'\
                    # .format(round(beforeEps,3),round(np.mean(self.epsilon[self.state]),3),round(beforeAlpha,3),round(np.mean(self.alpha[self.state]),3),shut))
                    .format(round(beforeEps,3),round(self.epsilon[self.state,self.actionIndx],3),round(beforeAlpha,3),round(self.alpha[self.state,self.actionIndx],3),shut))

