import cv2 as cv
import numpy as np
import random as rnd
from math import sin,cos,sqrt,atan2
def m2px(inp):
    return int(inp*512/2)

def RotStandard(inp):
    if inp<0: inp+=360
    if inp>360: inp-=360
    return inp

def dist(x,y):
    return sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)

class SUPERVISOR:
    def __init__(self,FinalTime,samplingPeriod):
        self.timeStep=512*0.001
        self.fps=int(self.timeStep*1000)//10
        self.samplingPeriod=samplingPeriod
        self.FinalTime=FinalTime
        self.ROBN=10
        self.velocity=14
        self.ground=cv.imread('ground.png') # (1024,512)
        self.swarm=[0]*self.ROBN
        self.wallNum=4
        self.time=0
        self.flagsR=[[False for _ in range(self.ROBN)] for _i in range(self.ROBN)] # [[False]*self.ROBN]*self.ROBN >>this one has address problem<<
        self.counterR= [[0 for _ in range(self.ROBN)] for _i in range(self.ROBN)]# [[0]*self.ROBN]*self.ROBN >>this one has address problem<<

        self.collisionDelay=1
        self.detectRad=0.06
        self.robotRad=0.06
        self.robotSenseRad=m2px(self.robotRad+self.detectRad)
        self.Xlen=np.shape(self.ground)[0]
        self.Ylen=np.shape(self.ground)[1]
        self.cueRadius=m2px(0.7)
        self.Wmax=120
    
    def generateRobots(self):
        margin=m2px(self.robotRad)*5
        possibleY=np.linspace(0+margin,self.Xlen-margin,int(self.Xlen//2*self.robotSenseRad))
        possibleX=np.linspace(0+margin,self.Ylen-margin,int(self.Ylen//2*self.robotSenseRad))
        possibleRot=np.linspace(0,360,self.ROBN)
        for i in range(self.ROBN):
            self.swarm[i]=ROBOT(str(i))
            self.swarm[i].position=np.array([int(rnd.sample(list(possibleX),1)[0]),int(rnd.sample(list(possibleY),1)[0])])
            self.swarm[i].rotation=rnd.sample(list(possibleRot),1)[0]

    def visualize(self):
        background=np.copy(self.ground)
        for i in range(self.ROBN):
            for ii in range(len(self.swarm[i].position)): self.swarm[i].position[ii]=int(self.swarm[i].position[ii])

            
            cv.circle(background,tuple(self.swarm[i].position),self.robotSenseRad,(255,100,100),1)
            cv.circle(background,tuple(self.swarm[i].position),m2px(self.robotRad),(255,100,100),-1)

            direction=np.array([int(m2px(self.robotRad)*sin(np.radians(self.swarm[i].rotation))) \
                , int(m2px(self.robotRad)*cos(np.radians(self.swarm[i].rotation)))])
            cv.line(background,tuple(self.swarm[i].position),tuple(self.swarm[i].position+direction),(0,0,200),3)
            cv.putText(background,str(i),tuple([self.swarm[i].position[0]-10,self.swarm[i].position[1]+10]),cv.FONT_HERSHEY_SIMPLEX\
                ,0.75,(0,0,0),3 )
        cv.putText(background,str(int(self.time))+' s',(20,20),cv.FONT_HERSHEY_SIMPLEX,0.75,(0,100,0),3 )

        cv.imshow("background",background)
        cv.waitKey(self.fps)

    def moveAll(self):
        for i in range(self.ROBN):
            if self.swarm[i].delayFlag==False:
                self.swarm[i].move(self.velocity,self.timeStep,self.Xlen,self.Ylen)
            else :
                self.swarm[i].waitingTime-=self.timeStep
                if self.swarm[i].waitingTime<=0:
                    self.swarm[i].delayFlag=False
                    self.checkCollision(specific=True,robotNum=i)            
                    self.swarm[i].move(self.velocity,self.timeStep,self.Xlen,self.Ylen)

        self.time+=self.timeStep

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
                            self.swarm[i].rotation2B=rnd.randint(0,180);# |<
                    if Ycond1:
                        if 270<=self.swarm[i].rotation<=360 or 0<=self.swarm[i].rotation<=90:
                            self.swarm[i].rotation2B=rnd.randint(90,270);# _
                    if Ycond2:
                        if 90<=self.swarm[i].rotation<=270:
                            self.swarm[i].rotation2B=rnd.randint(270,360+90);# -

                    self.swarm[i].rotation2B=RotStandard(self.swarm[i].rotation2B)
            
            for i in range(0,self.ROBN):
                for j in range(0,self.ROBN):
                    if j!=i:
                        if dist(self.swarm[j].position,self.swarm[i].position)<=self.robotSenseRad*2 and self.flagsR[i][j]==False: 
                            collisionAngle=RotStandard(np.degrees(atan2( self.swarm[j].position[0]-self.swarm[i].position[0] , self.swarm[j].position[1]-self.swarm[i].position[1] )))
                            self.swarm[i].rotation2B=collisionAngle+rnd.randint(90,270) 
                            self.swarm[i].rotation2B=RotStandard(self.swarm[i].rotation2B)
                            self.flagsR[i][j]=True
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
                    if dist(self.swarm[j].position,self.swarm[i].position)<=self.robotSenseRad*2 and self.flagsR[i][j]==False: 
                        self.swarm[i].rotation+=rnd.randint(90,270)
                        collisionAngle=RotStandard(np.degrees(atan2( self.swarm[j].position[0]-self.swarm[i].position[0] , self.swarm[j].position[1]-self.swarm[i].position[1] )))
                        self.swarm[i].rotation=collisionAngle+rnd.randint(90,270) 
                        self.swarm[i].rotation=RotStandard(self.swarm[i].rotation)
                        self.flagsR[i][j]=False
                        break

    def getGroundSensors(self):
        for i in range(self.ROBN):
            self.swarm[i].groundSense(self.ground,self.Xlen,self.Ylen,self.cueRadius)

    def aggregateSwarm(self):
        for i in range(self.ROBN):
            self.swarm[i].aggregate(self.flagsR,self.Wmax)

    def getTime(self):
        return int(self.time)

    def getStatus(self):
        inCue=0
        for i in range(self.ROBN):
            if self.swarm[i].groundSensorValue>0:
                inCue+=1
        return inCue/self.ROBN

class ROBOT(SUPERVISOR):
    def __init__(self,name):
        # super().__init__()
        self.rotation=0
        self.rotation2B=0
        self.position=[0,0]
        self.groundSensorValue=0
        self.robotName=name
        self.waitingTime=0
        self.delayFlag=False
    def move(self,velocity,timeStep,Xlen,Ylen):
            self.rotation=self.rotation2B
            self.position[0]=self.position[0]+velocity*sin(np.radians(self.rotation))*timeStep
            self.position[1]=self.position[1]+velocity*cos(np.radians(self.rotation))*timeStep

            self.position[0]=min(self.position[0],Ylen-1)
            self.position[1]=min(self.position[1],Xlen-1)

            self.position[0]=max(self.position[0],0+1)
            self.position[1]=max(self.position[1],0+1)



    def groundSense(self,ground,Xlen,Ylen,cueRadius):
        temp=ground[self.position[1],self.position[0]]
        if dist(self.position,[Xlen//4,Ylen//2])<=cueRadius:
            self.groundSensorValue=255-temp[0]
        else: self.groundSensorValue=0

    def aggregate(self,flagsR,Wmax):
        if any(flagsR[int(self.robotName)]) and self.groundSensorValue>0 and not self.delayFlag :
            self.waitingTime=Wmax*((self.groundSensorValue**2)/((self.groundSensorValue**2) + 5000))
            self.delayFlag=True