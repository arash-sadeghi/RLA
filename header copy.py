import cv2 as cv
import numpy as np
import random as rnd
from math import sin,cos
def m2px(inp):
    return int(inp*512/2)


class SUPERVISOR:
    def __init__(self):
        self.ROBN=10
        self.velocity=7
        self.ground=cv.imread('ground.png') # (1024,512)
        self.swarm=[0]*self.ROBN
        self.detectRad=0.06
        self.robotRad=0.06
        self.robotSenseRad=m2px(self.robotRad+self.detectRad)
        self.Xlen=np.shape(self.ground)[0]
        self.Ylen=np.shape(self.ground)[1]

        # cv.imshow('x',self.ground)
        # cv.waitKey()

    def generateRobots(self):
        margin=m2px(self.robotRad)*5
        possibleY=np.linspace(0+margin,self.Xlen-margin,int(self.Xlen//2*self.robotSenseRad))
        possibleX=np.linspace(0+margin,self.Ylen-margin,int(self.Ylen//2*self.robotSenseRad))
        possibleRot=np.linspace(0,360,self.ROBN)
        for i in range(self.ROBN):
            self.swarm[i]=ROBOT()
            self.swarm[i].position=np.array([int(rnd.sample(list(possibleX),1)[0]),int(rnd.sample(list(possibleY),1)[0])])
            self.swarm[i].rotation=rnd.sample(list(possibleRot),1)[0]
    def visualize(self):
        background=np.copy(self.ground)
        for i in range(self.ROBN):
            for ii in range(len(self.swarm[i].position)): self.swarm[i].position[ii]=int(self.swarm[i].position[ii])

            cv.circle(background,tuple(self.swarm[i].position),m2px(self.robotRad),(255,100,100),-1)
            direction=np.array([int(m2px(self.robotRad)*sin(np.radians(self.swarm[i].rotation))) \
                , int(m2px(self.robotRad)*cos(np.radians(self.swarm[i].rotation)))])
            cv.line(background,tuple(self.swarm[i].position),tuple(self.swarm[i].position+direction),(0,0,200),3)
            cv.putText(background,str(i),tuple([self.swarm[i].position[0]-10,self.swarm[i].position[1]+10]),cv.FONT_HERSHEY_SIMPLEX\
                ,0.75,(0,0,0),3 )

        cv.imshow("background",background)
        cv.waitKey()
        

    def moveAll(self):
        for i in range(self.ROBN):
            self.swarm[i].move()

    def dist(self,x,y):
        return sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)

    def chooseRandomRot(self):
        return rnd.randint(180,270)
        
    def checkCollision(self):
        for i in range(0,self.ROBN):
            Xcond=self.swarm[i].position[0]>=self.Ylen-self.robotSenseRad or self.swarm[i].position[0]<=0+self.robotSenseRad
            Ycond=self.swarm[i].position[1]>=self.Xlen-self.robotSenseRad or self.swarm[i].position[1]<=0+self.robotSenseRad
            if Ycond or Xcond: 
                self.swarm[i].rotation+=self.chooseRandomRot()

        # for i in range(0,self.ROBN):
        #     for j in range(0,self.ROBN):
        #         if j!=i:
        #             if dist(self.swarm[j].position,self.swarm[i].position)<=self.detectRad: 
        #                 MOD[i]="in colision w r"        

class ROBOT(SUPERVISOR):
    def __init__(self):
        super().__init__()
        self.rotation=0
        self.position=[0,0]

    def move(self):
        self.position[0]=self.position[0]+self.velocity*sin(np.radians(self.rotation))
        self.position[1]=self.position[1]+self.velocity*cos(np.radians(self.rotation))