from time import time 
from numpy.random import random,seed
import numpy as np
from math import sqrt,cos,sin,atan2,atan
import os,sys
import cv2 as cv
Time=0
sd=round(int(time())%1000)
seed(sd)

indp= not False
if indp:
    TIMESTEP=512*0.0001
    class Supervisor():
        global TIMESTEP,Time,LOCS,ROTS
        def hello(self):
            print("hello")
        def getBasicTimeStep(self):
            global TIMESTEP
            return TIMESTEP 
        def getFromDef(self,_):
            pass
        def getField(self,_):
            pass
        def getTime(self):
            return Time
        def step(self,_):
            global Time
            Time+=TIMESTEP
            return True
        def simulationReset(self):
            # exit()
            print("will be reseted")
            exit()
            pass
        def setSFRotation(self,d,i):
            global ROTS
            ROTS[i]=d[-1]

        def setSFVec3f(self,d,i):
            global LOCS
            # print("\n",i," before",LOCS[i])
            # print("d",d)
            LOCS[i]=[d[0],d[-1]]
            # print("AFTER\n",LOCS[i])
#    TIMESTEP=512*0.001

    sup=Supervisor()
else:
    from controller import Supervisor,Display, ImageRef
    sup=Supervisor()
    TIMESTEP=int(sup.getBasicTimeStep())

static=not True
sigma=0.05#0/180
Etol=6
ROBN=10
Ftime=5000
Htime=5000
method=not True
QRnum=6
recorde_flag=False
Lx=2
hl=0.31*(5/2)*sqrt(20/20)
Ly=2*Lx
backim=cv.imread(str(10)+"__1.png")

capture_rate=5
FPS=20
m2pix=512/2
stime=str(round(time()))
size=(int(Lx*m2pix),int(Ly*m2pix))
fourcc = cv.VideoWriter_fourcc(*'mp4v')
if recorde_flag: out = cv.VideoWriter('projectx'+stime+'.mp4',fourcc, FPS, size,True)

SAMPELING_PERIOD=5
timestep = TIMESTEP
strs=["_"+str(_) for _ in range(1,ROBN+1)]
defs=[sup.getFromDef(_) for _ in strs]
if not indp:
    fld=[_.getField("translation") for _ in defs]  
    rfld=[_.getField("rotation") for _ in defs]
Time=sup.getTime()
tottime=sup.getTime()
logname=stime+"_log.txt"
data=[]
count=0

if indp: V=0.05 #0.05*2#####################################################
else : V=0.0005*2

x=np.zeros((1,ROBN))[0]
y=np.zeros((1,ROBN))[0]

# stime=stime+str(determine_method())
stime=stime+str(True)
os.makedirs(stime)
robot_rad = 0.06
detect_rad = robot_rad + 0.06
ff=open(stime+"/"+"seed is "+str(sd)+" _ "+logname,"a")
fff=open(stime+"/"+"Stotal.txt","a")
ffff=open(stime+"/"+"e.txt","a")


rad2deg=180/np.pi
STATE2=[" " for _ in range(0,ROBN)]
STATE=[" " for _ in range(0,ROBN)]
MOD=[" " for _ in range(0,ROBN)]
QRmem=[" " for _ in range(0,ROBN)]
Stotal=np.zeros((ROBN,QRnum))
Stotal_len=np.zeros((ROBN,QRnum))

e=np.zeros((ROBN,QRnum))
LOCS=np.zeros((ROBN,2))
ROTS=np.zeros((ROBN,1))
Pflag=not True
changedF=False
Wmax=120
c1=Ly/4
if QRnum>6:
    QRlocs=[(0,-Ly/2),(Lx/2,-c1),(Lx/2,0),(Lx/2,c1),(0,Ly/2),(-(Lx/2),c1),(-(Lx/2),0),(-(Lx/2),-c1)]
else:
    QRlocs=[(Lx/2,-c1),(Lx/2,0),(Lx/2,c1),(-(Lx/2),c1),(-(Lx/2),0),(-(Lx/2),-c1)]


####################################################################################################################
def init():
    distribute()
    if not indp:
        dis=sup.getDisplay('display')
        # im=dis.imageLoad("Initial_background.png")
        im=dis.imageLoad("5__1.png")

        dis.imagePaste(im,0,0,False)        
####################################################################################################################
'''
# def delay(x):
#     if x=='inf':
#         while sup.step(timestep) != -1: pass
        
#     t=sup.getTime()
#     while sup.step(timestep) != -1: 
#         if sup.getTime()-t>x: return 1
'''
####################################################################################################################
def determine_method():
    # p=os.getcwd()
    p='/home/arash/Desktop/ph4/15 x/controllers/epuck'
    fm=open(p+"/counterr.txt",'r')
    cnt=int(fm.read())-1
    fm.close()
    if cnt>5: return True
    else: return False  
####################################################################################################################
def check_cue(pos):
    # global changedF
    # pos=getPos()
    # if changedF:
    #     return list(map(lambda x: True if abs(x[0])<0.56 and x[1]>0.24 and x[1]<1.37 else False,pos))

    # else:

    #     return list(map(lambda x: True if abs(x[0])<0.56 and x[1]>-1.37 and x[1]<-0.24 else False,pos))


    global ROBN
    tr=[]
    for i in range(0,ROBN): tr.append(check_ground(i,pos))
    return list(map(lambda x:True if x>1 else False,tr))
####################################################################################################################
def dist(x,y):
    return sqrt((x[0]-y[0])**2+(x[2]-y[2])**2)
####################################################################################################################
def dist2(x,y):
    return sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)
###########################################################################
def distribute():
    print("disributing....")
    fault_flag=False
    mem=[]    
    x=np.random.uniform(-Lx/2+0.2,Lx/2-0.2,ROBN)
    y=np.random.uniform(-Ly/2+0.2,Ly/2-0.2,ROBN)
    i=0

    # off=0.2#
    # x=[-Lx/2+off,-Lx/2+off,Lx/2-off,Lx/2-off]#
    # y=[-Ly/2+off,Ly/2-off,-Ly/2+off,Ly/2+off]#

    while i <ROBN:
        for j in range(0,min(len(mem),20)):
            if dist([x[i],0,y[i]],mem[j])<0.3:
                x[i]=np.random.uniform(-Lx/2+0.2,Lx/2-0.2)
                y[i]=np.random.uniform(-Ly/2+0.2,Ly/2-0.2)
                fault_flag=True
                break

        if dist([x[i],0,y[i]],[0,0,-Lx/2+0.2])<0.6:
            x[i]=np.random.uniform(-Lx/2+0.2,Lx/2-0.2)
            y[i]=np.random.uniform(-Ly/2+0.2,Ly/2-0.2)
            fault_flag=True

        if fault_flag:
            fault_flag=False
            continue
        mem.append([x[i],0,y[i]])
####
        pv=[x[i],0,y[i]]

        if indp: sup.setSFVec3f(pv,i)
        else: fld[i].setSFVec3f(pv)

        rot=[0,1,0,random()*2*np.pi] 
        if indp: sup.setSFRotation(rot,i)
        else: rfld[i].setSFRotation(rot)

        i+=1
    print("disributing done")

###########################################################################
def getPos():
    global LOCS,indp,defs
    if indp: return LOCS
    else: return [[_.getPosition()[0],_.getPosition()[2]] for _ in defs] 
###########################################################################
def getRot():
    global ROTS,indp,defs
    if indp: return ROTS
    else: return [_.getSFRotation()[-1] for _ in rfld] 
####################################################################################################################
def change_background():
    global backim,backimflag
    # dis=sup.getDisplay('display')

    # delay(0.1)
    # im=dis.imageLoad("Secondary_background.png")
    # im=dis.imageLoad("Initial_background.png")
    # im=dis.imageLoad("Initial_background_x.png")

    # dis.imagePaste(im,0,0,False)
    # if ROBN<=20: backim=cv.imread("Secondary_background.png")
    # else: backim=cv.imread(str(ROBN)+"__2.png")
    backim=cv.imread(str(ROBN)+"__2.png")
    print("ground changed")
####################################################################################################################
def loger(pos,rot):
    global Time,tottime,static,Ft,Stotal,e,backimflag,changedF

    incue=check_cue(pos)
    c=incue.count(True)
    
    data.append(c)
    Time=sup.getTime()
    for i in range(0,ROBN):    
        strng=str(round(sup.getTime()))+" "+str(i)+" "+str(pos[i][0])+" "+\
            str(pos[i][1])+" "
        if type(rot[i]) is float: strng+= str(rot[i])+" "+"\n"
        else: strng+= str(rot[i][0])+" "+"\n"

        strngfff=str(round(sup.getTime()))+" "+str(i)+" "+str(Stotal[i,:])+"\n"
        strngffff=str(round(sup.getTime()))+" "+str(i)+" "+str(e[i,:])+"\n"

        ff.write(strng)
        fff.write(strngfff)
        ffff.write(strngffff)
    
    if sup.getTime()-tottime>1000:
        tottime=sup.getTime()
        # if Pflag: print('.',round(sup.getTime()))
        print('.',round(sup.getTime()))
    
    if static and sup.getTime()>=Htime: 
        change_background()
        static=False
        changedF=True

    if sup.getTime()>=Ftime:
        if Pflag: print("\n----------------dynamic done-----------------\n")
        f = open(stime+"/"+stime+".txt", "w")
        f.write(str(data))
        f.close()
        ff.close()
        fff.close()
        ffff.close()
        if recorde_flag: out.release()
        sup.simulationReset()
#####################################################################################################
def check_ground(i,pos):
    global backim,Lx,Ly
    pix2m=512/2

    x,y=int((pos[i][0]+Lx/2)*pix2m),int((pos[i][1]+Ly/2)*pix2m)
    if y-1>=int(Ly*pix2m): print("========___========",pos[i][0],pos[i][1],x,y,Lx,Ly,np.shape(backim),int(Lx*pix2m),int(Ly*pix2m));return 0 
    if x-1>=int(Lx*pix2m): print("**********____*****",pos[i][0],pos[i][1],x,y,Lx,Ly,np.shape(backim),int(Lx*pix2m),int(Ly*pix2m));return 0 

    sen=backim[y-1,x-1][0]
    return sen
#####################################################################################################
def check_col(i,pos,rot):
    global MOD,detect_rad,Lx,Ly
    ROT=getRot()
    for j in range(0,ROBN):
        if j!=i:
            if dist2(pos[i],pos[j])<=detect_rad+0.05: 
                MOD[i]="in colision w r"
            # if dist2(pos[i],pos[j])<=detect_rad+0.05: 
            #     if not ( np.sign(cos(ROT[i]-np.pi/2))==np.sign(cos(ROT[j]-np.pi/2)) and\
            #          np.sign(sin(ROT[i]-np.pi/2))==np.sign(sin(ROT[j]-np.pi/2)) ):
            #         MOD[i]="in colision w r"
    
    while rot>=2*np.pi: rot=rot-2*np.pi
    while rot<0: rot=rot+2*np.pi

    pos=pos[i]
    if pos[0]>=Lx/2-detect_rad:
        if rot<np.pi and rot>0: MOD[i]="in colision w w"
    elif pos[0]<=Lx/2*(-1)+detect_rad:
        if rot>np.pi and rot<2*np.pi: MOD[i]="in colision w w"
    elif pos[1]>=Ly/2-detect_rad:
        if (rot>3*np.pi/2 and rot<2*np.pi) or (rot>0 and rot<np.pi/2) : MOD[i]="in colision w w"
    elif pos[1]<=Ly/2*(-1)+detect_rad:
        if (rot<3*np.pi/2 and rot>np.pi/2) : MOD[i]="in colision w w"
#####################################################################################################
def wait(i,pos):
    global MOD,Wmax
    sen=check_ground(i,pos)
    if sen>0:
        STATE[i]="in cue"
        Ws=Wmax*((sen**2)/((sen**2) + 5000))
        return Ws
    else : return 0
#####################################################################################################
def avoid_col(i,POS,rot):
    pos=POS[i]
    global MOD,detect_rad,rad2deg
    while rot>=2*np.pi: rot=rot-2*np.pi
    while rot<0: rot=rot+2*np.pi
    dir="x"
    # if i==0: if Pflag: print(MOD[i])
    if MOD[i]=="in colision w w":
        if pos[0]>=Lx/2-detect_rad:
            if rot>np.pi/2 and rot<np.pi: dir="r"
            elif rot<np.pi/2 and rot>0: dir="l"
            # if i==0: if Pflag: print("i")
        elif pos[0]<=-Lx/2+detect_rad:
            if rot>3*np.pi/2: dir="r"
            elif rot<3*np.pi/2 and rot>np.pi: dir="l"
            # if i==0: if Pflag: print("ii")
        elif pos[1]>=Ly/2-detect_rad:
            if rot<np.pi/2: dir="r"
            elif rot>3*np.pi/2: dir="l"
            # if i==0: if Pflag: print("iii")

        elif pos[1]<=-Ly/2+detect_rad:
            if rot>np.pi: dir="r"
            elif rot<np.pi : dir="l"
            # if i==0: if Pflag: print("iv")


        # if i==0 :if Pflag: print("before",rot*rad2deg)
        rndrot=(random()*np.pi/2+np.pi/2)
        if dir=="r": rot=rot+rndrot
        elif dir=="l": rot=rot-rndrot
        elif dir=="x":rot=rot
        while rot>=2*np.pi: rot=rot-2*np.pi
        while rot<0: rot=rot+2*np.pi

        if indp: sup.setSFRotation([0,1,0,rot],i)
        else: rfld[i].setSFRotation([0,1,0,rot])
        MOD[i]="delay"
        # if i==0 :if Pflag: print("rotated",rndrot*rad2deg,rot*rad2deg,dir)


    elif MOD[i]=="in colision w r":
        J=0
        for j in range(0,ROBN):
            if j != i:
                if dist2(POS[i],POS[j])<=0.12+0.05:
                    J=j
                    break

        if J != i:
        
            if (POS[J][0]-POS[i][0])==0:
                if POS[J][0]>0: orient=np.pi/2
                else: orient=3*np.pi/2
            else : orient= atan((POS[J][1]-POS[i][1])/(POS[J][0]-POS[i][0]))#+np.pi/2 #+np.pi

            if POS[i][0]>POS[J][0]: orient=3*np.pi/2-orient
            else: orient=np.pi/2-orient 
            # if Pflag: print(i,J,"orient",orient*rad2deg) ###########################################
            # exit()
            df=abs(orient-rot)
            if df>180: df=2*np.pi-df
            
            # if df>np.pi/2: dir='x'
            if orient>rot: dir='r'    
            else: dir='l'    
            # if i==0: 
                # if Pflag: print(i,J,"orient",orient*180/3.14,-POS[i][1]+POS[J][1],POS[i][0]-POS[J][0],dir,df*rad2deg)
                # exit()

            # if i==0 :if Pflag: print("before",rot*rad2deg)
            rndrot=(random()*np.pi/2+np.pi/2)
            if dir=="r": rot=orient+rndrot# rot=rot+rndrot
            elif dir=="l": rot=orient-rndrot # rot=rot-rndrot
            elif dir=="x":rot=rot
            while rot>=2*np.pi: rot=rot-2*np.pi
            while rot<0: rot=rot+2*np.pi

            if indp: sup.setSFRotation([0,1,0,rot],i)
            else: rfld[i].setSFRotation([0,1,0,rot])
            # if i==0 :if Pflag: print("rotated",rndrot*rad2deg,rot*rad2deg,dir)
        MOD[i]="delay"
#####################################################################################################
def go(i,pos,rot):
    global x,y,V,timestep
    rot=getRot()
    x[i]=pos[i][0]+V*sin(rot[i])*timestep
    y[i]=pos[i][1]+V*cos(rot[i])*timestep
    if indp: sup.setSFVec3f([x[i],0,y[i]],i)
    else: fld[i].setSFVec3f([x[i],0,y[i]])
#####################################################################################################
def check_field(i,pos):
    global x,y,Lx,Ly
    min_field_x= Lx/2*(-1)+detect_rad;max_field_x= Lx/2-detect_rad
    min_field_y= Ly/2*(-1)+detect_rad;max_field_y= Ly/2-detect_rad

    x[i]=min(max_field_x,max(pos[0],min_field_x))
    y[i]=min(max_field_y,max(pos[1],min_field_y))
    if indp: sup.setSFVec3f([x[i],0,y[i]],i)
    else: fld[i].setSFVec3f([x[i],0,y[i]])
#.....................................................................................................................
def checkQR(i,pos):
    global ROBN,Lx,Ly,hl,QRnum
    c1=Ly/4
    if abs(pos[0])<hl and pos[1]<-(Ly/2-hl) :
        if QRnum>6: return 'Q1'
        else : return 'Q0'

    if pos[0]>Lx/2-hl and pos[1]> -(c1+hl) and pos[1]< -(c1-hl):
        if QRnum>6: return 'Q2'
        else : return 'Q1'
    if pos[0]>Lx/2-hl and abs(pos[1])<hl:
        if QRnum>6: return 'Q3'
        else : return 'Q2'
    elif pos[0]>Lx/2-hl and pos[1]> (c1-hl) and pos[1]< (c1+hl):
        if QRnum>6: return 'Q4'
        else : return 'Q3'
    elif abs(pos[0])<hl and pos[1]>(Ly/2-hl):
        if QRnum>6: return 'Q5'
        else : return 'Q0'
    elif pos[0]<-(Lx/2-hl) and pos[1]> (c1-hl) and pos[1]< (c1+hl):
        if QRnum>6: return 'Q6'
        else : return 'Q4'
    elif pos[0]<-(Lx/2-hl) and abs(pos[1])<hl:
        if QRnum>6: return 'Q7'
        else : return 'Q5'
    elif pos[0]<-(Lx/2-hl) and pos[1]> -(c1+hl) and pos[1]< -(c1-hl):
        if QRnum>6: return 'Q8'
        else : return 'Q6'
    else: return 'Q0'
#.....................................................................................................................
def calculate_Stotal(i,pos):
    # if Pflag: print("---",pos)
    # exit()
    global Stotal,rad2deg,QRmem,Stotal_len,QRlocs
    c1=Ly/4
    

    # QRlocs=[[0,-1.65],[0.65,-1.3],[0.65,0],[0.65,1.3],[0,1.65],[-0.65,1.3],[-0.65,0],[-0.65,-1.3]]
    # QRlocs=[(0,-(Ly/2-hl)),(Lx/2-hl,-c1),(Lx/2-hl,0),(Lx/2-hl,c1),(0,(Ly/2-hl)),(-(Lx/2-hl),c1),(-(Lx/2-hl),0),(-(Lx/2-hl),-c1)]

    QRR=int(QRmem[i][-1])-1
    QRpos=QRlocs[QRR]
    if (QRpos[0]-pos[0])==0:
        if QRpos[1]-pos[1]>0: orient=np.pi
        else: orient=0
    else : 
        orient= atan((QRpos[1]-pos[1])/(QRpos[0]-pos[0]))
    
    lent=sqrt((QRpos[1]-pos[1])**2+(QRpos[0]-pos[0])**2)
    
    if pos[0]>QRpos[0]: orient=3*np.pi/2-orient
    else: orient=np.pi/2-orient 
    Stotal[i][QRR]=orient
    Stotal_len[i][QRR]=lent
    if Pflag: print("i",i,"Stotal",Stotal[i]*rad2deg,"pos",pos,"QRpos",QRpos)
#.....................................................................................................................
def Noise():
    global sigma
    noise= (1-2*random())*np.pi
    return noise*sigma
#.....................................................................................................................
def vis(POS,ROT):
    if recorde_flag==False:
        return 0
    F=False
    m2pix=512/2
    robot_rad_px=int(robot_rad*m2pix)
    # imarena=np.zeros((int(Lx*m2pix),int(Ly*m2pix)))
    # imarena=backim
    imarena = np.empty_like (backim)
    imarena[:] = backim
    for i in range(ROBN):
        
        rob_cen=(int((POS[i,0]+Lx/2)*m2pix),int((POS[i,1]+Ly/2)*m2pix))
        # rob_cen=(int((POS[i,1]+Ly/2)*m2pix),int((POS[i,0]+Lx/2)*m2pix))


        # rob_edge=(int(rob_cen[0]+robot_rad_px*cos(ROT[i]-np.pi/2)),int(rob_cen[1]+robot_rad_px*sin(ROT[i]-np.pi/2)))
        rob_edge=(int(rob_cen[0]+robot_rad_px*sin(ROT[i])),int(rob_cen[1]+robot_rad_px*cos(ROT[i])))

        # rob_edge=(int(rob_cen[1]+10*sin(ROT[i]-np.pi/2)),int(rob_cen[0]+10*cos(ROT[i]-np.pi/2)))
        
        if STATE2[i]=="going":
            cv.circle(imarena,rob_cen,robot_rad_px,(0,255,0),-1)
        else:
            cv.circle(imarena,rob_cen,robot_rad_px,(255,0,0),-1)

        cv.line(imarena,rob_cen,rob_edge ,0,thickness=1)
    
    cv.putText(imarena,str(int(sup.getTime())),(200,200),cv.FONT_HERSHEY_SIMPLEX,3,(255,255,255),5,cv.LINE_AA)
    
    # cv.imshow("imarenax",imarena)

    # cv.waitKey(5)
    out.write(imarena)
    # print("-->",sup.getTime(),"STATE2[0]",STATE2[0],"STATE[0]",STATE[0],"MOD[0]",MOD[0])















