from SupervisorUltraLib import *
init()
# exit()
dcount=0
stp=0
dif=0
stp=np.zeros((1,ROBN))[0]

t=sup.getTime()
# print("t",t)
# tt=time()
# while(sup.getTime()-t<5): print(".",sup.getTime()-t)
# tp=sup.getTime()
# print("tp",tp)
# print(round(tp-t))
# exit()

print(V,"*****hiiixi____kkk__dfg____","method",method,"sigma",sigma,"static",static,"Etol",Etol,"ROBN",ROBN,"Lx",Lx,"Ly",Ly,"TIMESTEP",TIMESTEP,"Wmax",Wmax,stime,"recorde_flag",recorde_flag)
QR=" "
c=0
S_flag=True
time_mem=sup.getTime()
while sup.step(timestep) != -1:
    # print(sup.getTime())
    if method==True:
        # if Pflag: print("\n STATE",STATE,"STATE2",STATE2,"MOD",MOD,"QRmem",QRmem,QR,"STOTAL",Stotal[0]*rad2deg,e[0])
        for i in range(0,ROBN):
            # if i==0: Pflag=True
            # else: Pflag=False

        # for i in range(0,1):
            POS=getPos()
            pos=POS[i]
            ROT=getRot()
            rot=ROT[i]
            if Pflag: print("\n STATE",STATE,"STATE2",STATE2,"MOD",MOD,"QRmem",QRmem,QR,"STOTAL",Stotal[0]*rad2deg,e[0])

            check_field(i,pos)

            POS=getPos()
            pos=POS[i]
            ROT=getRot()
            rot=ROT[i]

            if MOD[i]=="delay": 
                dcount+=1
                # if dcount<2:
                if dcount<0:
                    go(i,POS,ROT)
                else :
                    MOD[i]=" "
                    dcount=0

            elif MOD[i]=="wait":
                w=wait(i,POS)
                if (sup.getTime() - stp[i]) >= w:
                    MOD[i]="in colision w r"
                    STATE[i]="avoid col"
            else:
                check_col(i,POS,rot)
                if MOD[i]=="in colision w w" :
                    avoid_col(i,POS,rot)
                    
                    if STATE2[i]=="going":
                        e[i,int(QRmem[i][-1])-1]+=1
                        if Pflag: print("errrrrrrooooorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
                        if e[i,int(QRmem[i][-1])-1]>= Etol:
                            e[i,int(QRmem[i][-1])-1]=0
                            Stotal[i][int(QRmem[i][-1])-1]=0
                            if Pflag: print("********************************************** reseted")
                        STATE2[i]=" "

                elif MOD[i]=="in colision w r":
                    if STATE2[i]== "going": STATE2[i]=" "
                    if STATE[i]=="avoid col":
                        STATE[i]=" "
                        avoid_col(i,POS,rot)
                        
                    else : 
                        wait(i,POS)
                        if STATE[i]== "in cue":
                            MOD[i]="wait"
                            stp[i]=sup.getTime()
                            if STATE2[i]=="seen QR" :
                                calculate_Stotal(i,pos)
                                STATE2[i]=" "
                            # if Pflag: print("e",e)
                        else: 
                            avoid_col(i,POS,rot)
                            
                else:
                    QR=checkQR(i,pos)
                    if QR!= "Q0" and STATE2[i]!='going':
                        QRmem[i]=QR
                        angle=Stotal[i][int(QR[-1])-1]
                        if angle!=0:
                            angle=angle+np.pi
                            lentt=Stotal_len[i][int(QR[-1])-1]
                            Stotal_vec=[lentt*sin(angle),lentt*cos(angle)]
                            RQ=QRlocs[int(QR[-1])-1]-pos
                            Sdes=[0,0]
                            Sdes[0]=Stotal_vec[0]+RQ[0]
                            Sdes[1]=Stotal_vec[1]+RQ[1]
                            angle=atan2(Sdes[0],Sdes[1])
                            while angle>=2*np.pi: angle=angle-2*np.pi
                            while angle<0: angle=angle+2*np.pi
                            sup.setSFRotation([0,1,0,angle+Noise()],i)
                            if Pflag: print("--------------------------------------------------------------------------------",angle)
                            STATE2[i] = "going"

    
                        else:
                            if STATE2[i] != "going":
                                STATE2[i] = "seen QR"
                            else: pass
                go(i,POS,ROT)

            
            if sup.getTime()-t>2 and not round(sup.getTime())%SAMPELING_PERIOD:
                loger(POS,ROT)
                t=sup.getTime()
                c+=1
                if c>=1000 and  S_flag: print("----",t);c=0;S_flag=False

            if sup.getTime()-time_mem>=capture_rate:
                time_mem=sup.getTime()
                # print(sup.getTime())
                vis(POS,ROT)
            
#..................................................................................................................................................................
    if method== False:
        for i in range(0,ROBN):
            POS=getPos()
            pos=POS[i]
            ROT=getRot()
            rot=ROT[i]
            if Pflag: print("\n STATE",STATE,"STATE2",STATE2,"MOD",MOD,"QRmem",QRmem,QR,"STOTAL",Stotal[0]*rad2deg,e[0])

            # check_groundfield(i,pos)
            check_field(i,pos)
            POS=getPos()
            pos=POS[i]
            ROT=getRot()
            rot=ROT[i]

            if MOD[i]=="delay": 
                dcount+=1
                if dcount<2:
                    go(i,POS,ROT)
                else :
                    MOD[i]=" "
                    dcount=0

            elif MOD[i]=="wait":
                w=wait(i,POS)
                if (sup.getTime() - stp[i]) >= w:
                    MOD[i]="in colision w r"
                    STATE[i]="avoid col"
            else:
                check_col(i,POS,rot)
                if MOD[i]=="in colision w w" :
                    avoid_col(i,POS,rot)

                elif MOD[i]=="in colision w r":
                    if STATE[i]=="avoid col":
                        STATE[i]=" "
                        avoid_col(i,POS,rot)
                        
                    else : 
                        wait(i,POS)
                        if STATE[i]== "in cue":
                            MOD[i]="wait"
                            stp[i]=sup.getTime()
                        else: 
                            avoid_col(i,POS,rot)
                            
                go(i,POS,ROT)
            if sup.getTime()-t>2 and not round(sup.getTime())%(SAMPELING_PERIOD):
                loger(POS,ROT)
                t=sup.getTime()
                c+=1
                if c>=1000 and  S_flag: print("----",t);c=0;S_flag=False
            
            if sup.getTime()-time_mem>=capture_rate:
                time_mem=sup.getTime()
                # print(sup.getTime())
                vis(POS,ROT)
