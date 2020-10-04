from DataFillerNA20000_shade import *
from time import ctime , time
import statistics as st
sigma=0.08
c=str(1)
POP=10
FolderName="C:\\Users\\Aras\\Desktop\\LBA-DRL\\LBADRL\\old LBA codes\\plotter\\previous rad speed 0.05"
endtime=5000
SMAPLING_PERIOD=5
numel=int(endtime/SMAPLING_PERIOD)
# numel=1000*4

tav=20#int(500/20)
boxnum=int(numel/tav)
fig, ax10 = plt.subplots(1, 1,figsize=(18,8)) 
# cases=[0,0.3]
# cases=[100,200]
# cases=['BEECLUST','LBA Etol=inf','LBA Etol=6 sigma 0.1']
cases=[2,10,100]
cases=[r"LBA $\tau_{e}$ = "+str(_) for _ in cases]
for _ in range(len(cases)):
    if cases[_]==r"LBA $\tau_{e}$ = 100":
        cases[_]='BEECLUST'
    elif cases[_]=='LBA $E_{tol}$ = 200':
        cases[_]='LBA'

pltt=['tomato','green','royalblue','yellow',"grey","purple"]
#....................................................................................................................................
def plotter(pop):
    global rownum,numel,tav,boxnum,pltt,fig,ax,lst,cases,endtime
    rownum=20
    for c,v in enumerate(lst):
        lst[c]=np.delete(lst[c],slice(rownum,None),0)
        # print("---",c)

    lstshape=[len(lst),len(lst[0])]
    lst_av=np.zeros((lstshape[0],lstshape[1],boxnum))
    for c,v in enumerate(lst):
        lst[c]=lst[c]/pop
        for i in range(0,boxnum):
            for j in range(0,rownum):
                try:
                    lst_av[c,j,i]=sum(lst[c][j][i*tav:(i+1)*tav])/len(lst[c][j][i*tav:(i+1)*tav])
                except:
                    print("error",c,i,j)
                    exit()

    lst_av_sh=np.shape(lst_av)
    med=np.zeros((lst_av_sh[0],lst_av_sh[2]))
    q1=np.zeros((lst_av_sh[0],lst_av_sh[2]))
    q3=np.zeros((lst_av_sh[0],lst_av_sh[2]))

    for i in range(lst_av_sh[0]):
        for k in range(lst_av_sh[2]):
            med[i,k]=np.percentile(lst_av[i,:,k],50)
            q1[i,k]=np.percentile(lst_av[i,:,k],25)
            q3[i,k]=np.percentile(lst_av[i,:,k],75)
    dlen=len(med[0])
    x=list(range(dlen))
    for i in range(lst_av_sh[0]):
        if i==0:
            plt.plot(x,med[i],color=pltt[i],label=cases[i],linewidth=3,linestyle='--')
        if i==1:
            plt.plot(x,med[i],color=pltt[i],label=cases[i],linewidth=3,linestyle='-')
        if i==2:
            plt.plot(x,med[i],color=pltt[i],label=cases[i],linewidth=3,linestyle=':')

        plt.fill_between(x,med[i],q1[i],color=pltt[i],alpha=0.2)
        plt.fill_between(x,q3[i],med[i],color=pltt[i],alpha=0.2)

        # l=np.mean(med[i,-10:])
        # print(">> SS",l,med[i,-10:])
        # L=np.zeros((1,len(med[i])))[0]
        # L=L+l
        # plt.plot(x,L,'r--',color=pltt[i])
        
        # L=np.zeros((1,len(med[i])))[0]
        # L=L+l*0.9
        # plt.plot(x,L,'r--',color=pltt[i],alpha=1)

        # L=np.zeros((1,len(med[i])))[0]
        # L=L+l*1.1
        # plt.plot(x,L,'r--',color=pltt[i],alpha=1)
    xx=np.arange(0,dlen+5,(160))
    print(xx)
    xxt=xx*endtime/dlen
    xxt=np.array([int(_) for _ in xxt])
    plt.xticks(xx,xxt,fontweight='bold',fontsize=30)
    plt.yticks(fontweight='bold',fontsize=30)
    plt.ylim(0,1)
    plt.xlim(0,dlen)
    ax10.annotate('', xy=(xx[len(xx)//4], 0.8),  xycoords='data',
                xytext=(xx[len(xx)//4], 0.99), textcoords='data',
                # arrowprops=dict(facecolor='yellow', shrink=0.05),
                arrowprops=dict(facecolor='yellow', edgecolor='yellow',width=10,headwidth=40,headlength=40),
                horizontalalignment='left', verticalalignment='bottom',
                )

    # ax10.annotate('', xy=(xx[len(xx)//2], 0.8),  xycoords='data',
    #             xytext=(xx[len(xx)//2], 1), textcoords='axes fraction',
    #             arrowprops=dict(facecolor='yellow', shrink=0.05),
    #             horizontalalignment='left', verticalalignment='bottom',
    #             )

    # from matplotlib.patches import ConnectionPatch
    # xy = (20000, 0.2)
    # con = ConnectionPatch(xyA=xy, xyB=xy, coordsA="data", coordsB="data",
    #                     axesA=ax10, axesB=ax10)
    # ax10.add_artist(con)
    with open('C:\\Users\\Aras\\Desktop\\LBA-DRL\\LBADRL\\oldB.npy','wb') as f:
        np.save(f,lst_av)
#--------------------------------------------------------------------------------------------------------------------
ax=ax10
lst,rownum=ten(FolderName,numel)

plotter(POP)
strng=c+"MOD Steup 4, Dynamic env, shade, sigma "+str(sigma)+", duration "+str(endtime)+"s, 20 robots"+ctime(time()).replace(':','_')
plt.xlabel("Time [s]",fontweight='bold',fontsize=30)
plt.ylabel("Normalized aggregation size",fontweight='bold',fontsize=30)
# plt.subplots_adjust(top=0.97,bottom=0.15,left=0.1,right=0.96)
plt.subplots_adjust(top=0.9,bottom=0.15,left=0.1,right=0.96)

plt.legend(fontsize=30,loc=4,ncol=3)
plt.title('(a)',fontweight='bold',fontsize=25)
# plt.title('(a)',fontweight='bold',fontsize=30)

print(strng)
os.chdir("..")
# plt.savefig(strng+".pdf")
# plt.savefig("/home/arash/Desktop/performance_vs_Etol"+ctime(time()).replace(':','_')+".pdf")
# plt.savefig("/home/arash/Desktop/fig2_2 sigma 0.08 no title.pdf")

plt.show()
