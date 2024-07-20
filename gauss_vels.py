# Created on Tue Nov  2 03:33:39 2023

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import pylab as pyl
from scipy.optimize import minimize

csfont = {'fontname':'Serif', 'fontsize':'20'}

###############################################################################
source = "J1740emis" # The source;
projection = 0      # 0 --- standard projection; 1 --- inside-out projection;
fluxmod = 0          # 0 --- standard mode; 1 --- flux-modulated mode;
model_overlap=0      # 0 --- no model; 1 --- overlap doppmap by the model;
polar_grid = 0
caster_grid = 1
#title = 'V455 And, H$_{\\delta}$'
title = 'V455 And, HeI$\\lambda4471$'
#cmap = matplotlib.cm.get_cmap('gist_stern_r')
cmap = matplotlib.cm.get_cmap('hot_r')
vels0 = np.array([500, 1000, 1500])
trim_intes = 1
imax = 0.99; imin=0.1
###############################################################################

path = 'sources/' + source + '/out/doptomog/'
if fluxmod == 0 and projection == 0:
    path = path + 'doptomog.0.out'
    print(2)
elif fluxmod == 0 and  projection == 1:
    path = path + 'doptomog.1.out'
    print(1)
elif fluxmod == 1 and projection == 0:
    path = path + 'fluxmodmap/' + 'fluxmodmap.0.out'
elif fluxmod == 1 and projection == 1:
    path = path + 'fluxmodmap/' + 'fluxmodmap.1.out'
else:
    sys.exit('Error in input parameters')

def loaddata(file):
    f = open(file)
    ##########################################################################
    parts = f.readline().split()
    vmax=float(parts[0]); vf=float(parts[1]); nps=int(parts[2]); proj=int(parts[3])  
    ##########################################################################
    parts = f.readline().split()
    lll=float(parts[0]); nvm=int(parts[1]); npm=int(parts[2]); nvp=int(parts[3])  
    ##########################################################################
    parts = f.readline().split()
    vs, gam, w0 = float(parts[0]), float(parts[1]), float(parts[2])
    ##########################################################################
    f.close()
    
    f = open(file)
    lines = f.readlines()
    phases, n_end_line = readnumbers(lines, 3, npm)
    dphases, n_end_line = readnumbers(lines, n_end_line+1, npm)
    vp, n_end_line = readnumbers(lines, n_end_line+1, nvp)
    fluxes_input, n_end_line = readnumbers(lines, n_end_line+1, nvm)
    fluxes_recon, n_end_line = readnumbers(lines, n_end_line+2, nvm)
    intes, n_end_line = readnumbers(lines, n_end_line+3, nps**2)
    dopmap = np.reshape(intes, (nps,nps))
    print(intes)
    return dopmap, vmax, intes

    
def readnumbers(lines,startline, partsnum):
    res=np.zeros(partsnum)
    n_end_line = 0
    k=0
    for i in range(startline, len(lines)):
        parts = lines[i].split()
        for j in range(len(parts)):
            res[k] = float(parts[j])
            k=k+1
        if k==partsnum:
            n_end_line=i
            break
    return res, n_end_line

def circcoo(vv, vmax, nps, projection):
    alphas = np.linspace(0, 2*np.pi, 300)
    xx = 0; yy = 0
    if projection == 0:
        for i in range(len(vv)):
            xx = nps/2+(nps*vv[i]/vmax/2)*np.sin(alphas)
            yy = nps/2+(nps*vv[i]/vmax/2)*np.cos(alphas)
            plt.text(nps/2+(nps*vv[i]/vmax/2)+nps*0.01, nps/2 - nps*0.04, str(vv[i]), **csfont, rotation=90)
            plt.plot(xx, yy, '--k', linewidth=1.0)
    else:
        for i in range(len(vv)):
            xx = nps/2+(nps/2-nps*vv[i]/vmax/2)*np.sin(alphas)
            yy = nps/2+(nps/2-nps*vv[i]/vmax/2)*np.cos(alphas)
            plt.text(nps/2+(nps/2-nps*vv[i]/vmax/2)+nps*0.01, nps/2 - nps*0.04, str(vv[i]), **csfont, rotation=90)
            plt.plot(xx, yy, '--k', linewidth=1.0)
    return xx, yy

def cootrans(mas, nps, vmax, proj):
    #print(mas)
    vx = mas[:,0]
    vy = mas[:,1]
    
    vxb = []
    vyb = []
    for i in range(len(vx)):
        if np.sqrt(vx[i]**2+vy[i]**2)*1000<vmax:
            vxb.append(vx[i])
            vyb.append(vy[i])
    vx = np.array(vxb)
    vy = np.array(vyb)
    
    mod = np.sqrt(vx**2+vy**2)
    for i in range(len(mod)):
            if mod[i]*1000 > vmax:
                mod[i]=vmax
    xx = 0; yy = 0

    if proj == 0:
        xx=nps/2+nps*vx*1000/vmax/2
        yy=nps/2+nps*vy*1000/vmax/2

    if proj == 1:
        mod = (nps/2-nps*mod*1000/vmax/2)
        theta = np.arctan(vy/vx)
        for i in range(len(theta)):
            if vx[i] < 0:
                theta[i] = theta[i] + np.pi
        xx=nps/2+mod*np.cos(theta)
        yy=nps/2+mod*np.sin(theta)
    
    return np.array([xx, yy]).transpose()

def plotmodel(proj):

    path0 = 'sources/' + source + '/out/binarymodel/idl/'
    path = path0 + 'vStreamBal.out'
    data = np.loadtxt(path)
    res = cootrans(data[:,:2], nps, vmax, proj)
    vxb = res[:,0]
    vyb = res[:,1]
    
    path = path0 + 'vPrimary.out'
    data = np.loadtxt(path)
    res = cootrans(data[:,:2], nps, vmax, proj)
    vxp = res[:,0]
    vyp = res[:,1]

    path = path0 + 'vSecondary.out'
    data = np.loadtxt(path)
    res = cootrans(data[:,:2], nps, vmax, proj)
    vxs = res[:,0]
    vys = res[:,1]
  
    path = path0 + 'vCOM.out'
    data = np.loadtxt(path)
    res = cootrans(np.array([data[:2]]), nps, vmax, proj)
    vxc = res[0,0]
    vyc = res[0,1]
      
    path = path0 + 'vBinary.out'
    data = np.loadtxt(path)
    res = cootrans(data[:,:2], nps, vmax, proj)
    vxbin = res[:,0]
    vybin = res[:,1]
    
    path = 'sources/' + source + '/out/binarymodel/gnu/vStreamMag.out'
    f = open(path, 'r')
    f.readline()
    ldata = []
    ldata.append([])
    iii = 0
    while True:
        line = f.readline()
        if line == "":
            ldata = ldata[:-1]
            break
        parts = line.split()
        if len(parts) >= 3:
            ldata[iii].append([float(parts[0]), float(parts[1])])
        else:
            ldata.append([])
            iii = iii+1
            continue
    f.close()
    vxm=[]
    vym=[]
    for i in range(len(ldata)):
        res = cootrans(np.array(ldata[i]), nps, vmax, proj)
        vxm.append(res[:,0])
        vym.append(res[:,1])

    
    path = path0 + 'vStreamMagConnect.out'
    data = np.loadtxt(path)
    res = cootrans(data[:,:2], nps, vmax, proj)
    vxcon = res[:,0]
    vycon = res[:,1]
    
    nmag = int(len(vxcon)/2)
    
    plt.plot(vxb, vyb, '-r', linewidth=1)
    plt.plot(vxp, vyp, '--r', linewidth=1)
    plt.plot(vxs, vys, '-r', linewidth=1)
    plt.plot(vxc, vyc, 'xr')
    plt.plot(vxbin, vybin, '+r')
    for i in range(len(vxm)):
        print(vxm[i])
        plt.plot(vxm[i], vym[i], '--', markersize=5, color="b")
    for i in range(nmag):
        plt.plot([vxcon[i*2], vxcon[i*2+1]], [vycon[i*2], vycon[i*2+1]], '.--r', linewidth=0.5)
        
    return vxb, vyb, vxp, vyp, vxs, vys, vxc, vyc, vxbin, vybin
dopmap, vmax, intes= loaddata(path)
plt.imshow(dopmap)
plt.show()
XX = []
YY = []
In = []
# 2013
 
# Hb
#v = 1.0e-16 
#spot = 5.3e-16

# Hgam
#v = 0.5e-16 
#spot = 2.4e-16

# Hdel
#v = 0.5e-17 
#spot = 4.9e-17

# HeI
#v = 2.6e-17 
#spot = 7e-17

# HeII
#v = 5e-18 
#spot = 2.5e-17


# 2016

# Hb
#v = 2e-15 #for leastsq 
#v = 1e-15 #but let`s add more points for gauss fitting
#spot = 3e-15

# Hgam
#v = 0.1e-15 
#spot = 3e-15

# Hdel
#v = 0.1e-15 
#spot = 2e-15

# HeI
v = 2e-16
spot = 6e-16
# HeII
#v = 0.5e-17 
#spot = 2.2e-16

for i in range(len(dopmap[0])):
    for j in range(len(dopmap[0])):
        p = dopmap[i][j]
        if (p > v) and (p < spot):
           XX.append(j)
           YY.append(i)
           In.append(dopmap[i][j])

I = []
Xal = []
Yal = []
for i in range(len(dopmap[0])):
    for j in range(len(dopmap[0])):
        p = dopmap[i][j]
        
        #if (j >= 45 and j <=85 and i >=45 and i <=90):
            
        #if (j >=50 and j <= 80 and i >=30 and i <= 110):
        if (p < 3e-16) and p > 8e-17:
            I.append(dopmap[i][j])
            Xal.append(j)
            Yal.append(i)
        #elif (i >= 30 and i <=50 and j>=30 and j < 50):
            #if (p < 6e-16):
             #   I.append(dopmap[i][j])
             #   Xal.append(j)
             #   Yal.append(i)

XXsp = []
YYsp = []
Isp = []
for i in range(len(dopmap[0])):
    for j in range(len(dopmap[0])):
        #p = dopmap[i][j]
        #if (p > spot):
           XXsp.append(j)
           YYsp.append(i)
           Isp.append(dopmap[i][j])


nps = len(dopmap[0])
vmax = vmax/1000

ticks=0
if projection == 0:
    vels = np.concatenate([-vels0, vels0])    
    ticks = nps/2+vels*nps/vmax/2
else:
    vels = np.concatenate([-vels0, np.flip(vels0)])    
    ticks1 = vels0*nps/vmax/2
    ticks2 = nps - vels0*nps/vmax/2
    ticks = np.concatenate([ticks1, np.flip(ticks2)])
    
csfont = {'fontname':'Serif', 'fontsize':'20'}
value = -100
for i in range(nps):
    for j in range(nps):
        if ((i-nps/2)**2+(j-nps/2)**2) > (nps/2)**2:
            dopmap[i,j] = -100
dopmap = np.ma.masked_where(dopmap == value, dopmap)
cmap.set_bad(color='white')

imax0 = np.max(dopmap); imin0 = np.min(dopmap)
imin1 = (imax0-imin0)*imin+imin0
imax1 = (imax0-imin0)*imax+imin0

fig = plt.figure(figsize=(8,6))
if trim_intes:
    plt.imshow(dopmap, cmap=cmap, origin="lower", vmax=imax1, vmin=imin1)
else:
    plt.imshow(dopmap, cmap=cmap, origin="lower")

vels0 = np.array([50, 500, 1000, 1500])

if projection == 0:
    if polar_grid:
        if model_overlap:
            plotmodel(0)
        circcoo(vels0, vmax, nps, 0)
        #xx, yy = circcoo(vels0, vmax, nps, 0)
        plt.hlines(nps/2, -0.05*nps, nps+0.05*nps, linestyles='--', linewidth=0.5)
        plt.vlines(nps/2, -0.05*nps, nps+0.05*nps, linestyles='--', linewidth=0.5)
        plt.xlim([-0.05*nps, nps+0.05*nps])
        plt.ylim([-0.05*nps, nps+0.05*nps])
        plt.text(nps + nps*0.01, nps/2 + 0.02*nps, '0$\degree$', **csfont)
        plt.text(nps/2-nps*0.08, nps+0.02*nps, '90$\degree$', **csfont)
        plt.text(-nps*0.115, nps/2 - 0.045*nps, '180$\degree$', **csfont)
        plt.text(nps/2+nps*0.01, -nps*.045, '270$\degree$', **csfont)
        plt.xlabel('($\\theta$, v)')
        plt.axis('off')
    if caster_grid:
        if model_overlap:
            plotmodel(0)
        xx, yy = circcoo(vels0, vmax, nps, 0)
        plt.hlines(nps/2, -0.05*nps, nps+0.05*nps, linestyles='--', linewidth=0.5)
        plt.vlines(nps/2, -0.05*nps, nps+0.05*nps, linestyles='--', linewidth=0.5)
        plt.axis('on')
        plt.xticks(ticks = ticks, labels=vels)
        plt.yticks(ticks = ticks, labels=vels)
        plt.xlabel('$V_X$, km s$^{-1}$', **csfont)
        plt.ylabel('$V_Y$, km s$^{-1}$', **csfont)
        plt.tick_params(axis="both", direction="in", right='on', top='on')
        plt.xlim([-0.05*nps, nps+0.05*nps])
        plt.ylim([-0.05*nps, nps+0.05*nps])
if projection == 1:
    if model_overlap:
        plotmodel(1)
    circcoo(vels0, vmax, nps, 1)
    xx, yy = circcoo(vels0, vmax, nps, 1)
    plt.hlines(nps/2, -0.05*nps, nps+0.05*nps, linestyles='--', linewidth=0.5)
    plt.vlines(nps/2, -0.05*nps, nps+0.05*nps, linestyles='--', linewidth=0.5)
    plt.xlim([-0.05*nps, nps+0.05*nps])
    plt.ylim([-0.05*nps, nps+0.05*nps])
    plt.text(nps + nps*0.01, nps/2 + 0.02*nps, '0$\degree$', **csfont)
    plt.text(nps/2-nps*0.08, nps+0.02*nps, '90$\degree$', **csfont)
    plt.text(-nps*0.115, nps/2 - 0.045*nps, '180$\degree$', **csfont)
    plt.text(nps/2+nps*0.01, -nps*.045, '270$\degree$', **csfont)
    plt.axis('off')

cbar = plt.colorbar(orientation="vertical")
#cbar.set_ticks([0.8e-15, 2.4e-15, 3.8e-15]) #Hb_2016
#cbar.set_ticks([0.1e-16, 0.55e-16, 1e-16])
#cbar.set_ticklabels(['$low$', '$medium$', '$high$'])
x_m = np.mean(xx)
y_m = np.mean(yy)
#x_c = 69.5
#y_c = 69.5

def calc_R(x_c, y_c): 
    """Рассчитать расстояние между точкой данных s и центром (x_c, y_c)"""
    return np.sqrt((Xal-x_c)**2 + (Yal-y_c)**2)

def f_2(c, In):
    """Рассчитать остаточный радиус"""
    Ri = calc_R(*c)
    return (Ri - np.mean(Ri))*np.sqrt(In)

center_estimate = x_m, y_m
center_2, ier = leastsq(f_2, center_estimate, args=(I))

xc_2, yc_2 = center_2
Ri_2       = calc_R(*center_2)
R_2        = Ri_2.mean()
residu_2   = sum((Ri_2 - R_2)**2)
print('start R = ', R_2)
print('\n')
#cbar = plt.colorbar(shrink=0.7, ticks=[0, 2.5e-16, 5e-16])
#cbar.ax.set_yticklabels(['Low', 'Medium', 'High'])
#plt.title('(v, $\\vartheta$) [km s$^{-1}$, degrees]', y=-0.1, **csfont)
#plt.suptitle(title, **csfont, y=0.97, x=0.5)
#plt.savefig('HeI4471_ins.png')


angles2 = np.linspace(0,2*np.pi,100,True)
x2 = xc_2 + np.cos(angles2) * R_2
y2 = yc_2 + np.sin(angles2) * R_2
'''
plt.plot(XX, YY, '.', color = 'salmon')
plt.plot(x2, y2 ,"r-", lw=3)
plt.plot(xc_2, yc_2, '.', color='red')
plt.suptitle(title, **csfont, x = 0.5, y=0.85)
plt.show()
'''

def gauss(A, sig, R, xc, yc, x, y):
    return (A*np.exp(-(np.sqrt((x - xc)**2 + (y - yc)**2) - R)**2/(2*(sig**2))))

def chi2(pars):
    s = 0
    for i in range(0, len(r)):
        s = s + (I[i] - gauss(pars[0], pars[1], pars[2], pars[3], pars[4], Xal[i], Yal[i]))**2
    return s

pars = [np.max(I), 5, R_2, xc_2, yc_2] 
r = np.sqrt((Xal-xc_2)**2 + (Yal-yc_2)**2)
res = minimize(chi2, pars, method='Nelder-Mead')
print(res.x[0], res.x[1], res.x[2], res.x[3], res.x[4])

'''
plot accretion disk on tomogram
'''

xc_3 = res.x[3]
yc_3 = res.x[4]
R_3 = res.x[2]

x_ent = xc_3 + np.cos(0) * (R_3-res.x[1])
y_ent = yc_3 + np.sin(0) * (R_3-res.x[1])
x_out = xc_3 + np.cos(0) * (R_3+res.x[1])
y_out = yc_3 + np.sin(0) * (R_3+res.x[1])
 
r = np.sqrt((Xal - xc_3)**2 + (Yal - yc_3)**2)
#r_ent = np.sqrt((x_ent - xc_3)**2 + (y_ent - yc_3)**2)
#r_out = np.sqrt((x_out - xc_3)**2 + (y_out - yc_3)**2)
r_out = np.sqrt((x_ent - xc_3)**2 + (y_ent - yc_3)**2)
r_ent = np.sqrt((x_out - xc_3)**2 + (y_out - yc_3)**2)
x_cir, y_cir = [], []

for i in range(0, len(r)):
    if (r[i] >= r_ent and r[i] <= r_out):
        x_cir.append(Xal[i])
        y_cir.append(Yal[i])

plt.plot(x_cir, y_cir, '.', color = 'salmon')    
plt.plot(xc_3, yc_3, '.', color='salmon')
plt.suptitle(title, **csfont, x = 0.45, y=0.85)
plt.show()

# скорости центра масс БК
xe = (xc_3 - nps/2)*2*vmax/nps
ye = (yc_3 - nps/2)*2*vmax/nps
print('veloc = ', np.sqrt(xe**2+ye**2))


'''
ploting O-C diagram
'''

plt.figure(figsize=(8,6))
X, Y, x, y, OC, oc = [], [], [], [], [], []

for i in range (1, len(Yal)):
    if (Yal[i] == Yal[i-1]):
        x.append(Xal[i-1])
        y.append(Yal[i-1])
        oc.append(I[i-1] - gauss(res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], Xal[i-1], Yal[i-1]))
    else:
        x.append(Xal[i-1])
        y.append(Yal[i-1])
        oc.append(I[i-1] - gauss(res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], Xal[i-1], Yal[i-1]))
        OC.append(oc)
        X.append(x)
        Y.append(y)
        oc = []
        x = []
        y = [] 
oc.append(I[i-1] - gauss(res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], Xal[i-1], Yal[i-1]))
x.append(Xal[i])
y.append(Yal[i])
OC.append(oc)
X.append(x)
Y.append(y)


plt.pcolor(X, Y, OC, cmap='RdGy_r')
cbar = plt.colorbar(label="O-C", orientation="vertical")
#plt.clim(-3.5e-16, 3.5e-16) 
#cbar.set_ticks([-3.5e-16, 0, 3.5e-16]) 
#cbar.set_ticklabels(['-0.5e-16', '0', '0.5e-16']) #Hgam2013
cmap.set_bad(color='white')
if projection == 0:
    if polar_grid:
        if model_overlap:
            plotmodel(0)
        circcoo(vels0, vmax, nps, 0)
        #xx, yy = circcoo(vels0, vmax, nps, 0)
        plt.hlines(nps/2, -0.05*nps, nps+0.05*nps, linestyles='--', linewidth=0.5)
        plt.vlines(nps/2, -0.05*nps, nps+0.05*nps, linestyles='--', linewidth=0.5)
        plt.xlim([-0.05*nps, nps+0.05*nps])
        plt.ylim([-0.05*nps, nps+0.05*nps])
        plt.text(nps + nps*0.01, nps/2 + 0.02*nps, '0$\degree$', **csfont)
        plt.text(nps/2-nps*0.08, nps+0.02*nps, '90$\degree$', **csfont)
        plt.text(-nps*0.115, nps/2 - 0.045*nps, '180$\degree$', **csfont)
        plt.text(nps/2+nps*0.01, -nps*.045, '270$\degree$', **csfont)
        plt.xlabel('($\\theta$, v)')
        plt.axis('off')
    if caster_grid:
        if model_overlap:
            plotmodel(0)
        xx, yy = circcoo(vels0, vmax, nps, 0)
        plt.hlines(nps/2, -0.05*nps, nps+0.05*nps, linestyles='--', linewidth=0.5)
        plt.vlines(nps/2, -0.05*nps, nps+0.05*nps, linestyles='--', linewidth=0.5)
        plt.axis('on')
        plt.xticks(ticks = ticks, labels=vels)
        plt.yticks(ticks = ticks, labels=vels)
        plt.xlabel('$V_X$, km s$^{-1}$', **csfont)
        plt.ylabel('$V_Y$, km s$^{-1}$', **csfont)
        plt.tick_params(axis="both", direction="in", right='on', top='on')
        plt.xlim([-0.05*nps, nps+0.05*nps])
        plt.ylim([-0.05*nps, nps+0.05*nps])
if projection == 1:
    if model_overlap:
        plotmodel(1)
    circcoo(vels0, vmax, nps, 1)
    xx, yy = circcoo(vels0, vmax, nps, 1)
    plt.hlines(nps/2, -0.05*nps, nps+0.05*nps, linestyles='--', linewidth=0.5)
    plt.vlines(nps/2, -0.05*nps, nps+0.05*nps, linestyles='--', linewidth=0.5)
    plt.xlim([-0.05*nps, nps+0.05*nps])
    plt.ylim([-0.05*nps, nps+0.05*nps])
    plt.text(nps + nps*0.01, nps/2 + 0.02*nps, '0$\degree$', **csfont)
    plt.text(nps/2-nps*0.08, nps+0.02*nps, '90$\degree$', **csfont)
    plt.text(-nps*0.115, nps/2 - 0.045*nps, '180$\degree$', **csfont)
    plt.text(nps/2+nps*0.01, -nps*.045, '270$\degree$', **csfont)
    plt.axis('off')

plt.suptitle(title, **csfont, x = 0.45, y=0.85)


'''
#cheking chi2 of estimated parameters
pars = [res.x[0]+0.01e-15, res.x[1]+0.01, res.x[2]+0.01]
print(chi2(pars))
pars = [res.x[0], res.x[1], res.x[2]]
print(chi2(pars))
'''








