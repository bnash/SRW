from __future__ import print_function #Python 2.7 compatibility
from srwl_uti_dataProcess import *
import sys
sys.path.insert(0,'/home/vagrant/src/bnash/SRW/env/work/srw_python')
from propParamLib import*
import numpy as np, numpy.ma as ma
import math
import os 
import sys
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

##Create Gaussian wavefront
#sigr: requested RMS beam size [m]
#propLen: propagation length required by SRW to create numerical Gaussian [m]
#pulseE: energy per pulse [J]
#poltype: polarization type (0=linear horizontal, 1=linear vertical, 2=linear 45 deg, 3=linear 135 deg, 4=circular right, 5=circular left, 6=total)
#phE: photon energy [eV]
#sampFact: sampling factor to increase mesh density

sigrW = 10e-6
propLen = 20  ##20
zStart = 0
pulseE = .001
poltype = 1
phE = 10e3
nx=50
ny=nx
sampFact = 0 ##25, 44, 60
constConvRad = 1.23984186e-06/(4*3.1415926536)  ##conversion from energy to 1/wavelength
rmsAngDiv = constConvRad/(phE*sigrW)
mx = 0
my = 0
wfr0=createGsnSrc(sigrW,propLen,zStart,pulseE,poltype,phE,nx,ny,sampFact,mx,my)
#srwl.ResizeElecField(wfr0, 'c', [0, 16, .125, 16, .125])

print("SRW calculated: <xx'> = %s" %wfr0.arMomX[6])

print("grid size of initial wavefront from createGsnSrc function: %s x %s" %((wfr0.mesh.nx),((wfr0.mesh.ny))))

lvals = np.linspace(-150,150,2) 
lint = 0
srwsigxvals=[]
srwsigxpvals=[]
siganalyticvals=[]
pysigxvals=[]



for l in lvals:
        print('l value:',l)
        wfr0=createGsnSrc(sigrW,propLen,zStart,pulseE,poltype,phE,nx,ny,sampFact,mx,my)
        L=l
        ##Calculate expected RMS beam size after drift
        sxcalcL=np.sqrt(sigrW**2+(rmsAngDiv**2)*(L**2))
        siganalyticvals.append(sxcalcL)
        #print("Expected RMS beam size:",sxcalcL)
        wfr1 = deepcopy(wfr0)
        print("wavefront shape after deepcopy:%s %s" %(wfr1.mesh.nx,wfr1.mesh.ny))
        drift=createBLdrift(L,aResizeBefore=0,aResizeAfter=0,aResizePrec=1.,xRangeMod=5,xResMod=.1,yRangeMod=5,yResMod=.1)
        srwl.PropagElecField(wfr1, drift)
        print("wavefront shape after drift:%s %s" %(wfr1.mesh.nx,wfr1.mesh.ny))
        ##Extract <xx> and <xx'> from SRW wavefront object
        srwsigx = math.sqrt(wfr1.arMomX[5])
        srwsigxp = wfr1.arMomX[6]
        srwsigxvals.append(srwsigx)
        srwsigxpvals.append(srwsigxp)
        ##Calculate intensity from each wavefront
        arI2 = array('f', [0]*wfr1.mesh.nx*wfr1.mesh.ny) #"flat" array to take 2D intensity data
        srwl.CalcIntFromElecField(arI2, wfr1, 6, 0, 3, wfr1.mesh.eStart, 0, 0) #extracts intensity
        ##Reshaping electric field data from flat to 2D array
        arI22D = np.array(arI2).reshape((wfr1.mesh.nx, wfr1.mesh.ny), order='C')
        sx2, sy2 = rmsdata2D(arI22D,wfr1.mesh.xStart,wfr1.mesh.xFin,wfr1.mesh.yStart,wfr1.mesh.yFin,wfr1.mesh.nx,wfr1.mesh.ny)
        pysigxvals.append(sx2)
        #sigErr = sx2 - sxcalcL
        #sigErrvals.append(sigErr)
        lint+=1

##Plot <xx> comparison        
plt.rcParams.update({'legend.labelspacing':0.25, 'legend.handlelength': 2})
hfontLarge = {'fontname':'Latin Modern Roman', 'size' : 24, 'weight' : 'bold'}
plt.rcParams.update({'xtick.labelsize':20,'ytick.labelsize':20})

#fig = plt.figure(figsize=(14,9))
#ax = fig.gca()
#ax.plot(lvals,srwsigxvals, 'bo-',label=r'SRW <xx>',linewidth=6.)
#ax.plot(lvals,siganalyticvals, 'go-',label=r'Analytic <xx>',linewidth=6.)
#ax.plot(lvals,pysigxvals, 'r--',label=r'Python function <xx>',linewidth=6.)
  
#ax.set_ylabel(r'<xx> [m]',**hfontLarge)
#ax.set_xlabel(r'Drift Length [m]',**hfontLarge)
#ax.set_title('RMS Beam Size vs Rel. Change in Focal Length',**hfontLarge)
    
#ax.legend(loc='best',prop={'size': 22})
   
#ax.grid(color='k', linestyle='dashed', linewidth=1)


##Plot <xp> comparison
#plt.rcParams.update({'legend.labelspacing':0.25, 'legend.handlelength': 2})
#hfontLarge = {'fontname':'Latin Modern Roman', 'size' : 24, 'weight' : 'bold'}
#plt.rcParams.update({'xtick.labelsize':20,'ytick.labelsize':20})

#fig = plt.figure(figsize=(14,9))
#ax = fig.gca()
#ax.plot(lvals,srwsigxpvals, 'bo-',label=r'SRW <xp>',linewidth=6.)

  
#ax.set_ylabel(r'<xp> [m*rad]',**hfontLarge)
#ax.set_xlabel(r'Drift Length [m]',**hfontLarge)
#ax.set_title('RMS Beam Size vs Rel. Change in Focal Length',**hfontLarge)
    
#ax.legend(loc='best',prop={'size': 22})
   
#ax.grid(color='k', linestyle='dashed', linewidth=1)

print("SRW calculated: <xx'> = %s" %wfr0.arMomX[6])
print("SRW calculated: <xx'> values = %s" %srwsigxpvals)
print("Drift lengths: %s" %lvals)


