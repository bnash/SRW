from __future__ import print_function #Python 2.7 compatibility
from srwl_uti_dataProcess import *
import sys
sys.path.insert(0,'/home/vagrant/src/bnash/SRW/env/work/srw_python')
from propParamLib import*
import numpy as np, numpy.ma as ma
import math
import os
import time

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
nx=550
ny=nx
sampFact = 0 ##25, 44, 60
constConvRad = 1.23984186e-06/(4*3.1415926536)  ##conversion from energy to 1/wavelength
mx = 0
my = 0
wfr0=createGsnSrc(sigrW,propLen,zStart,pulseE,poltype,phE,nx,ny,sampFact,mx,my)
#srwl.ResizeElecField(wfr0, 'c', [0, 16, .125, 16, .125])

##Initial wfr calculation
arI1 = array('f', [0]*wfr0.mesh.nx*wfr0.mesh.ny) #"flat" array to take 2D intensity data
srwl.CalcIntFromElecField(arI1, wfr0, 6, 0, 3, wfr0.mesh.eStart, 0, 0) #extracts intensity

##Reshaping electric field data from flat to 2D array
arI12D = np.array(arI1).reshape((wfr0.mesh.nx, wfr0.mesh.ny), order='C')
wfrshapei=np.shape(arI12D)
wfrsizei=np.size(arI12D)
print('Shape of initial wavefront data array (coordinate):',wfrshapei)
print('Size of initial wavefront data array (coordinate):',wfrsizei)
xvals1=np.linspace(wfr0.mesh.xStart,wfr0.mesh.xFin,wfr0.mesh.nx)
yvals1=np.linspace(wfr0.mesh.yStart,wfr0.mesh.yFin,wfr0.mesh.ny)

sx1,sy1=rmsdata2D(arI12D,wfr0.mesh.xStart,wfr0.mesh.xFin,wfr0.mesh.yStart,wfr0.mesh.yFin,wfr0.mesh.nx,wfr0.mesh.ny)

print("RMS beam size at waist: %s, %s" %(sx1,sy1))

##Propagate wavefront through beamline numerical
L=20  ##Drift length
wfr1 = deepcopy(wfr0)
drift = createBLdrift(L,aResizeBefore=1,aResizeAfter=1,aResizePrec=1.,xRangeMod=1,xResMod=1,yRangeMod=1,yResMod=1)
srwl.PropagElecField(wfr1, drift)
##Calculate intensity from each wavefront
arI2 = array('f', [0]*wfr1.mesh.nx*wfr1.mesh.ny) #"flat" array to take 2D intensity data
srwl.CalcIntFromElecField(arI2, wfr1, 6, 0, 3, wfr1.mesh.eStart, 0, 0) #extracts intensity
##Reshaping electric field data from flat to 2D array
arI22D = np.array(arI2).reshape((wfr1.mesh.nx, wfr1.mesh.ny), order='C')

print('Size of final wavefront data array (coordinate):',np.shape(arI22D))
xvals2=np.linspace(wfr1.mesh.xStart,wfr1.mesh.xFin,wfr1.mesh.nx)
yvals2=np.linspace(wfr1.mesh.yStart,wfr1.mesh.yFin,wfr1.mesh.ny)

sx2, sy2 = rmsdata2D(arI22D,wfr1.mesh.xStart,wfr1.mesh.xFin,wfr1.mesh.yStart,wfr1.mesh.yFin,wfr1.mesh.nx,wfr1.mesh.ny)
print("RMS beam size after drift: %s, %s" %(sx2,sy2))
