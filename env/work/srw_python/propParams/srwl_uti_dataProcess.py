import uti_plot_com as srw_io
import numpy as np
from math import *
from numpy.fft import *

#Data processing functions

#Read and plot generic SRW .dat files created
def read_srw_file(filename):
    data, mode, ranges, labels, units = srw_io.file_load(filename)
    data = np.array(data).reshape((ranges[8], ranges[5]), order='C')
    return {'data': data,
            'shape': data.shape,
            'mean': np.mean(data),
            'photon_energy': ranges[0],
            'horizontal_extent': ranges[3:5],
            'vertical_extent': ranges[6:8],
            # 'mode': mode,
            'labels': labels,
            'units': units}

#RMS beam size calculation
def rmsfile(file):
    flux0=read_srw_file(file)
    data2D=flux0['data']
    datax=np.sum(data2D,axis=1) 
    datay=np.sum(data2D,axis=0)
    hx = flux0['horizontal_extent']
    hy = flux0['vertical_extent']
    shape=flux0['shape']
    xmin=hx[0]
    xmax=hx[1]
    ymin=hy[0]
    ymax=hy[1]
    Nx=shape[0]
    Ny=shape[1]
    dx = (xmax-xmin)/Nx
    dy = (ymax-ymin)/Ny
    xvals = np.linspace(xmin,xmax,Nx) 
    yvals = np.linspace(ymin,ymax,Ny)
    sxsq=sum(datax*xvals*xvals)/sum(datax) 
    xavg=sum(datax*xvals)/sum(datax)
    sx=sqrt(sxsq-xavg*xavg)

    sysq=sum(datay*yvals*yvals)/sum(datay) 
    yavg=sum(datay*yvals)/sum(datay)
    sy=sqrt(sysq-yavg*yavg)
    return sx, sy

#transform SRW intensity file format to matrix style
def transformSRWIntensityFile(filein,fileout):
    flux0=read_srw_file(filein)
    data2D=flux0['data']
    datax=np.sum(data2D,axis=1) 
    datay=np.sum(data2D,axis=0)
    hx = flux0['horizontal_extent']
    hy = flux0['vertical_extent']
    shape=flux0['shape']
    xmin=hx[0]
    xmax=hx[1]
    ymin=hy[0]
    ymax=hy[1]
    Nx=shape[0]
    Ny=shape[1]
    dx = (xmax-xmin)/Nx
    dy = (ymax-ymin)/Ny
    header1='#xmin,xmax,Nx = '+str(xmin)+','+str(xmax)+','+str(Nx)+'\n'
    header2='#ymin,ymax,Ny='+str(ymin)+','+str(ymax)+','+str(Ny)
    head=header1+header2
    np.savetxt(fileout,data2D,delimiter=',',header=head,comments="")
    #np.savetxt("results/intx_1mmAperture", datax, delimiter=',', header="Intensity", comments="")
    
    
#RMS beam size calculation function
def rmsdata2D(data2D,xmin,xmax,ymin,ymax,Nx,Ny):
    datax=np.sum(data2D,axis=1) 
    datay=np.sum(data2D,axis=0)
    dx = (xmax-xmin)/Nx
    dy = (ymax-ymin)/Ny
    xvals = np.linspace(xmin,xmax,Nx) 
    yvals = np.linspace(ymin,ymax,Ny)
    sxsq=sum(datax*xvals*xvals)/sum(datax) 
    xavg=sum(datax*xvals)/sum(datax)
    sx=sqrt(sxsq-xavg*xavg)
    sysq=sum(datay*yvals*yvals)/sum(datay) 
    yavg=sum(datay*yvals)/sum(datay)
    sy=sqrt(sysq-yavg*yavg)
    return sx, sy

#Intensity calculation function
def wfrGetIntensity(data2D,xmin,xmax,ymin,ymax,Nx,Ny):
    dx=(xmax-xmin)/Nx
    dy=(ymax-ymin)/Ny
    intensity=np.sum(data2D)*dx*dy
    return intensity

#Polarization from wavefront calculation function
def wfrGetPol(wfr):
    dx=(wfr.mesh.xFin-wfr.mesh.xStart)/wfr.mesh.nx
    dy=(wfr.mesh.yFin-wfr.mesh.yStart)/wfr.mesh.ny
    arReEx=wfr.arEx[::2]
    arImEx=wfr.arEx[1::2]
    arReEy=wfr.arEy[::2]
    arImEy=wfr.arEy[1::2]
    arReEx2d = np.array(arReEx).reshape((wfr.mesh.nx, wfr.mesh.ny), order='C')
    arImEx2d=np.array(arImEx).reshape((wfr.mesh.nx, wfr.mesh.ny), order='C')
    arReEy2d=np.array(arReEy).reshape((wfr.mesh.nx, wfr.mesh.ny), order='C')
    arImEy2d=np.array(arImEy).reshape((wfr.mesh.nx, wfr.mesh.ny), order='C')
    xvals=np.linspace(wfr.mesh.xStart,wfr.mesh.xFin,wfr.mesh.nx)
    yvals=np.linspace(wfr.mesh.yStart,wfr.mesh.yFin,wfr.mesh.ny)
    ReEx00=interpBright(0,0,arReEx2d,wfr.mesh.xStart,wfr.mesh.yStart,dx,dy,wfr.mesh.nx,wfr.mesh.ny)
    ImEx00=interpBright(0,0,arImEx2d,wfr.mesh.xStart,wfr.mesh.yStart,dx,dy,wfr.mesh.nx,wfr.mesh.ny)
    ReEy00=interpBright(0,0,arReEy2d,wfr.mesh.xStart,wfr.mesh.yStart,dx,dy,wfr.mesh.nx,wfr.mesh.ny)
    ImEy00=interpBright(0,0,arImEy2d,wfr.mesh.xStart,wfr.mesh.yStart,dx,dy,wfr.mesh.nx,wfr.mesh.ny)
    norm=math.sqrt(ReEx00**2+ImEx00**2+ReEy00**2+ImEy00**2)  ##Normalization so that abs(Re[Pvec])^2+abs(Im[Pvec])^2=1
    Pvec=(1/norm)*np.array([ReEx00+ImEx00*(1j), ReEy00+ImEy00*(1j)])
    return Pvec
    
def calcWigner(Earray,xarrayE,qvals):
    """
    #compute Wigner function (2-D) from 1-D electric field array
    #Earray is electric field, assumed to be normalized such that int E^*E(x) dx =1
    #xarrayE is the set of x values the electric field is defined on
    #qvals are the momentum values to compute W on.  q=theta/lambda
    #If Integral(E*(x)E(x)dx) = 1 then Integral(W(x,q)dxdq) = 1.
    """
    
    xmaxE = np.max(xarrayE)
    N=np.size(xarrayE)
    dx=xarrayE[1]-xarrayE[0]
    dq = qvals[1]-qvals[0]
    
    Eq=np.fft.fft(Earray)
    Eqmat=np.tile(np.matrix(Eq).T,np.size(xarrayE))
    phimat=np.matrix(xarrayE)

    shiftmatplus=np.exp(-np.pi*1j*np.matrix(qvals).T*phimat) ##divide by two to get phi/2 shift
    shiftmatminus=np.exp(np.pi*1j*np.matrix(qvals).T*phimat) ##divide by two to get phi/2 shift
    Eqshiftplus=np.multiply(Eqmat,shiftmatplus)
    Eqshiftminus=np.multiply(Eqmat,shiftmatminus)
    Exshiftplus=np.fft.ifft(Eqshiftplus,n=None,axis=0)
    Exshiftminus=np.conjugate(np.fft.ifft(Eqshiftminus,n=None,axis=0))
    
    gmatxphi=np.fft.fftshift(np.multiply(Exshiftplus,Exshiftminus),0)
    
    wmat=fft(np.fft.fftshift(np.real(gmatxphi)),n=None,axis=1,norm=None)
    wmatshift=np.fft.fftshift(wmat,axes=1)
    wmatshiftnorm=wmatshift/(N*dq)  ##Normalization factor required (1/(N*dq) = dx)
    return(wmatshiftnorm)


def WignerToE(Wvals,xvals,qvals):
    """
    #Reconstruct electric field from Wigner function with output E*(2x)E(0)
    #Wvals is array of Wigner function values with xvals and qvals
    #xvals and qvals are arrays of sorted x and q values
    """
    
    xq0mat=np.matrix(qvals).T*np.matrix(xvals)
    xqmat=np.exp((4*np.pi)*1j*xq0mat)
    Wxq=np.multiply(Wvals,xqmat.T)
    dq = qvals[1]-qvals[0]
    return np.sum(Wxq,axis=1)*dq
   
