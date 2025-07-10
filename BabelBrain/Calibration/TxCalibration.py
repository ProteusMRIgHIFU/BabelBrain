import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import pandas as pd
import yaml
import re
import os
from openpyxl.utils import column_index_from_string
from matplotlib.colors import TwoSlopeNorm
from openpyxl.utils import column_index_from_string
from openpyxl import load_workbook
from BabelViscoFDTD.H5pySimple import SaveToH5py, ReadFromH5py
sys.path.append('/Users/spichardo/Documents/GitHub/BabelBrain/')
from TranscranialModeling import BabelIntegrationANNULAR_ARRAY
from BabelViscoFDTD.tools.RayleighAndBHTE import InitCuda,InitOpenCL, InitMetal,ForwardSimple
from scipy.optimize import minimize


def read_excel_range(file_path, range_str, sheet_name=0, **kwargs):
    """
    Reads a rectangular Excel range (e.g., 'T20:AJ81') into a DataFrame.

    Parameters:
        file_path (str): Path to the Excel file.
        range_str (str): Excel-style range string (e.g., 'T20:AJ81').
        sheet_name (str or int): Sheet name or index (default is 0).
        **kwargs: Additional arguments passed to `pd.read_excel()`.

    Returns:
        pd.DataFrame: Data from the specified Excel range.
    """
    # Parse range like 'T20:AJ81'
    match = re.match(r'^([A-Z]+)(\d+):([A-Z]+)(\d+)$', range_str.upper())
    if not match:
        raise ValueError("Invalid range format. Use Excel-style range like 'T20:AJ81'.")

    col_start, row_start, col_end, row_end = match.groups()
    row_start, row_end = int(row_start), int(row_end)
    
    # Compute parameters
    skiprows = row_start - 1
    nrows = row_end - row_start 
    usecols = f"{col_start}:{col_end}"
    print('usecols', usecols)

    # Read the DataFrame
    df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        skiprows=skiprows,
        nrows=nrows,
        usecols=usecols,
        **kwargs
    )
    return df

def read_excel_range_with_numeric_headers(file_path, range_str, sheet_name=0, **kwargs):
    """
    Reads a rectangular Excel range (e.g., 'T20:AJ81') and assigns numeric headers from the first row of the range.

    Parameters:
        file_path (str): Path to the Excel file.
        range_str (str): Excel-style range string (e.g., 'T20:AJ81').
        sheet_name (str or int): Sheet name or index (default is 0).
        **kwargs: Additional arguments passed to `pd.read_excel()`.

    Returns:
        pd.DataFrame: Data from the specified Excel range with cleaned numeric headers.
    """
    # Parse range
    match = re.match(r'^([A-Z]+)(\d+):([A-Z]+)(\d+)$', range_str.upper())
    if not match:
        raise ValueError("Invalid range format. Use Excel-style range like 'T20:AJ81'.")

    col_start, row_start, col_end, row_end = match.groups()
    row_start, row_end = int(row_start), int(row_end)
    
    # Column indices (1-based)
    col_start_idx = column_index_from_string(col_start)+1
    col_end_idx = column_index_from_string(col_end)

    # Load header row manually using openpyxl
    wb = load_workbook(file_path, read_only=True, data_only=True)
    ws = wb[sheet_name if isinstance(sheet_name, str) else wb.sheetnames[sheet_name]]

    # Read header row cells
    header_cells = [
        ws.cell(row=row_start, column=col_idx).value
        for col_idx in range(col_start_idx, col_end_idx + 1)
    ]
    
    # Optional: Convert to float as pandas do weird things with numneric headers
    headers = []
    for cell in header_cells:
        if isinstance(cell, (int, float)):
            headers.append(np.round(float(cell),1))
        else:
            raise ValueError(f"Non-numeric header value encountered: {cell!r}")

    # Read the data (excluding header row)
    df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        skiprows=row_start,  # skip header row
        nrows=row_end - row_start,  # only data rows
        usecols=f"{col_start}:{col_end}",
        header=None,
        **kwargs
    )
    
    df.columns = headers
    return df

def MakePlots(infname,Tx,Frequency,ZDim,Locations,CB,df,deviceName,RealWeight=None,
              bUseRayleighPhase=True,dfPhase=None):
  
    try:
        x=np.load(infname+'.npz',allow_pickle=True)['res']
        if type(x) is np.ndarray:
            print('x is np.array')
            if len(x.shape)==0:
                x=x.flatten()[0]['x']
    except:
        print('exception loading x, trying to load res')
        x=np.load(infname+'.npz',allow_pickle=True)['res'].flatten()[0].x


    A=np.load(infname+'.npz',allow_pickle=True)['A']

    outdir=infname+'-plots'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    rootpath=outdir+os.sep+os.path.basename(infname)
        
    SSEnow=[]
    SSEw=[]

    report={}
    maxp=0.0
    for l in Locations:
        report[l]=np.array(df[l])
        maxp=np.max([maxp,report[l].max()])
    # for l in Locations:
    #     report[l]/=maxp

    if len(Locations)>12:
        f,ax=plt.subplots(4,4,figsize=(22/3*2,14/3*2))
    else:
        f,ax=plt.subplots(3,4,figsize=(22/3*2,14/3*2))
    ax=ax.flatten()
    B=np.ones((A.shape[1],1),np.float32)
    for n,l in enumerate(Locations):
        base=np.abs(A[n*ZDim.shape[0]:(n+1)*ZDim.shape[0],:].dot(B)).flatten()
        corrected=np.abs(A[n*ZDim.shape[0]:(n+1)*ZDim.shape[0],:].dot(x)).flatten()
        experimental=np.array(report[l]).flatten()

        # report=df[l]/df[l].max()
        if n==0:
            ax[n].plot(ZDim*1e3,base,':',label='Uncorrected')
            ax[n].plot(ZDim*1e3,experimental,label='Experimental')
            ax[n].plot(ZDim*1e3,corrected,label='Corrected')
        else:
            ax[n].plot(ZDim*1e3,base,':')
            ax[n].plot(ZDim*1e3,experimental)
            ax[n].plot(ZDim*1e3,corrected)
        try:
            SSEnow+=((base-experimental)**2).to_list()
            SSEw+=((corrected-experimental)**2).to_list()
        except:
            SSEnow+=((base-experimental)**2).tolist()
            SSEw+=((corrected-experimental)**2).tolist()
        
        ax[n].set_title('TPO = '+str(l) + ' mm',fontsize=10)
    # 
    plt.suptitle('Comparison of corrected and uncorrected source')
    f.legend(loc='upper center',bbox_to_anchor=[1.055,0.94])
    f.supxlabel('Z (mm)')
    f.supylabel('Pressure (a.u.)')
    plt.tight_layout()
    SSEnow=np.array(SSEnow)
    SSEw=np.array(SSEw)
    SSEnow=SSEnow.sum()/len(SSEnow)
    SSEw=SSEw.sum()/len(SSEw)
    
    print('SSE non corrected',SSEnow)
    print('SSE corrected',SSEw)
    print('SSE reduction',1.0 -SSEw/SSEnow)
    plt.savefig(rootpath+'.pdf',bbox_inches='tight')
 
    def PlotWeight(ax,xd,norm,cmap):
        
        nelem=0
        for VertDisplay, FaceDisplay in zip(Tx['RingVertDisplay'],
                                            Tx['RingFaceDisplay']):

            for e in VertDisplay[FaceDisplay][:,:,:2]:
                color = cmap(norm(xd[nelem]))
                pp3 = plt.Polygon(e*1e3,linewidth=0.1,color=color)
                ax.add_patch(pp3)
                nelem+=1
        ax.set_aspect('equal')
        for din,dout in zip(CB._InDiameters,CB._OutDiameters):
            ax.add_patch(plt.Circle((0,0),din/2*1e3,ls='--',linewidth=1,color='g',fill=False))
            ax.add_patch(plt.Circle((0,0),dout/2*1e3,ls='--',linewidth=1,color='g',fill=False))
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_xlim(-Tx['Aperture']/2*1e3-2,Tx['Aperture']/2*1e3+2)
        ax.set_ylim(-Tx['Aperture']/2*1e3-2,Tx['Aperture']/2*1e3+2)
        # plt.colorbar(im)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Needed only for colorbar
        plt.colorbar(sm, ax=ax)

    if RealWeight is None:
    
        cmap = plt.cm.coolwarm
        # norm = matplotlib.colors.Normalize(vmin=x.min(), vmax=x.max())  # Normalize scalar between 0 and 1
        if np.iscomplex(x[0]):
            nplots=2  
        else:
            nplots=1
            
        if nplots==1:
            f,axs=plt.subplots(1,nplots,figsize=(5,4))
        else:
            f,axs=plt.subplots(1,nplots,figsize=(12,4))
        if nplots>1:
            axs=axs.flatten()
        else:
            axs=[axs]

        for nf in range(nplots):
            ax=axs[nf]
            if nf==0:
                xd=np.abs(x)
                norm =  matplotlib.colors.TwoSlopeNorm(vmin=xd.min(), vmax=xd.max(),vcenter=1.0)
            else:
                xd=np.angle(x)
                norm =  matplotlib.colors.TwoSlopeNorm(vmin=-np.pi, vmax=np.pi,vcenter=0.0)

            PlotWeight(ax,xd,norm,cmap)
            if nf==0:
                ax.set_title('Optimized amplitude',fontsize=12)
            else:
                ax.set_title('Optimized phase',fontsize=12)

    else:

        f,axs=plt.subplots(1,2,figsize=(12,4))
        axs=axs.flatten()
        cmap = plt.cm.coolwarm
        norm =  matplotlib.colors.Normalize(vmin=RealWeight.min(), vmax=RealWeight.max())
        PlotWeight(axs[0],RealWeight,norm,cmap)
        PlotWeight(axs[1],x,norm,cmap)


    plt.savefig(rootpath+'-weight.pdf',bbox_inches='tight')
    # plt.savefig(rootpath+'-weight.png',dpi=300,bbox_inches='tight')

    zf=np.array(df.index).astype(np.float32)*1e-3
    Step=1500/Frequency/6
    zf=np.arange(zf.min(),zf.max()+Step,Step)
    xf=np.arange(Tx['center'][:,0].min(),
                 Tx['center'][:,0].max()+Step,Step)
    
    # xf=np.arange(-20e-3,
    #              20e-3+Step,Step)
    yf=np.zeros(1)
    
    yp,xp,zp=np.meshgrid(yf,xf,zf)
    rf=np.hstack((np.reshape(xp,(xp.size,1)),np.reshape(yp,(xp.size,1)), np.reshape(zp,(xp.size,1)))).astype(np.float32)
    
    
    Material=BabelIntegrationANNULAR_ARRAY.Material
    cwvnb_extlay=np.array(2*np.pi*Frequency/Material['Water'][1]+1j*0).astype(np.complex64)

    def DoXZ(inx):
        
        AcPlanes=[]
        
        for l in Locations:
            ds=np.ones((1))*1e-3 #arbitary number
            center=np.zeros((1,3),np.float32)
            center[0,2]=l*1e-3
            u2back=np.zeros(Tx['NumberElems'],np.complex64)
            
            nBase=0
            # print('Locations Tx and center',Tx['center'].min(axis=0),center)
            if bUseRayleighPhase:
                for n in range(Tx['NumberElems']):
                    u0=np.ones(Tx['elemdims'][n][0],np.complex64)
                    SelCenters=Tx['center'][nBase:nBase+Tx['elemdims'][n][0],:].astype(np.float32)
                    SelDs=Tx['ds'][nBase:nBase+Tx['elemdims'][n][0],:].astype(np.float32)
                    u2back[n]=ForwardSimple(cwvnb_extlay,SelCenters,SelDs,u0,center,deviceMetal=deviceName)[0]
                    nBase+=Tx['elemdims'][n][0]
                AllPhi=np.zeros(Tx['NumberElems'])
                for n in range(Tx['NumberElems']):
                    phi=-np.angle(u2back[n])
                    AllPhi[n]=phi
            else:
                AllPhi=np.zeros(Tx['NumberElems'])
                for n in range(Tx['NumberElems']):
                    AllPhi[n]=np.interp(l,dfPhase.index,dfPhase['EL%i Phase' %(n+1)])
            u0=np.zeros((Tx['center'].shape[0],1),np.complex64)
            nBase=0
            for n in range(Tx['NumberElems']):
                u0[nBase:nBase+Tx['elemdims'][n][0]]=(np.exp(1j*AllPhi[n])).astype(np.complex64)
                nBase+=Tx['elemdims'][n][0]
            u0*=inx.reshape((len(inx),1))
            acplane=np.abs(ForwardSimple(cwvnb_extlay,Tx['center'].astype(np.float32),Tx['ds'].astype(np.float32),u0,rf,deviceMetal=deviceName))
            # acplane/=acplane.max()
            acplane=acplane.reshape((len(xf),len(zf)))
            AcPlanes.append(acplane)
        maxPlanes=np.max(np.hstack(AcPlanes))
        print('maxPlanes',maxPlanes)
        for n in range(len(AcPlanes)):
            AcPlanes[n]/=maxPlanes

        return AcPlanes
    AcPlanes=DoXZ(x)

    def PlotXZ(InPlanes):
        f,axs=plt.subplots(2,8,figsize=(16,6))
        axs=axs.flatten()
        ext=[xf.min()*1e3,xf.max()*1e3,zf.max()*1e3,zf.min()*1e3]

        for l,ax,acplane,nax in zip(Locations,axs,InPlanes,range(len(Locations))): 
            im=ax.imshow(acplane.T,cmap=plt.cm.jet,extent=ext,vmax=1.0,vmin=0)
            ax.set_title('TPO = '+str(l) + ' mm',fontsize=8)

            if nax % 8 != 0:
                ax.tick_params(labelleft=False)

            if nax < 8:
                ax.tick_params(labelbottom=False)

            ax.tick_params(axis='both', labelsize=9)

        f.subplots_adjust(right=0.83)
        cbar_ax = f.add_axes([0.85, 0.53, 0.015, 0.35])
        cb=f.colorbar(im, cax=cbar_ax)
        cbar_ax.tick_params(labelsize=9)

    PlotXZ(AcPlanes)
        
    plt.suptitle(infname+'- XZ planes')
    
   
    plt.savefig(rootpath+'-Acplanes.pdf',bbox_inches='tight')
    # plt.savefig(rootpath+'-Acplanes.png',dpi=300,bbox_inches='tight')

    AcPlanesReal=[]
    if RealWeight is not None:
        AcPlanesReal=DoXZ(RealWeight)
        PlotXZ(AcPlanesReal)
        plt.suptitle('REAL - XZ planes')
        plt.savefig(rootpath+'-REAL-Acplanes.pdf',bbox_inches='tight')
        
        DiffPlanes=[]
        for n in range(len(AcPlanes)):
            DiffPlanes.append(np.abs(AcPlanes[n]-AcPlanesReal[n]))
        PlotXZ(DiffPlanes)
        plt.suptitle('Diff - XZ planes')
        plt.savefig(rootpath+'-diff-Acplanes.pdf',bbox_inches='tight')
        # plt.savefig(rootpath+'-diff-Acplanes.png',dpi=300,bbox_inches='tight')
        AllError=np.abs(np.hstack(AcPlanes)-np.hstack(AcPlanesReal))
        print('Mean error with real planes',np.mean(AllError))
        print('Std error with real planes',np.std(AllError))
        PlanesMeanError=np.mean(AllError)
        PlanesStdError=np.std(AllError)
        PlanesSSE=np.sum(AllError**2)/len(AllError)
    
    

    SaveToH5py({'AcPlanes':AcPlanes},rootpath+'-planes.h5')
    if RealWeight is not None:
        return SSEnow,SSEw,PlanesMeanError, PlanesStdError,PlanesSSE
    else:
        return SSEnow,SSEw
    
def spherical_cap_area(A, R):
    """
    Calculate the surface area of a spherical cap.

    Parameters:
    A (float): Diameter of the circular aperture.
    R (float): Radius of the sphere (focal length).

    Returns:
    float: Surface area of the spherical cap.
    """
    a = A / 2  # aperture radius
    h = R - np.sqrt(R**2 - a**2)
    area = 2 * np.pi * R * h
    return area

def aperture_from_area(S, R):
    """
    Calculate the aperture diameter A of a spherical cap given surface area S and sphere radius R.

    Parameters:
    S (float): Surface area of the spherical cap.
    R (float): Radius of the sphere (focal length).

    Returns:
    float: Aperture diameter A.
    """
    term = R - (S / (2 * np.pi * R))
    a = np.sqrt(R**2 - term**2)
    A = 2 * a
    return A

def equal_area_ring_apertures(S, R, N):
    """
    Divide a spherical cap of area S and radius R into N rings of equal surface area.
    
    Returns aperture diameters (boundaries) for each ring.
    
    Parameters:
    S (float): Total spherical cap surface area.
    R (float): Sphere radius.
    N (int): Number of concentric rings.
    
    Returns:
    List[float]: List of aperture diameters defining ring boundaries (length N).
    """
    delta_h = S / (2 * np.pi * R * N)
    diameters = []
    for i in range(1, N + 1):
        h_i = i * delta_h
        a_i = np.sqrt(R**2 - (R - h_i)**2)
        A_i = 2 * a_i  # aperture diameter
        diameters.append(A_i)
    return np.array(diameters)

def equal_area_ring_apertures_with_inner(S, R, N, A_min):
    """
    Divide a spherical cap (starting from aperture A_min) into N rings of equal surface area.
    
    Parameters:
    S (float): Total surface area of the spherical cap (excluding inner blocked region).
    R (float): Sphere radius.
    N (int): Number of concentric rings.
    A_min (float): Inner blocked aperture diameter.
    
    Returns:
    List[float]: Aperture diameters (outer edges) of the N rings.
    """
    a_min = A_min / 2
    h_min = R - np.sqrt(R**2 - a_min**2)
    
    delta_h = S / (2 * np.pi * R * N)
    
    diameters = []
    for i in range(1, N + 1):
        h_i = h_min + i * delta_h
        a_i = np.sqrt(R**2 - (R - h_i)**2)
        A_i = 2 * a_i
        diameters.append(A_i)
    
    return np.array(diameters)

def RayleighCoeff(Tx,Loc,cwn,u0):
    Distance=np.linalg.norm(Tx['center']-Loc,axis=1)
    coeff=1j*cwn/(2*np.pi)
    coeff=coeff*np.exp(-1j*cwn*Distance)/Distance*Tx['ds'].flatten()*u0.flatten()
    return coeff.astype(np.complex64)

# ----------------------
# Utility Functions
# ----------------------
def compute_data_term(b, A, e):
    Ab = A @ b
    return np.sum((np.abs(Ab) - e)**2)

def compute_data_gradient(b, A, e):
    Ab = A @ b
    abs_Ab = np.abs(Ab)
    safe_abs = np.where(abs_Ab == 0, 1e-12, abs_Ab)
    Ab_conj = np.conj(Ab)

    grad = np.zeros_like(b)
    for k in range(len(b)):
        partial = np.real(A[:, k] * Ab_conj / safe_abs)
        grad[k] = 2 * np.sum((abs_Ab - e) * partial)
    return grad

def phase_objective(b, A, c):
    phi = np.exp(1j * b)        # element-wise e^{i b_j}
    z = A @ phi                 # resulting complex vector
    return np.sum((np.abs(z) - c) ** 2)

def phase_jacobian(b, A, c):
    phi = np.exp(1j * b)        # e^{i b}
    z = A @ phi                 # shape (M,)
    abs_z = np.abs(z)
    safe_abs = np.where(abs_z == 0, 1e-12, abs_z)  # avoid div-by-zero
    z_conj = np.conj(z)

    grad = np.zeros_like(b)
    for k in range(len(b)):
        d_z_dbk = 1j * A[:, k] * phi[k]  # derivative of z wrt b_k
        term = np.real(z_conj / safe_abs * d_z_dbk)
        grad[k] = 2 * np.sum((abs_z - c) * term)
    return grad

# ----------------------
# Regularization Classes
# ----------------------
class Regularizer:
    def __call__(self, b):
        return 0.0

    def gradient(self, b):
        return np.zeros_like(b)

class L2Regularizer(Regularizer):
    def __init__(self, lam):
        self.lam = lam

    def __call__(self, b):
        return self.lam * np.sum(b**2)

    def gradient(self, b):
        return 2 * self.lam * b

class L1Regularizer(Regularizer):
    def __init__(self, lam, delta=1e-6):
        self.lam = lam
        self.delta = delta

    def __call__(self, b):
        return self.lam * np.sum(np.sqrt(b**2 + self.delta))

    def gradient(self, b):
        return self.lam * b / np.sqrt(b**2 + self.delta)

class EntropyRegularizer(Regularizer):
    def __init__(self, lam):
        self.lam = lam

    def __call__(self, b):
        return self.lam * np.sum(b * np.log(np.clip(b, 1e-12, None)))

    def gradient(self, b):
        return self.lam * (np.log(np.clip(b, 1e-12, None)) + 1)

class GroupL2Regularizer(Regularizer):
    def __init__(self, lam, groups):
        self.lam = lam
        self.groups = groups

    def __call__(self, b):
        return self.lam * sum(np.linalg.norm(b[g]) for g in self.groups)

    def gradient(self, b):
        grad = np.zeros_like(b)
        for g in self.groups:
            norm_g = np.linalg.norm(b[g])
            if norm_g > 1e-12:
                grad[g] += self.lam * b[g] / norm_g
        return grad

class GroupHomogeneityRegularizer(Regularizer):
    def __init__(self, lam, group_indices):
        self.lam = lam
        self.group_indices = group_indices  # list of lists of indices

    def __call__(self, b):
        reg = 0.0
        for group in self.group_indices:
            group_b = b[group]
            mean_b = np.mean(group_b)
            reg += np.sum((group_b - mean_b)**2)
        return self.lam * reg

    def gradient(self, b):
        grad = np.zeros_like(b)
        for group in self.group_indices:
            group_b = b[group]
            mean_b = np.mean(group_b)
            for i in group:
                grad[i] += 2 * self.lam * (b[i] - mean_b)
        return grad
    
class TotalVariationRegularizer(Regularizer):
    def __init__(self, lam, delta=1e-6):
        self.lam = lam
        self.delta = delta

    def __call__(self, b):
        diff = np.diff(b)
        return self.lam * np.sum(np.sqrt(diff**2 + self.delta))

    def gradient(self, b):
        grad = np.zeros_like(b)
        diff = np.diff(b)
        denom = np.sqrt(diff**2 + self.delta)
        grad[:-1] -= self.lam * diff / denom
        grad[1:] += self.lam * diff / denom
        return grad

class ElasticNetRegularizer(Regularizer):
    def __init__(self, lam1, lam2, delta=1e-6):
        self.lam1 = lam1
        self.lam2 = lam2
        self.delta = delta

    def __call__(self, b):
        l1 = np.sum(np.sqrt(b**2 + self.delta))
        l2 = np.sum(b**2)
        return self.lam1 * l1 + self.lam2 * l2

    def gradient(self, b):
        l1_grad = b / np.sqrt(b**2 + self.delta)
        l2_grad = 2 * b
        return self.lam1 * l1_grad + self.lam2 * l2_grad

class TikhonovRegularizer(Regularizer):
    def __init__(self, lam, L):
        self.lam = lam
        self.L = L  # L is a matrix

    def __call__(self, b):
        return self.lam * np.sum((self.L @ b)**2)

    def gradient(self, b):
        return 2 * self.lam * self.L.T @ (self.L @ b)
# ----------------------
# Objective with Regularization
# ----------------------
def objective_reg(b, A, e, regularizer):
    return compute_data_term(b, A, e) + regularizer(b)

def jacobian_reg(b, A, e, regularizer):
    return compute_data_gradient(b, A, e) + regularizer.gradient(b)

# ----------------------
# Optimization Wrapper
# ----------------------
def optimize_b(A, e, b0, regularizer):
    options={'maxfun':2000,'disp':0}
    return minimize(
        objective_reg, b0, args=(A, e, regularizer),
        jac=jacobian_reg,
        bounds=[(1e-6, 100)] * len(b0),
        method='L-BFGS-B',
        options=options
    )

def objective_reg_phaae(b, A, e, regularizer):
    return phase_objective(b, A, e) + regularizer(b)

def jacobian_reg_phase(b, A, e, regularizer):
    return phase_jacobian(b, A, e) + regularizer.gradient(b)

# ----------------------
# Optimization Wrapper
# ----------------------
def optimize_b_phase(A, e, b0, regularizer):
    options={'maxfun':2000,'disp':2}
    return minimize(
        objective_reg_phaae, b0, args=(A, e, regularizer),
        jac=jacobian_reg_phase,
        bounds=[(-np.pi, np.pi)] * len(b0),
        method='L-BFGS-B',
        options=options
    )

def complex_objective(b_complex, A, e):

    b =b_complex[:A.shape[1]]*np.exp(1j*b_complex[A.shape[1]:])
    Ab = A @ b
    return np.sum((np.abs(Ab) - e) ** 2)

def complex_gradient(b_complex, A, e):
    b =b_complex[:A.shape[1]]*np.exp(1j*b_complex[A.shape[1]:])
    Ab = A @ b
    abs_Ab = np.abs(Ab)
    safe_abs = np.where(abs_Ab == 0, 1e-12, abs_Ab)
    z_ratio = ((abs_Ab - e) / safe_abs) * Ab  # shape (M,)
    grad_complex = A.conj().T @ z_ratio       # shape (N,)

    # Return real-valued gradient for real optimizer
    return 2 * np.hstack([grad_complex.real, grad_complex.imag])

def objective_reg_complex(b, A, e, regularizer):
    return complex_objective(b, A, e) + regularizer(b)

def jacobian_reg_complex(b, A, e, regularizer):
    return complex_gradient(b, A, e) + regularizer.gradient(b)

# ----------------------
# Optimization Wrapper
# ----------------------
def optimize_b_complex(A, e, b0, regularizer):
    options={'maxfun':2000,'disp':0}
    bounds=[(1e-6, 100.0)] * (len(b0)//2)
    bounds+=[(-np.pi, np.pi)] * (len(b0)//2)
    return minimize(
        objective_reg_complex, b0, args=(A, e, regularizer),
        jac=jacobian_reg_complex,
        bounds=bounds,
        method='L-BFGS-B',
        options=options)


def RUN_FITTING(GeometricFocus,
              FocalDepth,
              Frequency,
              Aperture,
              InD,
              OutD,
              df,
              rootname,
              complex_fit=False,
              regularizer='L2',
              lam=1e-5,
              config=1,
              bUseRayleighPhase=True,
              dfPhase=None,
              InnerD=0.0,
              deviceName='M3',
              COMPUTING_BACKEND=3):
    #From Report

    if COMPUTING_BACKEND==1:
        InitCuda(deviceName)
    elif COMPUTING_BACKEND==2:
        InitOpenCL(deviceName)
    elif COMPUTING_BACKEND==3:
        InitMetal(deviceName)

    DistanceZ0Outplane= GeometricFocus-FocalDepth
    print('DistanceZ0Outplane',DistanceZ0Outplane)

    FocalLength=GeometricFocus

    AreaCap=spherical_cap_area(Aperture, FocalLength)
    if InnerD>0:
        AreaCap-=spherical_cap_area(InnerD, FocalLength)
    RadiusIn=aperture_from_area(AreaCap/2, FocalLength)
    print('Aperture',Aperture)
    print('AreaCap',AreaCap)
    print('RadiusIn',RadiusIn)


    if config ==1:
        InDiameters=InD
        OutDiameters=OutD
    else:
        if InnerD>0.0:
            print('using ring calculation with dead inner diameter', InnerD)
            ArtRings=equal_area_ring_apertures_with_inner(AreaCap, FocalLength, len(InD), InnerD)
        else:
            ArtRings=equal_area_ring_apertures(AreaCap, FocalLength, len(InD))
        InDiameters=ArtRings.copy()
        InDiameters[1:]=InDiameters[0:-1]
        InDiameters[0]=InnerD
        OutDiameters=ArtRings

    print('rings areas',(spherical_cap_area(OutDiameters,FocalLength)-spherical_cap_area(InDiameters,FocalLength))*1e6)

    CB=BabelIntegrationANNULAR_ARRAY.SimulationConditions(Frequency=Frequency,
                                                      Aperture=Aperture, # m, aperture of the Tx, used to calculated cross section area entering the domain
                                                      FocalLength=FocalLength,
                                                      InDiameters=InDiameters, #inner diameter of rings
                                                      OutDiameters=OutDiameters)
    Tx=CB.GenTx(PPWSurface=8)
    for k in ['center','RingVertDisplay','elemcenter']:
        if k == 'RingVertDisplay':
            for n in range(len(Tx[k])):
                Tx[k][n][:,2]-=DistanceZ0Outplane
        else:
            Tx[k][:,2]-=DistanceZ0Outplane

    ZDim=np.array(df.index).astype(np.float32)*1e-3
    Step=1500/Frequency/6
    ZDim=np.arange(ZDim.min(),ZDim.max()+Step,Step)
    XDim=np.zeros(1,dtype=np.float32)
    YDim=np.zeros(1,dtype=np.float32)

    yp,xp,zp=np.meshgrid(YDim,XDim,ZDim)
    rf=np.hstack((np.reshape(xp,(len(ZDim),1)),np.reshape(yp,(len(ZDim),1)), np.reshape(zp,(len(ZDim),1)))).astype(np.float32)


    Material=BabelIntegrationANNULAR_ARRAY.Material
    cwvnb_extlay=np.array(2*np.pi*Frequency/Material['Water'][1]+1j*0).astype(np.complex64)

    Locations=np.array(df.columns)

    PhaseReprogram={}

    MaxIntensity=0.0
    MaxIntensityReport=0.0
    IdealAcAxes={}
    for l in Locations:
        ds=np.ones((1))*1e-3 #arbitary number
        center=np.zeros((1,3),np.float32)
        center[0,2]=l*1e-3
        u2back=np.zeros(Tx['NumberElems'],np.complex64)
        nBase=0
        # print('Locations Tx and center',Tx['center'].min(axis=0),center)
        if bUseRayleighPhase:
            for n in range(Tx['NumberElems']):
                u0=np.ones(Tx['elemdims'][n][0],np.complex64)
                SelCenters=Tx['center'][nBase:nBase+Tx['elemdims'][n][0],:].astype(np.float32)
                SelDs=Tx['ds'][nBase:nBase+Tx['elemdims'][n][0],:].astype(np.float32)
                u2back[n]=ForwardSimple(cwvnb_extlay,SelCenters,SelDs,u0,center,deviceMetal=deviceName)[0]
                nBase+=Tx['elemdims'][n][0]
            AllPhi=np.zeros(Tx['NumberElems'])
            for n in range(Tx['NumberElems']):
                phi=-np.angle(u2back[n])
                AllPhi[n]=phi
        else:
            AllPhi=np.zeros(Tx['NumberElems'])
            for n in range(Tx['NumberElems']):
                AllPhi[n]=np.interp(l,dfPhase.index,dfPhase['EL%i Phase' %(n+1)])
        
        # print('AllPhi',np.rad2deg(AllPhi))
        BasePhasedArrayProgramming=np.exp(1j*AllPhi)
        u0=np.zeros((Tx['center'].shape[0],1),np.complex64)
        nBase=0
        for n in range(Tx['NumberElems']):
            u0[nBase:nBase+Tx['elemdims'][n][0]]=(np.exp(1j*AllPhi[n])).astype(np.complex64)
            nBase+=Tx['elemdims'][n][0]

        AcAxis=np.abs(ForwardSimple(cwvnb_extlay,Tx['center'].astype(np.float32),Tx['ds'].astype(np.float32),u0,rf,deviceMetal=deviceName))
        AcAxis=AcAxis**2  # Square the amplitude to get intensity
        IdealAcAxes[l]=AcAxis/AcAxis.max()
        MaxIntensity=np.max([AcAxis.max(),MaxIntensity])
        distance6dB = ZDim[AcAxis/AcAxis.max()>=0.5]*1e3
        length6dB = distance6dB[-1]-distance6dB[0]
        center6dB=np.mean([distance6dB[-1],distance6dB[0]])
        MaxLoc=ZDim[np.argmax(AcAxis)]*1e3
        
        PhaseReprogram[l]=AllPhi

        MaxIntensityReport=np.max([df[l].max(),MaxIntensityReport])


    for l in Locations:
        relative_=df[l].max()/MaxIntensityReport
        IdealAcAxes[l]=IdealAcAxes[l]*relative_

    plt.figure(figsize=(16/3*2,5/3*2))
    lines=[]
    for l in Locations:
        lines.append(plt.plot(np.array(df.index),np.sqrt(df[l]/MaxIntensityReport),'-',label=str(l))[0])
    plt.legend()
    for l,p in zip(Locations,lines):
        plt.plot(ZDim*1e3,np.sqrt(IdealAcAxes[l]),':',color=p.get_color())



    ZDim=np.array(df.index).astype(np.float32)*1e-3
    A=np.zeros((ZDim.shape[0]*len(Locations),Tx['center'].shape[0]),np.complex64)


    for n,l in enumerate(Locations):
        u0=np.zeros((Tx['center'].shape[0],1),np.complex64)
        nBase=0
        for m in range(Tx['NumberElems']):
            u0[nBase:nBase+Tx['elemdims'][m][0]]=(np.exp(1j*PhaseReprogram[l][m])).astype(np.complex64)
            nBase+=Tx['elemdims'][m][0]

        sA=np.zeros((ZDim.shape[0],Tx['center'].shape[0]),np.complex64)
        Loc=np.zeros((1,3))
        for m,z in enumerate(ZDim):
            Loc[0,2]=z
            sA[m,:]=RayleighCoeff(Tx,Loc,cwvnb_extlay,u0)

        B=np.ones((Tx['center'].shape[0],1),np.float32)
        R=np.abs(sA.dot(B))

        # we normalize the amplitude to the maximum of the Rayleigh coefficient and multiplied by the sq. root of maximum of the df[l] (intensity) to match the amplitude
        sA/=R.max() 
        sA*=np.sqrt(df[l].max()/MaxIntensityReport) 

        A[n*ZDim.shape[0]:(n+1)*ZDim.shape[0],:]=sA


    alllines=[]

    plt.figure(figsize=(12,5))
    dfMeasurements=df.copy()
    for n,l in enumerate(Locations):
        dfMeasurements[l]=np.sqrt(dfMeasurements[l]/MaxIntensityReport)
        alllines.append(plt.plot(ZDim, dfMeasurements[l],label=str(l))[0])
    plt.legend()
    for n,l in enumerate(Locations):
        sA=A[n*ZDim.shape[0]:(n+1)*ZDim.shape[0],:]
        plt.plot(ZDim,np.abs(sA.dot(B.flatten())),':',color=alllines[n].get_color())



    sz=len(df[Locations[0]])
    E=np.zeros(sz*len(Locations))
    for n,l in enumerate(Locations):
        E[n*sz:(n+1)*sz]=np.array(dfMeasurements[l])
    

    if regularizer == 'None':
        reg = Regularizer()
    elif regularizer=='L2':

        reg =L2Regularizer(lam=lam)
    elif regularizer in ['Grouped','GroupedL2']:
        groups=[]
        nBase=0
    
        if complex_fit:
            for kk in range(2):
                for m in range(Tx['NumberElems']):
                    groups.append(np.arange(nBase,nBase+Tx['elemdims'][m][0]))
                    nBase+=Tx['elemdims'][m][0]
        else:
            for m in range(Tx['NumberElems']):
                groups.append(np.arange(nBase,nBase+Tx['elemdims'][m][0]))
                nBase+=Tx['elemdims'][m][0]
        if regularizer=='Grouped':
            reg =GroupHomogeneityRegularizer(lam,groups)
        else:
            reg =GroupL2Regularizer(lam,groups)
    else:
        raise ValueError("Unknown regularization " + regularizer)
    if complex_fit:
        x0=np.zeros(A.shape[1]*2)
        x0[:A.shape[1]]=1.0
        res=optimize_b_complex(A, E, x0, reg)
        res.x=res.x[:A.shape[1]]*np.exp(1j*res.x[A.shape[1]:])
    else:
        x0=np.ones(A.shape[1])
        res=optimize_b(A, E, x0, reg)


    # options={'maxfun':160000,'disp':2}
    # res=minimize(objective2,x0,method='L-BFGS-B',bounds=Bounds(0.0,100.0),args=(A,E),options=options,jac=jacobian2)#jac=grad_func)
    # #res=minimize(funreg,x0,method='L-BFGS-B',bounds=Bounds(0.0,5.0),args=(A,E),options=options,jac=jacreg)#jac=grad_func)

    if config == 1:
        fname=rootname+'-config1'
    else:
        fname=rootname+'-config2'

    if bUseRayleighPhase:
        fname+='_RayleighPhase'
    else:
        fname+='_ReportPhase'
    np.savez(fname,res=res,A=A)

    if complex_fit:
        f,axs=plt.subplots(1,2,figsize=(12,4))

        im=axs[0].scatter(Tx['center'][:,0], Tx['center'][:,1], c=np.angle(res.x),cmap=plt.cm.gray)
        plt.colorbar(im,label='Angle')
        axs[0].set_title('fitted')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        axs[0].set_aspect('equal')

        im=axs[1].scatter(Tx['center'][:,0], Tx['center'][:,1], c=np.abs(res.x),cmap=plt.cm.gray)
        plt.colorbar(im,label='amplitude')
        axs[1].set_title('fitted')
        axs[1].set_xlabel('X')
        axs[1].set_ylabel('Y')
        axs[1].set_aspect('equal')
    else:

        f,axs=plt.subplots(1,1,figsize=(12,4))

        im=axs.scatter(Tx['center'][:,0]*1e3, Tx['center'][:,1]*1e3, c=res.x,cmap=plt.cm.gray)
        plt.colorbar(im,label='Amplitude')
        axs.set_title('fitted')
        axs.set_xlabel('X (mm)')
        axs.set_ylabel('Y (mm)')
        axs.set_aspect('equal')

    plt.figure(figsize=(12,5))
    dfMeasurements=df.copy()
    for n,l in enumerate(Locations):
        dfMeasurements[l]=np.sqrt(dfMeasurements[l]/MaxIntensityReport)
        alllines.append(plt.plot(ZDim, dfMeasurements[l],label=str(l))[0])
    plt.legend()
    for n,l in enumerate(Locations):
        sA=A[n*ZDim.shape[0]:(n+1)*ZDim.shape[0],:]
        plt.plot(ZDim,np.abs(sA.dot(res.x)),':',color=alllines[n].get_color())


    dfMeasurements=df.copy()
    for n,l in enumerate(Locations):
        dfMeasurements[l]=np.sqrt(dfMeasurements[l]/MaxIntensityReport)

    SSEnow,SSEw_BFGS=MakePlots(fname,Tx,Frequency,ZDim,Locations,CB,dfMeasurements,deviceName,
                                   bUseRayleighPhase=bUseRayleighPhase,dfPhase=dfPhase)
    return SSEw_BFGS
