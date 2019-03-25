#Sample script to run ptychography simulations.
import numpy as np
import matplotlib.pyplot as p
from microscopes import Microscope
from simTEMimages import simTFS
#from skimage import io as skimage_io
#import time as time
#import h5py as h5py

#dimensions
dim = 256
del_px = 1.0

#Create microscope object
altem = Microscope(Cs = 200.0e3, theta_c = 0.05e-3, def_spr = 80.0)

#Parameters for TFS
jobID = 'TEMsim256'
#path = '/Users/cphatak/Box Sync/phase_recon/FSR_py/data/'
dim = 256
del_px = 1
num = 15
defstep = 500.0
stype = 'Quadratic'
#jobId = 'TEMsim256'
path = '/Users/cphatak/ANL_work/ptychography/sims_py/data_highfreq/'

res = simTFS(altem, jobID=jobID, path=path, dim=dim, del_px=del_px, num=num,
        defstep=defstep, stype=stype, display=True)

