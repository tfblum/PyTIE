#!/usr/bin/python
#
# This module is for simulating TEM images for various requirements such as
# through-focus series, ptychographic diff. patterns etc.
# The module consists of 2 functions currently -
# (1) Obj_Setup - This function defines the object wave function i.e. amplitude, phase for the object
#                 This will need to be modified manually to vary the configurations. Currently setup
#                 for a disc of magnetic material forming a vortex state.
# (2) simTFS -    This function will use the Object defined using Obj_Setup and simulate the through-focus
#                 series of images for either linear or quadratic defocus series.
#
# Written, CD Phatak, ANL, 05.Mar.2015.

#import necessary modules
import numpy as np
import matplotlib.pyplot as p
from scipy.interpolate import RectBivariateSpline as spline_2d
from microscopes import Microscope
from skimage import io as skimage_io
import time as time
import h5py as h5py


def Obj_Setup(dim = 256,
        del_px = 1.0,
        ran_phase = False):
    # This function is currently setup for generating the amplitude and phase shift
    # for a magnetic vortex disc on a supporting membrane. Options for random phase
    # of the membrane (carbon film) can be turned ON/OFF. It will return the amplitude
    # and phase required for simulating TEM images.
    #
    # [Amp, Mphi, Ephi] = Obj_Setup(dim=256,del_px=1.0)
    
    #Dimensions and Co-ordinates.
    d2 = dim/2
    line = np.arange(dim)-float(d2)
    [X,Y] = np.meshgrid(line,line)
    th = np.arctan2(Y,X)

    #Disc parameters
    disc_rad = 64.0 #nm
    disc_rad /= del_px #px
    disc_thk = 10.0 #nm
    disc_thk /= del_px #px
    disc_V0 = 20.0 #V
    disc_xip0 = 200.0 #nm
    disc_xip0 /= del_px
    r_vec = np.sqrt(X**2 + Y**2)
    disc = np.zeros(r_vec.shape)
    disc[r_vec <= disc_rad] = 1.0

    #Support membrane
    mem_thk = 50.0 #nm
    mem_thk /= del_px
    mem_V0 = 10.0 #V
    mem_xip0 = 800.0 #nm
    mem_xip0 /= del_px
    
    #Magnetization parameters
    b0 = 1.6e4 #Gauss
    phi0 = 20.7e6 #Gauss.nm^2
    cb = b0/phi0*del_px**2 #1/px^2

    #Magnetic Phase shift - Vortex State
    mphi = np.pi * cb * disc_thk * (disc_rad - r_vec) * disc

    #Lattice potential as a phase grating - mostly will run into Nyquist freq. problems...
    mult = 1.5
    var_pot = np.sin(X*mult)

    #Electrostatic Phase shift
    ephi = disc_thk * disc_V0 * disc * del_px * var_pot
    ephi2 = mem_thk * mem_V0 * del_px * np.ones(disc.shape)

    if ran_phase:
        ephi2 = mem_thk * mem_V0 * del_px * np.random.uniform(low = -np.pi, high = np.pi, size=disc.shape)

    #Total Ephase
    ephi += ephi2

    #Amplitude
    amp = np.exp((-np.ones(disc.shape) * mem_thk / mem_xip0) - (disc_thk / disc_xip0 * disc))

    return amp, mphi, ephi

def simTFS(microscope,
        jobID = 'simTFS',
        path = '/Users/cphatak/',
        dim = 256,
        del_px = 1.0,
        num = 11,
        defstep = 500.0,
        stype = 'Linear',
        display = False):

    # This function will take first argument as the microscope object and additional 
    # parameters for number of images, defocus step, type of series. The jobID is used
    # as a prefix for all the data that is saved (TFS images in float32 format).

    #Dimensions and coordinates
    d2=dim/2
    line = np.arange(dim)-float(d2)
    [X,Y] = np.meshgrid(line,line)
    th = np.arctan2(Y,X)
    qq = np.sqrt(X**2 + Y**2) / float(dim)

    #Get the Object values
    [Amp, Mphi, Ephi] = Obj_Setup(dim=dim, del_px=del_px)

    #Create Object WaveFunction
    Ephi *= microscope.sigma
    Tphi = Mphi + Ephi
    ObjWave = Amp * (np.cos(Tphi) + 1j * np.sin(Tphi))

    #define the required defocus values
    num_def = np.arange(num)-float(num-1)/2
    if stype == 'Linear':
        defvals = num_def * defstep #for linear defocus series
    else:
        defvals = num_def**2 * np.sign(num_def) * defstep #for quadratic defocus series

    #check display
    if display:
        p.ion()
        fig, (im1) = p.subplots(nrows=1,ncols=1,figsize=(3,3))
        time.sleep(0.05)

    #Compute the TFS images and save them.
    for d in range(num):
        #set defocus value
        microscope.defocus = defvals[d]

        #get the image
        im = microscope.getImage(ObjWave, qq, del_px)

        #save the image
        fname_pref = path+jobID+stype+'_'+str(defstep)+'_'
        skimage_io.imsave(fname_pref+'{:04d}.tiff'.format(d),im.astype('float32'),plugin='tifffile')

        #check display
        if display:
            im1.imshow(im,cmap=p.cm.gray)
            im1.axis('off')
            im1.set_title('Image',fontsize=20)
            p.draw()

    #end of function
    return 1
















