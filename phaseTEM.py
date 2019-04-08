#!/usr/bin/python
#
# This module contains routines for retrieval of phase using various methods.
# Currently implemented methods are FSR (Focal series reconstruction), 
# Future work will also include TIE.
# The functions are :
# 1) fsr.py - Focal series reconstruction using iterative approach.
#
# Written CD Phatak, ANL, 24.Mar.2015.

import numpy as np
import matplotlib.pyplot as p
from microscopes import Microscope
from skimage import io as skimage_io
from scipy.interpolate import RectBivariateSpline
import time as time
import h5py as h5py


def fsr(niter=100,display=True):
    #This routine is for performing focal series reconstruction. The filenames, series type, and 
    #defstep need to be manually set in the routine before running it. The algorithm is FRWR algorithm
    #as detailed in [1]. The routine saves the final object as a complex wavefunction in csv format from
    #which the amp and phase can be recovered. Additionally the convergence criterion value is saved for
    #each iteration.
    #
    # [1] C. Koch, Ultramicroscopy, 108, 141-150 (2008).
    
    #manual settings for filenames, series type and defstep
    num = 15
    stype = 'Quadratic'
    defstep = 1000.0
    fname_pref = 'tfsim_'+stype+'_'+str(defstep)+'_'
    tmp = skimage_io.imread(fname_pref+'0000.tiff',plugin='tifffile')
    [xsz,ysz] = tmp.shape

    #define the required defocus values
    num_def = np.arange(num)-float(num-1)/2
    if stype == 'Linear':
        defvals = num_def * defstep #for linear defocus series
    else:
        defvals = num_def**2 * np.sign(num_def) * defstep #for quadratic defocus series

    #load images into image stack
    imstack = np.zeros([xsz,ysz,num])
    for d in range(num):
        tmp = skimage_io.imread(fname_pref+'{:04d}.tiff'.format(d),plugin='tifffile')
        imstack[:,:,d] = tmp


    #Initialize various arrays required during the reconstruction - 
    
    #dimensions
    dim = 256
    d2 = dim/2
    del_px = 1.0 # scaling - 1nm = 1px

    #co-ordinates required
    line = np.arange(dim)-float(d2)
    [X,Y] = np.meshgrid(line,line)
    th = np.arctan2(Y,X)
    qq = np.sqrt(X**2 + Y**2) / float(dim)

    #embed data into 512 by 512 size
    #bimstack = np.zeros([dim,dim,num])
    #bimstack[d2-xsz/2:d2+xsz/2,d2-ysz/2:d2+ysz/2,:] = imstack
    #imstack = bimstack

    #Get the microscope object - 
    altem = Microscope(Cs = 200.0e3, theta_c = 0.05e-3, def_spr = 80.0)
    
    #Initial guess for Amplitude and Phase
    GuessAmp = np.sqrt(imstack[:,:,(num-1)/2])
    GuessPhi = np.zeros(GuessAmp.shape)
    
    #Guess Object wavefunction
    #ObjWave = GuessAmp * (np.cos(GuessPhi) + 1j*np.sin(GuessPhi))
    ObjWave = np.ones(GuessAmp.shape,dtype=complex)

    #array to hold SSE values
    sse_vals = np.zeros(niter)

    #Aperture function
    q_aper = 0.4
    q_pow = 6.0

    #Display
    if display:
        #show the reconstruction steps...
        p.ion()
        fig, (im1,im2,im3) = p.subplots(nrows=1, ncols=3, figsize=(9,3))
        time.sleep(0.05)

    #start loop for each iteration - 
    for ii in range(niter):
        
        #Arrays for storing the weighting factor (Damping
        #Envelope) and updated wavefunction for later use.
        avg_fObjWave = np.zeros(ObjWave.shape, dtype=ObjWave.dtype)
        avg_DampEnv = np.zeros(qq.shape, dtype=qq.dtype)
        aper_damp = 0.0
        sse = 0.0
        #print 'Iteration: ',ii

        #start loop over each through-focus image
        for itfs in range(num):

            #set the defocus
            altem.defocus = defvals[itfs]

            #Step 1: Propagate the wavefunction - Eqn.10 & 12 in [1]
            ImgWave = altem.PropagateWave(ObjWave, qq, del_px)
            Img = np.abs(ImgWave)**2

            #Compute the difference -Eqn.13 in [1]
            diff_Img = np.sqrt(imstack[:,:,itfs]) - np.sqrt(Img)

            #Step 2: Update the Wavefunction - Eqn.14 in [1]
            #ImgWave_up = (np.abs(ImgWave) + diff_Img) * ImgWave / np.abs(ImgWave)
            ImgWave_up = np.sqrt(imstack[:,:,itfs]) * ImgWave / np.abs(ImgWave)

            #Step 3: Backpropagate the updated wavefunction to estimate the
            #next iteration of ObjWave
            avg_fObjWave += altem.BackPropagateWave(ImgWave_up, qq, del_px)
            avg_DampEnv += altem.getDampEnv(qq, del_px)
            aper_damp += altem.getDampEnv(q_aper,del_px)

            #Compute SSE
            sse += np.sum((np.sqrt(imstack[:,:,itfs]) - np.abs(ImgWave))**2) / np.sum(imstack[:,:,itfs])


        #Now compute the weighted average of the ObjWave
        aper_func = np.exp((-(np.power(-np.log(aper_damp/num),1/q_pow)/q_aper) * qq)**q_pow)
        f_ObjWave = avg_fObjWave/avg_DampEnv#*aper_func
        ObjWave = np.fft.ifftn(np.fft.ifftshift(f_ObjWave))
        sse_vals[ii] = sse/float(num)
        print 'Iteration:', ii, ' SSE: ',sse_vals[ii]
        #print
        if display:
            im2.imshow(np.abs(ObjWave)**2,cmap=p.cm.gray)
            im2.axis('off')
            im2.set_title('|Obj|^2',fontsize=20)
            #p.subplot(132)
            im1.imshow(np.angle(ObjWave),cmap=p.cm.gray)
            im1.axis('off')
            im1.set_title('Obj_Phi',fontsize=20)
            im3.imshow(imstack[:,:,(num-1)/2],cmap=p.cm.gray)
            im3.axis('off')
            im3.set_title('Image',fontsize=20)
            fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                    bottom=0.02,left=0.02,right=0.98)
            p.draw()


    #We are done with all iterations, save the ObjWave file as complex CSV file
    np.savetxt('ObjectWaveFunc.csv',ObjWave,delimiter=',')

    return 1



        
    






        







