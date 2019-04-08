#!/usr/bin/python
#
# NAME: Plot_ColorMap.py
#
# PURPOSE:
# This routine will be used for plotting the colormap from the input
# data consisting of 2D images of the vector field. The output image
# will be stored as a tiff color image. There are options to save it
# using custom RGB colorwheel, or standard HSV colorwheel.
#
# CALLING SEQUENCE:
# result = Plot_ColorMap(Bx = Bx, By = By, hsvwheel = True,
#                       filename = filename)
#
# PARAMETERS:
#  Bx : 2D Array consisting of the x-component of the vector field
#  By : 2D Array consisting of the y-component of the vector field
#  hsvwheel : If True then colorimage using the standard HSV scheme or using the custom RGB scheme.
#  filename : The output filename to be used for saving the color image. If not provided, default
#             Vector_ColorMap.jpeg will be used.
#
# RETURNS:
#  result : A (M x N x 3) array containing the color-image.
#
# AUTHOR:
# C. Phatak, ANL, 07.Aug.2017.
#----------------------------------------------------------------------------------------------------

#import necessary modules
import numpy as np
from skimage import io as skimage_io
from skimage import color as skimage_color
from matplotlib import colors as mt_cols

def Plot_ColorMap(Bx = np.random.rand(256,256), By = np.random.rand(256,256), \
                  hsvwheel = False, filename = 'Vector_ColorMap.jpeg'):
    # first get the size of the input data
    [dimx,dimy] = Bx.shape
    #inset colorwheel size - 100 px
    csize = 100
    #co-ordinate arrays for colorwheel.
    line = np.arange(csize) - float(csize/2)
    [X,Y] = np.meshgrid(line,line,indexing = 'xy')
    th = np.arctan2(Y,X)
    h_col = (th + np.pi)/2/np.pi
    rr = np.sqrt(X**2 + Y**2)
    msk = np.zeros(rr.shape)
    msk[np.where(rr <= csize/2)] = 1.0
    rr *= msk
    rr /= np.amax(rr)
    val_col = np.ones(rr.shape) * msk
    

    #Compute the maximum in magnitude BB = sqrt(Bx^2 + By^2)
    mmax = np.amax(np.sqrt(Bx**2 + By**2))
    # Normalize with respect to max.
    Bx /= float(mmax)
    By /= float(mmax)
    #Compute the magnitude and scale between 0 and 1
    Bmag = np.sqrt(Bx**2 + By**2)
    
    if hsvwheel:
        # Here we will proceed with using the standard HSV colorwheel routine.
        # Get the Hue (angle) as By/Bx and scale between [0,1]
        hue = (np.arctan2(By,Bx) + np.pi)/2/np.pi
        # Array to hold the colorimage.
        color_im = np.zeros([dimx+csize, dimy, 3])
        #First the Hue.
        color_im[0:dimx,0:dimy,0] = hue
        # Then the Sat.
        color_im[0:dimx,0:dimy,1] = Bmag
        # Then the Val.
        color_im[0:dimx,0:dimy,2] = np.ones([dimx,dimy])
        # Store the colorwheel in the image
        color_im[dimx:,dimy/2-csize/2:dimy/2+csize/2,0] = h_col
        color_im[dimx:,dimy/2-csize/2:dimy/2+csize/2,1] = rr
        color_im[dimx:,dimy/2-csize/2:dimy/2+csize/2,2] = val_col
        # Convert to RGB image.
        rgb_image = mt_cols.hsv_to_rgb(color_im)
    else:
        #Here we proceed with custom RGB colorwheel.
        #Arrays for each RGB channel
        red = np.zeros([dimx,dimy])
        gr = np.zeros([dimx,dimy])
        blue = np.zeros([dimx,dimy])
    
        #Scale the magnitude between 0 and 255
        cmag = Bmag #* 255.0
        #Compute the cosine of the angle
        cang =  Bx / cmag
        #Compute the sine of the angle
        sang = np.sqrt(1.0 - cang**2)
        #first the green component
        qq = np.where((Bx < 0.0) & (By >= 0.0))
        gr[qq] = cmag[qq] * np.abs(cang[qq])
        qq = np.where((Bx >= 0.0) & (By < 0.0))
        gr[qq] = cmag[qq] * np.abs(sang[qq])
        qq = np.where((Bx < 0.0) & (By < 0.0))
        gr[qq] = cmag[qq]
        # then the red
        qq = np.where((Bx >= 0.0) & (By < 0.0))
        red[qq] = cmag[qq]
        qq = np.where((Bx >=0.0) & (By >= 0.0))
        red[qq] = cmag[qq] * np.abs(cang[qq])
        qq = np.where((Bx < 0.0) & (By < 0.0))
        red[qq] = cmag[qq] * np.abs(sang[qq])
        # then the blue
        qq = np.where(By >= 0.0)
        blue[qq] = cmag[qq] * np.abs(sang[qq])
        # Store the color components in the RGB image
        rgb_image = np.zeros([dimx+csize,dimy,3])
        rgb_image[0:dimx,0:dimy,0] = red
        rgb_image[0:dimx,0:dimy,1] = gr
        rgb_image[0:dimx,0:dimy,2] = blue
    
        #Recompute cmag, cang, sang for the colorwheel representation.
        mmax = np.amax([np.abs(X),np.abs(Y)])
        X /= mmax
        Y /= mmax
        cmag = np.sqrt(X**2 + Y**2) #* 255.0
        cang =  X / cmag
        sang = np.sqrt(1.0 - cang**2)
        # Arrays for colorwheel sizes
        red = np.zeros([csize,csize])
        gr = np.zeros([csize,csize])
        blue = np.zeros([csize,csize])
        #first the green component
        qq = np.where((X < 0.0) & (Y >= 0.0))
        gr[qq] = cmag[qq] * np.abs(cang[qq])
        qq = np.where((X >= 0.0) & (Y < 0.0))
        gr[qq] = cmag[qq] * np.abs(sang[qq])
        qq = np.where((X < 0.0) & (Y < 0.0))
        gr[qq] = cmag[qq]
        # then the red
        qq = np.where((X >= 0.0) & (Y < 0.0))
        red[qq] = cmag[qq]
        qq = np.where((X >=0.0) & (Y >= 0.0))
        red[qq] = cmag[qq] * np.abs(cang[qq])
        qq = np.where((X < 0.0) & (Y < 0.0))
        red[qq] = cmag[qq] * np.abs(sang[qq])
        # then the blue
        qq = np.where(Y >= 0.0)
        blue[qq] = cmag[qq] * np.abs(sang[qq])

        #Store in the colorimage
        rgb_image[dimx:,dimy/2-csize/2:dimy/2+csize/2,0] = red * msk
        rgb_image[dimx:,dimy/2-csize/2:dimy/2+csize/2,1] = gr * msk
        rgb_image[dimx:,dimy/2-csize/2:dimy/2+csize/2,2] = blue * msk

    # Now we have the RGB image. Save it and then return it.
    skimage_io.imsave(filename,rgb_image)

    return rgb_image

