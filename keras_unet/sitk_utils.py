
import SimpleITK as sitk 
import numpy as np
import cv2
import h5py

import matplotlib
from token import SLASH

import matplotlib.pyplot as plt


def myshow2(img, title=None, margin=0.05, dpi=80):
    nda = sitk.GetArrayViewFromImage(img)
    spacing = img.GetSpacing()
        
    if nda.ndim == 3:
        # fastest dim, either component or x
        c = nda.shape[-1]
        
        # the the number of components is 3 or 4 consider it an RGB image
        if not c in (3,4):
            nda = nda[nda.shape[0]//2,:,:]
    
    elif nda.ndim == 4:
        c = nda.shape[-1]
        
        if not c in (3,4):
            raise Runtime("Unable to show 3D-vector Image")
            
        # take a z-slice
        nda = nda[nda.shape[0]//2,:,:,:]
            
    ysize = nda.shape[0]
    xsize = nda.shape[1]
      
    # Make a figure big enough to accommodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Make the axis the right size...
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
    
    extent = (0, xsize*spacing[1], ysize*spacing[0], 0)
    
    t = ax.imshow(nda,extent=extent,interpolation=None)
    
    if nda.ndim == 2:
        t.set_cmap("gray")
    
    if(title):
        plt.title(title)




def myshow(img_sitk, txt_title=None, vmin=None, vmax = None):
    npa = sitk.GetArrayViewFromImage( img_sitk )
    if npa.shape[0] == 1:
        npa = np.squeeze(npa,axis = 0)
    plt.figure()
    
    if not vmin:
        vmin = npa.min()
    if not vmax:
        vmax = npa.max()
    
    plt.imshow( npa, cmap=plt.cm.Greys_r , vmin=vmin, vmax = vmax )
#    plt.axis('off')
    plt.title(txt_title, fontsize=10)
    plt.show()
    
	
def myhist(img_sitk):
    npa = sitk.GetArrayViewFromImage( img_sitk )
    if npa.shape[0] == 1:
        npa = np.squeeze(npa,axis = 0)
    plt.figure()
    plt.hist(npa, bins=200)
    plt.show()
    