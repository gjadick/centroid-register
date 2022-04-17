#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 15:00:24 2022

@author: glj
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from scipy.ndimage.measurements import center_of_mass

        
def resize_pixels(img, px_sz, px_sz_target):
    '''
    Resize the pixels of img to match a new target size, 
    in real units.

    Parameters
    ----------
    img : 2D numpy array (the input image)
    px_sz : original pixel size of img.
    px_sz_target : target pixel size for image.

    Returns
    -------
    result : 2D numpy array with shape rescaled by the ratio px_sz/px_size_target

    '''
    Ny, Nx = img.shape
    scale_percent = px_sz/px_sz_target   # ratio of pixel sizes
    
    # convert to PIL to resize
    img_PIL = Image.fromarray(img)
    result_PIL = img_PIL.resize((int(Nx*scale_percent), int(Ny*scale_percent)) )
    
    # return as numpy array
    result = np.array(result_PIL)
    return result


def get_mask(img, t):
    '''
    Generates a mask for an image about a given threshold.

    Parameters
    ----------
    img : 2D numpy array (the input image)
    t : int/float, the threshold value.

    Returns
    -------
    result : 2D numpy array, the thresholded image mask.

    '''
    result = np.zeros(img.shape)
    result[img > t] = 1
    return result


def register_imgs(img, px_sz, img_target, px_sz_target, mask_threshold=20, N_pad=50, plot_check=False):
    '''
    Resizes and registers an input image of given pixel size to match a 
    target image with desired pixel size.
    First, the input image is rescaled to match pixel size of the target.
    Then, both images are thresholded, and the two centroids are calculated.
    The final result aligns the two centroids and crops the input image to match
    the dimensions of the target.

    Parameters
    ----------
    img : 2D numpy array (the input image)
    px_sz : original pixel size of img (real units)
    img_target : 2D numpy array (the target image to match)
    px_sz_target : pixel size of the target image (real units)
    mask_threshold: cutoff value for mask. The default is 20.
    N_pad : Number of 0s to pad with in case input img is too small. The default is 50.
    plot_check : Bool, whether to plot a comparison of in/output. The default is False.

    Returns
    -------
    img_output : 2D numpy array, the resized and registered image.

    '''
    
    # resize original img pixels to match target pixel size (in real units)
    img_resize = resize_pixels(img, px_sz, px_sz_target)
    
    # add 0 padding, to avoid errors
    img_resize = np.pad(img_resize, [(N_pad,N_pad), (N_pad,N_pad)])

    # get centroid coordinates 
    # use masks to eliminate error due to different dose distributions 
    yc, xc = center_of_mass(get_mask(img_resize, mask_threshold))
    yc_target, xc_target = center_of_mass(get_mask(img_target, mask_threshold))

    # get differences in x,y dimensions w.r.t the center of mass
    Ny_target, Nx_target = img_target.shape
    x0 = int(xc - xc_target)
    y0 = int(yc - yc_target)
    img_output = img_resize[y0:y0+Ny_target, x0:x0+Nx_target]     

    # plot
    if plot_check:
        fig,ax=plt.subplots(1,3, figsize=[6,2], dpi=150)
        ax[0].imshow(img_resize)
        ax[0].plot(xc, yc, 'r+')
        rect = patches.Rectangle((x0, y0+1), Nx_target, Ny_target, linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)
        ax[0].set_title('Scanned film')
        ax[1].imshow(img_target)
        ax[1].plot(xc_target, yc_target, 'r+')
        ax[1].set_title('Target distribution')
        ax[2].imshow(img_output-img_target)
        ax[2].set_title('Difference')
        for axi in ax.ravel():  # set ticks off
            axi.set_xticks([])
            axi.set_yticks([])
        fig.tight_layout()
        plt.show()
    
    return img_output

