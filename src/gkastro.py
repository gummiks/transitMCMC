#!/usr/bin/python
# -*- coding: utf-8 -*-
# File: gkastro.py
# Created: 2016-03-31 by gks 
"""
Description: Helpful for astronomy
"""

from __future__ import print_function

import everest
import pandas as pd
import numpy as np
from astropy import constants as aconst
import datetime
import matplotlib.pyplot as plt
import math
import os

from astropy.visualization import LogStretch, SqrtStretch, AsinhStretch, HistEqStretch,ZScaleInterval
from astropy.visualization.mpl_normalize import ImageNormalize
from astroquery.simbad import Simbad

norm_mean_sub = lambda x: x - np.nanmean(x)
norm_mean     = lambda x: x/np.nanmean(x)
norm_median   = lambda x: x/np.median(x)
compactString = lambda string: string.replace(' ', '').replace('-', '').lower()
cosd = lambda x : np.cos(np.deg2rad(x))
sind = lambda x : np.sin(np.deg2rad(x))

def make_dir(dirname,verbose=True):
    try:
        os.makedirs(dirname)
        if verbose==True: print("Created folder:",dirname)
    except OSError:
        if verbose==True: print(dirname,"already exists. Skipping")
    



def round_sig(x, sig=2,return_round_to=False):
    """
    Roundint to *sig* significant digits
    
    INPUT:
    x - number to round
    sig - significant digits
    """
    if (np.isnan(x)) & (return_round_to==False):
        return 0.
    if (np.isnan(x)) & (return_round_to==True):
        return 0., 0
    if (x==0.) & (return_round_to==False):
        return 0.
    if (x==0.) & (return_round_to==True):
        return 0., 0
    round_to = sig-int(math.floor(np.log10(abs(x))))-1
    num = round(x, round_to)
    if np.abs(num) > 1e-4:
        num = str(num).ljust(round_to+2,"0") # pad with 0 if needed
    else:
        num = "{0:.{width}f}".format(num,width=round_to-1)
    if return_round_to==False:
        return num
        #return round(x, round_to)
    else:
        return num, round_to
        #return round(x,round_to), round_to

