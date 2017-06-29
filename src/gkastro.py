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
import gklib as gk
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
    

def stretch_data(data,method="HistEqStretch"):
    """
    methods = 
    LogStretch,
    SqrtStretch,
    AsinhStretch,
    HistEqStretch
    """
    if method=="LogStretch":
        norm = ImageNormalize(stretch=LogStretch(data))
    elif method=="SqrtStretch":
        norm = ImageNormalize(stretch=SqrtStretch(data))
    elif method=="AsinhStretch":
        norm = ImageNormalize(stretch=AsinhStretch(data))
    elif method=="HistEqStretch":
        norm = ImageNormalize(stretch=HistEqStretch(data))
    else:
        norm = data
    return norm

def get_johnson_filter(filtername):
    """
    Get Johnson Filter Transmission curves.
    
    INPUT:
        filtername, can be "U","B","V","R","I" for the Johnson filters
        
    RETURNS:
        A pandas dataframe with two columns: "wavelength", and "transmission"
        Wavelength is in Angstroms, transmission is normalized.
        
    EXAMPLE:
        rr= get_johnson_filter("R")
        ii= get_johnson_filter("I")
        plt.plot(ii["wavelength"],ii["transmission"])
        plt.plot(rr["wavelength"],rr["transmission"])
    """
    DIRLOC = "/Users/gks/Dropbox/mypylib/filters/"
    
    filters = ["U","B","V","R","I"]
    filenames = ["bess-u.pass",
                 "bess-b.pass",
                 "bess-v.pass",
                 "bess-r.pass",
                 "bess-i.pass"]    
    df = pd.DataFrame(zip(filters,filenames),columns=["name","filename"])
    
    assert filtername in filters
    filename = df[df["name"]==filtername]["filename"].values[0]
    
    filterdata = pd.read_csv(DIRLOC+str(filename),sep="\t",header=None)
    filterdata.columns = ["wavelength","transmission"]
    return filterdata


from astropy import constants as c

def f_nu_from_magAB(mag):
    """
    Calculate the flux f_nu from AB magnitudes.
    
    INPUT:
        AB magnitude. Be careful in what band. See iObserve too.
        
    RETURNS:
        f_nu in units of ergs/cm^2/s/Hz
    
    EXAMPLE:
        f_nu_from_magAB(0.)/1e-23
            3630.780547701003
    NOTES:
        1 Jansky = 1e-23 ergs/cm^2/s/Hz
        AB mag of 0 should be 3631 Jansky
        f_nu * nu = f_lambda * lambda
        See f_nu2f_lambda to change from f_nu to f_lambda
        https://www.astro.umd.edu/~ssm/ASTR620/mags.html
        http://roban.github.io/CosmoloPy/docAPI/cosmolopy.magnitudes-module.html#f_nu_lambda
    """
    f_nu = 10.**((mag+48.6)/(-2.5))
    return f_nu

def f_nu2f_lambda(f_nu,lam):
    """
    Change flux from ergs/cm^2/s/Hz to erg/cm^2/s/A.
    
    INPUT:
        f_nu in units: ergs/cm^2/s/Hz
        lam in units of A
        
    RETURNS:
        f_lam in erg/cm^2/s/A
        
    EXAMPLE:
        f_nu = f_nu_from_magAB(0.)
        f_nu2f_lambda(f_nu,7884.105)
    
    NOTES:
        http://roban.github.io/CosmoloPy/docAPI/cosmolopy.magnitudes-module.html#f_nu_lambda
    """
    cc = c.c.value * 1e10 # change to angstrom
    f_lam = f_nu*cc/(lam**2.)
    return f_lam


def scintillation_noise(diameter,airmass,exptime,altitude,wavelength=None,withextra=False):
    """
    See eq B4 on page 22 in AIJ paper.
    
    INPUT:
    .diameter. in cm
    .airmass. 
    .exptime. in s
    .altitude. altitude of the observatory in m
    .wavelength. if not None, include (lambda/550nm)**-7/12 dependence from Birney et al. 2006. Units: nm
    
    OUTPUT:
    .sigma_scintillation. - the scintillation noise (not calculated by AIJ)
    
    EXAMPLE:
        diameter = 61. #cm
        airmass = 1. #varies 1 - 2
        exptime = 120. # in s
        altitude = 360. # in m
        scint_noise = scintillation_noise(diameter,airmass,exptime,altitude)
    """
    scint_err = 0.09*(diameter**(-2./3.))*(airmass**1.75)*((2.*exptime)**-0.5)*np.exp(-altitude/8000.)
    if withextra==True:
        # Extra factor of 1.5 according to paper
        scint_err *= 1.5
    if wavelength != None:
        return scint_err * (wavelength/550.)**(-7./12.)
    else:
        return scint_err

def rel_error(source_sky_T1,source_error_T1,source_sky_E,source_error_E,N=1,scint_error=np.array([])):
    """
    Calculate the photon error using equation B3 from AIJ paper. See page 21 in AIJ paper.
    Can add scintillation.
    
    INPUT:
        source_sky_T1 - is the net counts from AIJ ("Source_sky_T1" in AIJ)
        source_error_T1 - is the error in the net counts
        source_sky_E    - net ensamble counts 
        source_error_E  - error in ensamble counts
        N - is the number of comparison stars.
        scint_error - is an array of scintillation errors in relative units. U
    
    OUTPUT:
        rel_error       - in units of (Delta Source) / Source, where Source is in counts
        
    NOTES:
        Scintillation is assumed to be uncorrelated, and is added in quadrature.
        See gkastro.scintillation for scintillation equation. Might need to add 1.5 multiplication factor (Osborn 2015).
        
    EXAMPLE:
        diameter = 61. #cm
        #airmass = 1. #varies 1 - 1.45
        exptime = 120. # in s
        altitude = 360. # in m
        df_all = pd.read_csv("20160326_2degdiff/results/ap_100_150_200/ap_100_150_200_allbin_good_measurements.txt",sep="\t")

        calc_err_no_scint = gkastro.rel_error(df_all["Source-Sky_T1"],df_all["Source_Error_T1"],df_all["Source-Sky_C2"],df_all["Source_Error_C2"])
        scint_err = gkastro.scintillation_noise(diameter,df_all["AIRMASS"],exptime,altitude)
        calc_err_scint = gkastro.rel_error(df_all["Source-Sky_T1"],df_all["Source_Error_T1"],df_all["Source-Sky_C2"],df_all["Source_Error_C2"],scint_error=scint_err,N=1)

        # The next two should be the same
        plt.plot(df_all["HJD_UTC"],calc_err_no_scint,label="No Scint")
        plt.plot(df_all["HJD_UTC"],df_all["rel_flux_err_T1"],label="AIJ",lw=1)
        plt.plot(df_all["HJD_UTC"],scint_err,label="Scint only") # Scint is large
        plt.plot(df_all["HJD_UTC"],calc_err_scint,label="Total") # Total error in quadrature
        plt.legend(loc="upper left")
        plt.xlabel("HJD"); plt.ylabel("STD")
    """
    rel_source = source_sky_T1/source_sky_E
    rel_T1 = source_error_T1/source_sky_T1
    rel_E = source_error_E/source_sky_E
    
    rel_error = rel_source * np.sqrt(rel_T1**2. + rel_E**2.) # Same as "rel_flux_err_T1" in AIJ
    
    if scint_error.any() == False:    
        return rel_error
    else:
        # Old ideas (wrong)
        #rel_error = rel_source * np.sqrt(rel_T1**2. + rel_E**2. + (1.+1./N)*(scint_error**2.0))
        #rel_error_no_scint = rel_source * np.sqrt(rel_T1**2. + rel_E**2.)
        
        # Add scint error in quadrature (correct)
        rel_error = np.sqrt(rel_error**2.0+(scint_error**2.0)*(1.+1./N))

    return rel_error




def weighted_average(group, avg_name, weight_name):
    """
    INPUT:
        A Pandas dataframe with columns *avg_name* and *weight_name* which are used to calculate a weighted average.
        
    RETURNS:
        A single float, the weighted average.
      
    EXAMPLE
        ii = gkastro.get_johnson_filter("I")
        gkastro.weighted_average(ii, "wavelength","transmission")
        >>> 8059.88
        
    NOTES:
        http://stackoverflow.com/questions/10951341/pandas-dataframe-aggregate-function-using-multiple-columns
    """
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return d.mean()


def get_johnson_filter_plots(filters,plot=True):
    """
    A function to get the effective wavelength of a Johnson filter, and the effective bandpass. Can plot if plot==True.
    
    INPUT:
        filters = ["U","B","V","R","I"]
        
    RETURNS:
        effective_data, a Pandas dataframe defined as:
            > pd.DataFrame(zip(filters,wavg,effWave),columns=["filter","wavg","bandpass"])
    EXAMPLE:
        get_johnson_filter_plots(["U","B","V","R","I"],plot=True)
        gkastro.get_johnson_filter_plots(["U","B","V","R","I"],plot=True)
    """
    filter_data = [get_johnson_filter(myfilt) for myfilt in filters]
    wavg = [weighted_average(data,"wavelength","transmission") for data in filter_data]
    effWave = [np.trapz(data["transmission"],data["wavelength"]) for data in filter_data]
    cc = ["purple","blue","green","red","darkred"]
    cc = dict(zip(filters,cc))
    
    if plot == True:
        fig, ax = plt.subplots()
        # plot bandpasses
        for i in range(len(filters)):
            # labelstring
            labelstr = "Johnson "+filters[i]+"\nMean: "+gk.num2str(wavg[i],1)+" A\nBandpass: "+gk.num2str(effWave[i],1)+" A"

            # data
            ax.plot(filter_data[i]["wavelength"],filter_data[i]["transmission"],label=labelstr,lw=1,color=cc[filters[i]])

            # effective wavelength
            ax.axvline(x=wavg[i],ymin=0,ymax=1,lw=1,ls="--",alpha=0.3,color=cc[filters[i]])

            # plot bandpasses
            xmin = wavg[i]-effWave[i]/2.
            xmax = wavg[i]+effWave[i]/2.
            x = np.linspace(xmin,xmax,100)
            ax.plot(x,np.zeros(100)-0.01*i,lw=1.5,ls="-",alpha=0.7,color=cc[filters[i]])

        ax.legend(fontsize="x-large",bbox_to_anchor=(1.45, 1.0),fancybox=True,labelspacing=1.5)
        ax.grid()

        ax.margins(0.1,0.1)
        ax.set_xlabel("Wavelength (A)")
        ax.set_ylabel("Transmission")
        ax.set_title("Johnson Filters",y=1.05)
        ax.margins(0.3,0.05)
    
    effective_data = pd.DataFrame(zip(filters,wavg,effWave),columns=["filter","wavg","bandpass"])
    return effective_data

def airmass(alt):
    """
    Alt should be in deg.
    """
    z = 90. - alt
    z = np.pi*z/180.
    return 1./np.cos(z)

def altitudeFromAirmass(airmass):
    """
    """
    return 90.-(np.arccos(1./airmass))*180./np.pi

# ----
# Observatory class for frequently used observatories
# ----
import astropy.coordinates as coord
import astropy.units as u

class observatory:
    def __init__(self,longitude,latitude,elevation):
        self.longitude = longitude
        self.longitude_deg = coord.Angle(longitude,unit=u.deg).deg
        self.latitude = latitude
        self.latitude_deg = coord.Angle(latitude,unit=u.deg).deg
        self.elevation = elevation #m

apo = observatory("-105:49:13.0","32:46:49.0",2788.)
davey = observatory("-77:51:46.4","40:47:54.3",360.)

def get_sdss_filter(filtername):
    """
    Get Johnson Filter Transmission curves.
    
    INPUT:
        filtername, can be "g'","r'","i'","z'"
        
    RETURNS:
        A pandas dataframe with two columns: "wavelength", and "transmission"
        Wavelength is in Angstroms, transmission is normalized.
        
    EXAMPLE:
        rr= get_sdss_filter("r'")
        ii= get_sdss_filter("i'")
        plt.plot(ii["wavelength"],ii["transmission"])
        plt.plot(rr["wavelength"],rr["transmission"])
    """
    DIRLOC = "/Users/gks/Dropbox/mypylib/filters/arctic_sdss_filters/"
    
    filters = ["g'","r'","i'","z'"]
    filenames = ["sloan_g_filter2.txt",
                 "sloan_r_filter2.txt",
                 "sloan_i_filter2.txt",
                 "sloan_z_filter2.txt"]    
    df = pd.DataFrame(zip(filters,filenames),columns=["name","filename"])
    
    assert filtername in filters
    filename = df[df["name"]==filtername]["filename"].values[0]
    
    filterdata = pd.read_csv(DIRLOC+str(filename),sep="\t",header=None)
    filterdata.columns = ["wavelength","transmission"]
    filterdata["wavelength"]*=10. # to angstroms
    filterdata["transmission"]/=100. # to <1
    return filterdata

# ----
# ----

def read_arctic_qe():
    """
    Helper function to read the ARCTIC qe curve.
    
    Returns a dataframe with 
        df["wavelength"] in angstrom
        df["transmission"] in 0 to 1
        
    EXAMPLE:
        df_qe = read_arctic_qe()
        plt.plot(df_qe.wavelength,df_qe.transmission)
    """
    df_qe = pd.read_csv("/Users/gks/Dropbox/mypylib/filters/arctic_qe/arctic_qe_extrap.txt",sep="\t",header=None)
    df_qe.columns = ["wavelength","transmission"]
    df_qe["wavelength"]*=10.
    df_qe["transmission"]/=100.
    return df_qe

def read_arctic_h_alpha():
    """
    Helper function to read the ARCTIC qe curve.
    
    Returns a dataframe with 
        df["wavelength"] in angstrom
        df["transmission"] in 0 to 1
        
    EXAMPLE:
        df_qe = read_arctic_qe()
        plt.plot(df_qe.wavelength,df_qe.transmission)
    """
    df = pd.read_csv("/Users/gks/Dropbox/mypylib/filters/arctic_h_alpha/arctic_h_alpha_redmost.txt",sep="\t",header=None)
    df.columns = ["wavelength","transmission"]
    df["transmission"]/=100.
    return df


import scipy.constants as const

def flux_mono(wave,temp,radius,dist,units="cgs"):
    """
    Calculate the black body flux for a star at a temp t, radius R and at a distance dist.
    cgs unit verified.

    INPUT:
        wave = wavelength, in A
        temp = temperature, in K
        radius = radius of star, in m
        dist = distance to star, in m
    
    OUTPUT:
        if units = "SI", then outputs flux in W/s/m^2/A
        if units = "cgs", then outputs flux in erg/s/cm^/A
        
    EXAMPLE for Vega:
        from astropy import constants as aconst
        wave = np.linspace(100,1400000,14000)
        temp = 9602. #K
        r = 2.362 * aconst.R_sun
        dist = 7.68 * aconst.pc
        flux = flux_mono(wave,temp,r,dist) # W/s/m^2 /m -> W/s/m^2/A
        fig,ax = plt.subplots()
        ax.plot(wave,flux)
        ax.set_xscale("log")
        ax.set_yscale("log")
    """
    wave = wave*1e-10 # from A to m
    numer = (2.*np.pi*const.h*(const.c)**2./wave**5.)*((radius/dist)**2.0)
    denom = np.exp(const.h*const.c/(const.k*temp*wave))-1.
    flux = numer/denom # in W/s/m^2/m
    flux*= 1e-10 # W/s/m^2/m > W/s/m^2/A
    if units =="SI":
        return flux
    if units =="cgs":
        print("Using cgs units")
        flux *= 1e7 # W/s/m^2/A -> erg/s/m^2 /A
        flux *= 0.01*0.01 # erg/s/m^2/A -> erg /s /cm^2 / A
        return flux
    else:
        print("Unit system not correct! == 'SI' or 'cgs'")
        return 0.

def resample_array(new_sampling_x,old_sampling_x,old_sampling_y):
    """
    Filter x-points and terradata x-points do not have the same sampling.

    Resample, so the filter has the same number of points as the terraspec data.
    """
    new_sampling_y =     np.interp(new_sampling_x,
                                   old_sampling_x,
                                   old_sampling_y)
    return new_sampling_y
    
def resample_filter(new_sampling_x,df):
    """
    RETURNS:
        A dataframe with resampled x and y.
    
    EXAMPLE:
        xx = np.linspace(300,12000,100)
        dff = resample_filter(xx,df_sdss_g)
        plt.plot(dff.wavelength,dff.transmission)
    """
    new_sampling_y =     np.interp(new_sampling_x,
                                   df.ix[:,0],
                                   df.ix[:,1])
    
    df_new = pd.DataFrame(zip(new_sampling_x,new_sampling_y),columns=["wavelength","transmission"])
    return df_new
    
def lambda_eff(wavelength,r_v,f_v):
    """
        Wavelength in angstrom
        r_v - filter response function (bessel file)
        f_v - flux of star in erg/s/cm^2/A
    """
    numer = np.trapz(wavelength*r_v*f_v,wavelength)
    denom = np.trapz(r_v*f_v,wavelength)
    lambda_eff = numer/denom
    return lambda_eff

def magnitude_vega(wavelength,r_v,f_v):
    """
    INPUT:
        wave - Wavelength in angstrom
        r_v - filter response function
        f_v - flux of star
        
    EXAMPLE:
        magnitude_vega(wave,df_johnson_v_red.transmission,flux)
            # 0.3 (blackbody)
        magnitude_vega(wave,df_johnson_v_red.transmission,df_vega_red.transmission)
            # 0.01325
     
    NOTES:
        See Bessel 1998: file:///Users/gks/Downloads/bessel_1998.pdf
    """
    mag = -2.5*np.log10( np.trapz(r_v*f_v,wavelength) / np.trapz(r_v,wavelength) ) - 21.100
    return mag

def poly_detrend(x,y,order,return_coeffs=False,plot=False):
    """
    A polynomial detrend. 
    Returns the prediction, and the residual.
    Returns a list of the polynomial coefficients if return_coeffs == True
    """
    model = np.polyfit(x,y,order)
    predicted = np.polyval(model, x)
    residual = y - predicted
    if plot == True:
        fig, axes = plt.subplots(nrows=2, sharex=True)
        axes[0].plot(x, y, 'ro',markersize=4)
        axes[0].plot(x, predicted, 'k-')
        axes[0].set(title='Original Data and '+str(order)+'-Order Polynomial Trend')
        axes[1].plot(x, residual, 'ro',markersize=4)
        axes[1].set(title='Detrended Residual')
        for label in axes[1].get_xticklabels():
        #negative clockwise; positive counterclockwise
            label.set_rotation(20)
        plt.show()
    if return_coeffs == True:
        return residual, predicted, model
    else:
        return residual, predicted

from PyAstronomy.pyasl import binningx0dt

def bin_data(x,y,numbins):
        """
        Returns a dataframe with binned data.
        
        INPUT:
            numbins is the number of bins 
        
        OUTPUT:
            Dataframe with the binned data. Columns x, y
        """
        timestamp = x
        rel_flux  = y
        
        rel_flux_rms_per_bin = np.zeros(numbins)
        bins = np.arange(numbins)+1
        for i in bins:
            if i == 1:
                rms = np.std(rel_flux)
            else:
                bin_timeAndRelFlux, pointsPerBin = binningx0dt(timestamp, rel_flux,useBinCenter=True, removeNoError=True, reduceBy=i,useMeanX=True)
                rms = np.std(bin_timeAndRelFlux[::,1])
                #print("len calc",len(bin_timeAndRelFlux[::,1]))
            rel_flux_rms_per_bin[i-1] = rms
        df_binned = pd.DataFrame(zip(bin_timeAndRelFlux[::,0],bin_timeAndRelFlux[::,1]),columns=["x","y"])
        return df_binned #, rel_flux_rms_per_bin

def sigma_clip(df,num):
    """
    INPUT
        num - number of sigmas to clip
    
    RETURNS:
        Dataframe without the outliers

    WORKING:
        Needed to use df.ix in pandas to get y column
        
    """
    y = df.ix[:,0]
    df_sigma_clipped = df[np.abs(y-y.mean())<=(num*y.std())]
    return df_sigma_clipped



def get_semrock_filter():
    df = pd.read_csv("/Users/gks/Dropbox/mypylib/filters/semrock/FF01-857_30_spectrum.txt",skiprows=4,header=None,sep="\t",names=["wavelength","transmission"])
    df.wavelength *= 10.
    return df

from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u
import astropy.coordinates as coord
import astropy


def calc_altitude_airmass(ra,dec,time,timeformat="iso",location="Apache Point Observatory",verbose=False,sun=False):
    """
    A function to calculate the altitude and airmass of an object at a given time from a given observatory.
    
    INPUT:
        ra - '11:30:14.51'
        dec - '+07 35 18.2'
        time - "2016-09-20 08:00:00"
        timeformat="jd", "iso"
        location - EarthLocation.get_site_names()
        full - False
        verbose = False
    
    EXAMPLE:
        k218b._RA = '11:30:14.51'
        k218b._DEC = '+07 35 18.2'
        time = "2016-09-20 08:00:00"
        altair = calc_altitude_airmass(k218b._RA,k218b._DEC,"2016-09-20 08:00:00",verbose=True)
        
        ### Additional
        c = SkyCoord(k218b._RA,k218b._DEC,unit=(u.hourangle,u.deg))
        apo = EarthLocation.of_site("Apache Point Observatory")
        obstime = Time("2016-09-20 08:00:00")
        c.transform_to(AltAz(obstime=obstime,location=apo)).secz
    """
    obstime = Time(time,format=timeformat)
    if location=="IAO":
        loc = EarthLocation.from_geodetic(lon=78.9641,lat=32.77944,height=4500.)
    elif location=="Mt Abu":
        loc = EarthLocation.from_geodetic(lon=72.779217,lat=24.654817,height=1680.)
    elif location=="Skinkas":
        loc = EarthLocation.from_geodetic(lon=24.8991667,lat=35.2119444,height=1750.)
    elif location=="Kryoneri":
        loc = EarthLocation.from_geodetic(lon=22.6186111,lat=37.971944,height=930.)
    else:
        loc = EarthLocation.of_site(location)
    altaz_frame = AltAz(obstime=obstime,location=loc)
    c_t = SkyCoord(ra,dec,unit=(u.hourangle,u.deg)).transform_to(altaz_frame)
    if verbose==True:
        print("Date =",obstime.iso,"JD:",obstime.jd,
              "\nObservatory at Lat=",loc.latitude.deg,"Long=",loc.longitude.deg)
    if sun == False:
        return c_t.alt.value, c_t.secz.value
    else:
        # Sun
        c_s = astropy.coordinates.get_sun(obstime).transform_to(altaz_frame)
        return c_t.alt.value, c_t.secz.value, c_s.alt.value

def norm_flux_SavGol(flux,win=49):
    flux = everest.math.SavGol(flux)
    return flux / np.nanmedian(flux)

def mass_from_radius(r_p):
    """
    From Weiss et al. for 1 R_earth < R < 4 R_earth
    
    r_p = radius of planet in earth radii
    """
    return 2.69*(r_p)**(0.93)

def rv_semiamplitude(m_1,m_2,P,i,e):
    """

    INPUT:
        m_1: mass of star in Solar masses
        m_2: mass of planet in Earth masses
        i:   inclination in degrees
        P: period in days
    
    OUTPUT:
        RV semiamplitude in m/s

    NOTES:
        For Earth:
        rv = gkastro.rv_semiamplitude(m_1=1.,m_2=1.,P=365.245,i=90.,e=0.)
        rv # 0.0887 - correct!
    """
    m_2 = m_2*aconst.M_earth.value/(aconst.M_jup.value) # convert to Jupiter masses
    P = P/356.245 # change to years
    i = i*np.pi/180. # change to radians

    # Perform calculation
    f1 = 28.4329/(np.sqrt(1.-e**2.))
    f2 = m_2*np.sin(i)
    f3 = ((m_1)**(-2/3))*(P**(-1./3))
    return f1*f2*f3

def filter_dates(df,startdates,stopdates):
    """
    A function to filter dates
    
    df - a pandas dataframe with a time index
    startdates - an array of datetimes
    stopdates  - an array of datetimes
    """
    data = []
    for i in range(len(startdates)):
        df_temp = df[str(startdates[i]):str(stopdates[i])]
        if len(df_temp)>0:
            data.append(df_temp)
    if len(data)>0:
        df_final = pd.concat(data)
        print("Found %s overlapping events" % len(df_final))
        return df_final
    else:
        print("No overlapping dates!")
        return None


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

def dist(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2.+(y1-y2)**2.)

def sigma_tc(tau=0.01*24.*60.,gamma=1.,error=4000.,depth=5600.):
    """
    Function to calculate timing error for transit 0 point
    
    INPUT:
    - tau   - the duration of ingress/egress in minutes
    - gamma - observing cadence in obs/minute
    - error - photometric error
    - depth - depth of transit
    
    RETURNS:
    - Error on t_c in minutes
    
    Notes:
    See J.A.C. paper: http://adsabs.harvard.edu/abs/2008ApJ...689..499C
    
    tau and sigma are correlated (photon noise scales as sqrt(Gamma))
    """
    sigma_tc = np.sqrt(tau/(2.*gamma))*error/depth
    return sigma_tc

def sigma_delta(tau=0.01*24.*60.,gamma=1/2.,error=4000.,T=0.142*24.*60.):
    """
    Function to calculate the error in depth in ppm for a transit
    
    INPUT:
    - tau   - the duration of ingress/egress in minutes
    - gamma - observing cadence in obs/minute
    - error - photometric error
    - T     - duration of transit
    
    RETURNS:
    - error on depth in ppm
    
    NOTES:
     See J.A.C. paper: http://adsabs.harvard.edu/abs/2008ApJ...689..499C
    """
    sigma_delta = error/(np.sqrt(1.-tau/T)*(np.sqrt(gamma*T)))
    return sigma_delta

def get_simbad_fluxes(name):
    customSimbad = Simbad()
    customSimbad.add_votable_fields('flux(V)','flux(R)','flux(I)','flux(J)','flux(H)','flux(g)','flux(r)','flux(i)')
    #customSimbad.get_votable_fields()
    table = customSimbad.query_object(name)
    columns = ["FLUX_V","FLUX_R","FLUX_I","FLUX_J","FLUX_H","FLUX_g","FLUX_r","FLUX_i"]
    df = table[columns].to_pandas()
    return df

def sigma_clip_savgol(x,win=101,sigma=3.):
    x_savgol = everest.math.SavGol(x,win=101)
    mask = astropy.stats.sigma_clip(x_savgol,sigma=sigma).mask
    print("Removed",np.sum(mask),"outliers")
    return x[~mask], mask

def sigma_clip_savgol_df_from_column(df,columnname,win=101,sigma=3.):
    #masks = []
    #for columnname in list_of_columns:
    x = df[columnname].values
    _, m = sigma_clip_savgol(x,win=win,sigma=sigma)
    #masks.append(m)
    df = df[~m]
    return df

class FilePath(object):
    """
    A simple filepath class to deal with files

    NOTES:
    """
    def __init__(self,fullpath):
        self._fullpath = fullpath

    @property
    def basename(self):
        return self._fullpath.split(os.sep)[-1]

    @property
    def barename(self):
        return self.basename.split(".")[0]

    @property 
    def extension(self):
        return self._fullpath.split(".")[-1]

    @extension.setter
    def extension(self,value):
        self._fullpath = self.directory + self.barename + "." + value

    @property
    def directory(self):
        return os.sep.join(self._fullpath.split(os.sep)[:-1])+"/"

    @directory.setter
    def directory(self,value):
        if value[-1]==os.sep:
            self._fullpath = value + self.basename
        else:
            self._fullpath = value + "/" + self.basename

    def __str__(self):
        return self._fullpath

    def add_prefix(self,value):
        self._fullpath = self.directory + value + self.basename 

    def add_suffix(self,value):
        self._fullpath = self.directory + self.barename + value + "." + self.extension
    



def chunker(df, size):
    """
    Split dataframe *df* into chunks of size *size*
    """
    return [df[pos:pos + size] for pos in xrange(0, len(df), size)]

def resample_df(df,new_x_array,old_x_col_name,columns_to_resample=None):
    """
    Resample a dataframe using linear interpolation
    
    INPUT:
        df - a pandas dataframe
        new_x_array - the x array to use as the base for the interpolation
        old_x_col_name - the column name of the x-array 
        columns_to_resample - optional list of columns to resample. If None, then resample all of the columns
        
    OUTPUT:
        df_res - a pandas dataframe with linearly interpolated values
        
    NOTES:
    
    EXAMPLE:
        resample_df(df_config_f01_res,[2.5,3.5,3.65,4.,5.,6.,7.],"f_ratio_in")
    """
    if columns_to_resample==None:
        cols = df.keys().tolist()
        cols.remove(old_x_col_name)
    else:
        cols = columns_to_resample
    df_res = pd.DataFrame(columns=[old_x_col_name]+cols)
    df_res[old_x_col_name] = new_x_array
    for key in cols:
        df_res[key] = resample_array(new_x_array,df[old_x_col_name],df[key])
    return df_res
