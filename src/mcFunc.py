from __future__ import print_function
import numpy, math, sys, scipy, emcee, corner, batman, progressbar
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.optimize import minimize
from scipy.stats import binned_statistic, gamma, norm
import numpy as np

import gkastro
from IPython.display import Latex 
import pandas as pd
from matplotlib import rcParams
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
rcParams['font.weight'] = "normal"
rcParams["axes.formatter.useoffset"] = False
rcParams['xtick.major.width']=1
rcParams['xtick.major.size']=4
rcParams['xtick.minor.width']=0.5
rcParams['xtick.minor.size']=2
rcParams['xtick.direction'] = "in"
rcParams['ytick.direction'] = "in"
rcParams['ytick.major.width']=1
rcParams['ytick.major.size']=4
rcParams['ytick.minor.width']=0.5
rcParams['ytick.minor.size']=2
rcParams["savefig.dpi"] = 100
rcParams['xtick.major.pad']=5
rcParams['ytick.major.pad']=5


def latex_mean(mean,std,sig=2):
    #latexstr = "%.10f_{-%.10f}^{+%.10f}" % (mean,low,up)
    f_std, round_to = gkastro.round_sig(std,return_round_to=True,sig=sig)
    f_mean = round(mean, round_to)
    latexstr = "$%s \pm %s$" % (f_mean,f_std)
    return latexstr


def latex_mean_low_up(mean,low,up,sig=2):
    #latexstr = "%.10f_{-%.10f}^{+%.10f}" % (mean,low,up)
    f_low, round_to1 = gkastro.round_sig(low,return_round_to=True,sig=sig)
    f_up,  round_to2 = gkastro.round_sig(up,return_round_to=True,sig=sig)
    round_to = max(round_to1,round_to2)
    f_mean = round(mean, round_to)
    latexstr = "$%s_{-%s}^{+%s}$" % (f_mean,f_low,f_up)
    return latexstr

def calc_medvals2(df,lowerlevel=16.,upperlevel=84):
    """
    USE THIS ONE. NOT THE OTHER ONE.

    [mcFunc.latex_mean_low_up(b[i],c[i],d[i]) for i in range(len(a))]

    A function to calculate the mean values

    INPUT:
        df - a dataframe of posteriors
        minlevel = lower percentile
        maxlevel = upper percentile

    OUTPUT:
        df.keys()
        medvals,
        minus,
        plus
    """
    num_params = len(df.keys())
    medvals = numpy.zeros(num_params)
    minus = numpy.zeros(num_params)
    plus = numpy.zeros(num_params)
    for i,key in enumerate(df.keys()):
        medvals[i] = numpy.median(df[key])
        minus[i] = medvals[i] - numpy.percentile(df[key],lowerlevel)
        plus[i] = numpy.percentile(df[key],upperlevel)- medvals[i]
    return pd.DataFrame(zip(medvals,minus,plus),columns=["medvals","minus","plus"])

def calc_medvals(df,lowerlevel=16.,upperlevel=84):
    """
    A function to calculate the mean values

    INPUT:
        df - a dataframe of posteriors
        minlevel = lower percentile
        maxlevel = upper percentile

    OUTPUT:
        df.keys()
        medvals,
        minus,
        plus
    """
    num_params = len(df.keys())
    medvals = numpy.zeros(num_params)
    minus = numpy.zeros(num_params)
    plus = numpy.zeros(num_params)
    for i,key in enumerate(df.keys()):
        medvals[i] = numpy.median(df[key])
        minus[i] = medvals[i] - numpy.percentile(df[key],lowerlevel)
        plus[i] = numpy.percentile(df[key],upperlevel)- medvals[i]
    return df.keys(),medvals,minus,plus

def print_medvals(labels,description,mean,minus,plus,latex=False,ipynb=False):
    """
    A function to print a nice latex table with labels.
    
    EXAMPLE:
        labels_latex = np.array([["$T_{0}$ $(\mathrm{BJD_{TDB}})$","Transit Midpoint"],
                    ["$P$ (days)","Orbital period"],
                    ["$R_p/R_*$","Radius ratio"],
                    ["$\delta$","Transit depth"],
                    ["$a/R_*$","Normalized orbital radius"],
                    ["$i$ $(^{\circ})$","Transit inclination"],
                    ["$b$","Impact parameter"],
                    ["$e$","Eccentricity"],
                    ["$\omega$ $(^{\circ})$","Argument of periastron"],
                    ["$T_{14}$ (days)","Transit duration"],
                    ["$\\tau$ (days)","Ingress/egress duration"],
                    ["$T_{S}$ $(\mathrm{BJD_{TDB}})$","Time of secondary eclipse"]])
        labels = labels_latex[:,0]
        description = labels_latex[:,1]
        ll = mcFunc.print_medvals(labels,description,mean,minus,plus,latex=True,ipynb=False)
        mcFunc.print_medvals(labels,description,mean,minus,plus,latex=True,ipynb=True)
    """
    if latex==False:
        for i in range(5):
            print("%s \t \t %s \t \t %.10f \t \t %.10f \t \t %.10f" % (labels[i],description[i],mean[i],minus[i],plus[i]))
    if (latex==True and ipynb==False):
        numcols = 3
        latex_table_columns = "".join(["l" for i in range(numcols)])
        latexstr = "\\begin{table} \n"
        latexstr += "\\centering \n"
        latexstr += "\\caption{MCMC transit results} \n"
        latexstr += "\\begin{tabular}{"+ latex_table_columns+"}\n"
        latexstr +="\hline \hline \n"
        latexstr +="Label & Description & Value \\\\ \hline \n"
        for i in range(len(labels)):
            latexstr += "%s \t & %s \t & $%.10f_{-%.10f}^{+%.10f}$ \\\\ \n" % (labels[i],description[i],mean[i],minus[i],plus[i])
        latexstr += "\hline \n"
        latexstr += "\\end{tabular} \n"
        latexstr += "\\label{tab:mcmc}\n"
        latexstr += "\\end{table}\n "
        print(latexstr)
        return(Latex(latexstr))

    elif (latex==True and ipynb==True):
        numcols = 3
        latex_table_columns = "".join(["l" for i in range(numcols)])
        latexstr = "\\begin{array}{"+ latex_table_columns+ "}\n"
        latexstr +="\hline \hline \n"
        latexstr +="Label & Description & Value \\\\ \hline \n"
        for i in range(len(labels)):
            latexstr += "%s \t & \mathrm{%s} \t & %.10f_{-%.10f}^{+%.10f} \\\\ \n" % (labels[i],description[i],mean[i],minus[i],plus[i])
        latexstr += "\hline \n"
        latexstr += "\\end{array} \n"
        return(Latex(latexstr))


import statsmodels.api as sm
def acf(flatchains,lags=100,length=None,labels=None):
    """
    Plot the acf for all of the parameters
    """
    for i in range(flatchains.shape[1]):
        print(i)
        fig, ax = plt.subplots()
        if length!=None:
            sm.graphics.tsa.plot_acf(x=flatchains[:length,i],lags=lags,ax=ax)
        else:
            sm.graphics.tsa.plot_acf(x=flatchains[:,i],lags=lags,ax=ax)

        if labels!=None:
            ax.set_title("ACF: "+labels[i])

def pacf(flatchains,lags=100,length=None,labels=None):
    """
    Plot the pacf for all of the parameters
    """
    for i in range(flatchains.shape[1]):
        print(i)
        fig, ax = plt.subplots()
        if length!=None:
            sm.graphics.tsa.plot_pacf(x=flatchains[:length,i],lags=lags,ax=ax)
        else:
            sm.graphics.tsa.plot_pacf(x=flatchains[:,i],lags=lags,ax=ax)

        if labels!=None:
            ax.set_title("PACF: "+labels[i])



def calc_MCMCerrors(bjd_obs=0.,t0=[0,0],per=[0,0],inc=[0,0],RpRs=[0,0],aRs=[0,0]):
    """
    A function to calculate the errors to go into the MCMC
    """
    print("Assuming:")
    print("BJD_obs: ",bjd_obs)
    print("t0: \t",t0)
    print("per: \t",per)
    print("inc: \t",inc)
    print("aRs: \t",aRs)
    print("")

    # Midpoint
    n = (bjd_obs[0] - t0[0])/per[0]
    print("n",n)
    tranmid = [bjd_obs[1],numpy.sqrt(t0[1]**2. + n*per[1]**2.)]
    print("t0: \t",tranmid[0],"\t \t \t Err t0: \t",tranmid[1])
    
    # Log Period
    logPer = [numpy.log10(per[0]),per[1]/(per[0]*numpy.log(10.))]
    print("Log(P):\t",logPer[0],"\t Err Log(P): \t",logPer[1])
    
    # cos(i)
    cosi = [numpy.cos(inc[0]*numpy.pi/180.),numpy.sin(inc[0]*numpy.pi/180.)*(inc[1])*numpy.pi/180.]
    print("cos(i):\t",cosi[0],"\t Err cos(i): \t",cosi[1])
    
    # RpRs
    print("RpRs:\t",RpRs[0],"\t Err RpRs: \t",RpRs[1])
    
    # Log(aRs)
    _aRs = [numpy.log10(aRs[0]),aRs[1]/(aRs[0]*numpy.log(10.))]
    print("Log(aRs):\t",_aRs[0],"\t Err log(aRs): \t",_aRs[1])

    return tranmid, logPer, cosi, RpRs, _aRs

def plot_chains(chain,labels=None,burn=0,thin=1,plot_priors=False,alpha=0.1):
    """
    chain = self.sampler.chain[:,burn::thin,:]
    """
    chain = chain[:,burn::thin,:]
    ndim = chain.shape[2]
    nwalkers = chain.shape[0]
    fig, ax = plt.subplots(nrows=ndim,sharex=True)
    for i in range(ndim):
        for walker in range(nwalkers):
            ax[i].plot(chain[walker,:,i],color="black",alpha=alpha,lw=0.5);
            #if plot_priors==True:
            #    ax[i].hlines(self.pars[i],0,chain.shape[1],lw=1,color=cp[2])
        if labels:
            ax[i].set_ylabel(labels[i],fontsize=8)
        ax[i].margins(y=0.1)
        for label in ax[i].get_yticklabels():
            label.set_fontsize(6)

    ax[i].set_xlabel("sample",fontsize=8)
    ax[i].minorticks_on()
    ax[0].set_title("Overview of chains",y=1.03,fontsize=12)
    for label in ax[i].get_xticklabels():
            label.set_fontsize(6)
    fig.subplots_adjust(hspace=0.015)

def plot_corner(chain,labels=None,burn=0,thin=1,title_fmt=".5f",**kwargs):
    """
    Plot a nice corner plot for the jump parameters.
    """
    rcParams["lines.linewidth"] = 1.0
    rcParams["axes.labelpad"] = 20.0
    rcParams["xtick.labelsize"] = 14.0
    rcParams["ytick.labelsize"] = 14.0

    ndim = chain.shape[2]
    chain = chain[:,burn::thin,:].reshape((-1, ndim))

    figure = corner.corner(chain,
                           labels=labels,
                           quantiles=[0.16, 0.5, 0.84],
                           verbose=False,
                           title_kwargs={"fontsize": 14},
                           hist_kwargs={"lw":1.},
                           label_kwargs={"fontsize":18},
                           xlabcord=(0.5,-0.45),
                           ylabcord=(-0.45,0.5),
                           title_fmt=title_fmt,**kwargs)

def gelman_rubin(chains):
    """
    Calculates the gelman rubin statistic.

    # NOTE: 
    Should be close to 1
    """
    nwalker = chains.shape[0]
    niter = chains.shape[1]
    npar = chains.shape[2]
    grarray = numpy.zeros(npar)
    for i in range(npar):
        sj2 = numpy.zeros(nwalker)
        chainmeans = numpy.zeros(nwalker)
        for j in range(nwalker):
            chainmeans[j] = numpy.mean(chains[j,:,i])
            sj2[j] = numpy.sum((chains[j,:,i]-chainmeans[j])**2.) / (niter-1)
        W = numpy.sum(sj2) / nwalker
        ThetaDoubleBar = numpy.sum(chainmeans) / nwalker
        B = numpy.sum((chainmeans-ThetaDoubleBar)**2.) * niter / (nwalker-1)
        VarTheta = (1-(1/niter))*W + (B/niter)
        grarray[i] = numpy.sqrt(VarTheta/W)
    return grarray

def u1_u2_from_q1_q2(q1,q2):
    u1, u2 = 2.*np.sqrt(q1)*q2, np.sqrt(q1)*(1.-2*q2)
    return u1, u2

def q1_q2_from_u1_u2(u1,u2):
    q1, q2 = (u1 + u2)**2. , u1/(2.*(u1+u2))
    return q1, q2

def err_cosi_from_i(inc=(0,0)):
    cosi = [np.cos(inc[0]*np.pi/180.),np.sin(inc[0]*np.pi/180.)*(inc[1])*np.pi/180.]
    return cosi

def err_logP_from_p(per=(0,0)):
    logPer = [np.log10(per[0]),per[1]/(per[0]*np.log(10.))]
    return logPer

def err_logaRs_from_aRs(aRs=(0,0)):
    aRs = [np.log10(aRs[0]),aRs[1]/(aRs[0]*np.log(10.))]
    return aRs

def err_Tc_from_T0_and_P(jd_date_of_obs=0.,t0=(0,0),per=(0,0)):
    # Midpoint
    n = int((jd_date_of_obs - t0[0])/per[0])
    #print("n",n)
    tranmid = [t0[0] + n*per[0],np.sqrt(t0[1]**2. + n*per[1]**2.)]
    return tranmid


def get_mean_values_for_posterior(posterior,latexlabel="",description=""):
    """
    INPUT:
    posterior - a array of posterior values
    latexlabel- a latexlabel
    description-a description of the posterior
    
    OUTPUT:
    a pandas dataframe with:
    - medvals - minus - plus - values - Labels - Description
    """
    dff = pd.DataFrame(data=posterior,columns=["_"])
    df =  calc_medvals2(dff)
    df["values"] =  [latex_mean_low_up(df.medvals[i],df.minus[i],df.plus[i]) for i in range(len(df))]
    df["Labels"] =  [latexlabel]
    df["Description"] = [description]
    return df

def err_sqrtecosw_from_e_and_w(e=(0,0),w=(0,0)):
    """
    A function to calculate the value, and error of sqrtcosw
    
    INPUT:
    e - eccentricity, tuple: (value, error)
    w - omega, tuple (value, error), in degrees
    
    OUTPUT:
    sqrtcosw, sqrtcosw_err (using radians)
    
    EXAMPLE:
    print(err_sqrtecosw_from_e_and_w(e=(0.13,0.12),w=(343.,9.6)))

    """
    w = np.deg2rad(w)
    sqrtcosw = np.sqrt(e[0])*np.cos(w[0])
    a1 = (np.cos(w[0])*0.5/np.sqrt(e[0]))*e[1]
    a2 = (np.sqrt(e[0])*np.sin(w[0]))*w[1]
    sqrtcosw_err = np.sqrt(a1**2. + a2**2.)
    return sqrtcosw, sqrtcosw_err

def err_sqrtesinw_from_e_and_w(e=(0,0),w=(0,0)):
    """
    A function to calculate the value, and error of sqrtsinw
    
    INPUT:
    e - eccentricity, tuple: (value, error)
    w - omega, tuple (value, error), in degrees
    
    OUTPUT:
    sqrtsinw, sqrtsinw_err (using radians)
    
    EXAMPLE:
    print(err_sqrtesinw_from_e_and_w(e=(0.13,0.12),w=(343.,9.6)))
    """
    w = np.deg2rad(w)
    sqrtsinw = np.sqrt(e[0])*np.sin(w[0])
    a1 = (np.sin(w[0])*0.5/np.sqrt(e[0]))*e[1]
    a2 = (np.sqrt(e[0])*np.cos(w[0]))*w[1]
    sqrtsinw_err = np.sqrt(a1**2. + a2**2.)
    return sqrtsinw, sqrtsinw_err

