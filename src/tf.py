import math
from exotk.utils.likelihood import ll_normal_ev_py, ll_normal_es, ll_normal_ev
from scipy.optimize import minimize
import IPython
import ipywidgets
ipywidgets.IntProgress
from IPython.display import display
from IPython.display import HTML
import corner
import emcee
import mcFunc
import numpy as np
import pandas as pd
import sys
import exotk.priors
from exotk.priors import PriorSet, UP, NP, JP
import batman
import pyde
from astropy import constants as aconst
import matplotlib.pyplot as plt

class TransitFit(object):
    """
    A class that does transit fitting.
    
    NOTES:
    - Needs to have LPFunction defined
    TODO:
    """
    def __init__(self,LPFunction,convergence_diagnostics=False):
        self.lpf = LPFunction(convergence_diagnostics=convergence_diagnostics)
        self.latex_labels_arr       = np.array([["$T_{0}$ $(\mathrm{BJD_{TDB}})$","Transit Midpoint"],
                                                ["$P$ (days)","Orbital period"],
                                                ["$R_p/R_*$","Radius ratio"],
                                                ["$R_p (R_\oplus)$","Planet radius"],
                                                ["$R_p (R_J)$","Planet radius"],
                                                ["$\delta$","Transit depth"],
                                                ["$a/R_*$","Normalized orbital radius"],
                                                ["$a$ (AU)","Semi-major axis"],
                                                ["$i$ $(^{\circ})$","Transit inclination"],
                                                ["$b$","Impact parameter"],
                                                ["$e$","Eccentricity"],
                                                ["$\omega$ $(^{\circ})$","Argument of periastron"],
                                                ["$T_{\mathrm{eq}}$(K)","Equilibrium temperature"],
                                                ["$T_{14}$ (days)","Transit duration"],
                                                ["$\\tau$ (days)","Ingress/egress duration"],
                                                ["$T_{S}$ $(\mathrm{BJD_{TDB}})$","Time of secondary eclipse"]])
        self.latex_labels = self.latex_labels_arr[:,0]
        self.latex_description = self.latex_labels_arr[:,1]
        self.latex_jump_labels = [r"$T_0 (BJD_{\mathrm{TBD}})$",r"$\log(P)$",r"$\cos(i)$","$R_p/R_*$","$\log(a/R_*)$","Baseline","Airmass",r"$x$-centroid",r"$y$-centroid"]
    
    def minimize_AMOEBA(self):
        #random = self.lpf.ps.random
        centers = np.array(self.lpf.ps.centers)
        
        def neg_lpf(pv):
            return -1.*self.lpf(pv)
        self.min_pv = minimize(neg_lpf,centers,method='Nelder-Mead',tol=1e-9,
                                   options={'maxiter': 100000, 'maxfev': 10000, 'disp': True}).x
        if self.lpf.convergence_diagnostics==True:
            self.lpf.df_minimize = self.lpf.df_minimize.dropna()
            self.ax = self.lpf.df_minimize.plot(subplots=True,sharex=True,fontsize=5,lw=1)
            mmin = self.lpf.df_minimize.index.min()
            mmax = self.lpf.df_minimize.index.max()
            self.ax[0].set_title("Convergence plot")
            [self.ax[i].hlines(self.lpf.ps.pmins[i],mmin,mmax,lw=1,color="orange",linestyles="--") for i in range(len(self.ax)-1)]
            [self.ax[i].hlines(self.lpf.ps.pmaxs[i],mmin,mmax,lw=1,color="orange",linestyles="--") for i in range(len(self.ax)-1)]
            [self.ax[i].legend(fontsize=8) for i in range(len(self.ax))]
            [self.ax[i].minorticks_on() for i in range(len(self.ax))]
            self.ax[len(self.ax)-1].set_yscale("log")
            # plot log_likelihood plot
            self.fig, self.ax = plt.subplots()
            self.ax.plot(self.lpf.df_minimize["log_ln"][200:],label="log_ln")
            self.ax.legend(loc="upper right")
            self.ax.set_xlabel("Iteration")
            
    
    def minimize_PyDE(self,npop=100,de_iter=200,mc_iter=1000,mcmc=True,threads=8,maximize=True):
        """
        Minimize using the PyDE
        
        NOTES:
        https://github.com/hpparvi/PyDE
        """
        centers = np.array(self.lpf.ps.centers)
        print("Running PyDE Optimizer")
        self.de = pyde.de.DiffEvol(self.lpf, self.lpf.ps.bounds, npop, maximize=maximize) # we want to maximize the likelihood
        self.min_pv, self.min_pv_lnval = self.de.optimize(ngen=de_iter)
        print("Optimized using PyDE")
        print("Final parameters:",self.min_pv)
        print("LogLn value:",self.min_pv_lnval)
        print("Log priors",self.lpf.ps.c_log_prior(self.min_pv))
        if mcmc:
            print("Running MCMC")
            self.sampler = emcee.EnsembleSampler(npop, self.lpf.ps.ndim, self.lpf,threads=threads)
            
            pb = ipywidgets.IntProgress(max=mc_iter/50)
            display(pb)
            for i,c in enumerate(self.sampler.sample(self.de.population,iterations=mc_iter)):
                if i%50 == 0:
                    pb.value += 1
            print("Finished MCMC")

    def plot_chains(self,labels=None,burn=0,thin=1):
        print("Plotting chains")
        if labels==None:
            labels = self.lpf.ps.descriptions
        mcFunc.plot_chains(self.sampler.chain,labels=labels,burn=burn,thin=thin)

    def plot_corner(self,labels=None,burn=0,thin=1,title_fmt='.5f',**kwargs):
        if labels==None:
            labels = self.lpf.ps.descriptions
        mcFunc.plot_corner(self.sampler.chain,labels=labels,burn=burn,thin=thin,title_fmt=title_fmt,**kwargs)
        
    def get_transit_parameters(self,flatchain=None,burn=0,thin=1,st_rad=1.0,st_raderr1=0.022,st_teff=5650.,st_teff_err1=75.,e="fixed"):
        if flatchain==None:
            flatchain = self.sampler.chain[:,burn::thin,:].reshape((-1, self.lpf.ps.ndim))
            
        print("Assuming")
        print("R_s:",st_rad,"+-",st_raderr1)
        print("Teff:",st_teff,"+-",st_teff_err1)
        # Working with the posteriors
        #sampler = self.sampler
        t0 = flatchain[:,0]
        self._p_pl_orbper = 10.**flatchain[:,1]
        cosi = flatchain[:,2]
        inc = np.arccos(flatchain[:,2])
        incdeg = inc*180./math.pi
        sini = np.sin(inc)
        RpRs = flatchain[:,3]
        depth = RpRs**2.
        aRs = 10.**flatchain[:,4]
        if e=="fixed":
            e = np.zeros(len(flatchain)) #sampler.flatchain[:,5]**2. + sampler.flatchain[:,6]**2.
            w = np.zeros(len(flatchain)) #numpy.arctan(sampler.flatchain[:,5]/sampler.flatchain[:,6])
            #w = numpy.nan_to_num(w) # removes the nans
            #w = w*180./math.pi
            ecosw = np.sqrt(e)#*flatchain[:,5]                       
            esinw = np.sqrt(e)#*flatchain[:,6]    
        else:
            e = flatchain[:,5]**2. + flatchain[:,6]**2.
            w = np.arctan(flatchain[:,5]/flatchain[:,6])
            w = np.nan_to_num(w) # removes the nans
            w = w*180./math.pi
            ecosw = e*np.cos(w)#flatchain[:,5]                       
            esinw = e*np.sin(w)#flatchain[:,6]    
        b = aRs*cosi*(1.-e**2.)/(1.+esinw)
        
        t14 = np.copy(b) * 0.0
        t23 = np.copy(b) * 0.0
        notgrazing = np.where(((1.+RpRs)**2. - b**2.)>0.0)
        t14[notgrazing] = self._p_pl_orbper[notgrazing]/math.pi*np.arcsin(np.sqrt((1.+RpRs[notgrazing])**2. - b[notgrazing]**2.)/(sini[notgrazing]*aRs[notgrazing]))*np.sqrt(1.-e[notgrazing]**2.)/(1.+esinw[notgrazing])
        notgrazing = np.where(((1.-RpRs)**2. - b**2.)>0.0)
        t23[notgrazing] = self._p_pl_orbper[notgrazing]/math.pi*np.arcsin(np.sqrt((1.-RpRs[notgrazing])**2. - b[notgrazing]**2.)/(sini[notgrazing]*aRs[notgrazing]))*np.sqrt(1.-e[notgrazing]**2.)/(1.+esinw[notgrazing])
        t14 = np.nan_to_num(t14) # removes the nans
        t23 = np.nan_to_num(t23) # removes the nans
        tau = (t14-t23)/2.
        Tfwhm = t14-tau

        # Calculate the peri. time based on tC
        nu = math.pi/2. - w # true anomaly for transit away from peri.
        E = 2.*np.arctan(np.sqrt((1.0-e)/(1.+e))*np.tan(nu/2.)) # ecc. anomaly
        M = E - e*np.sin(E) # mean anomaly
        periPhase = M/(2.*math.pi)
        tP = t0 - self._p_pl_orbper*periPhase # peri time

        nu = 3.*math.pi/2. - w # true anomaly for eclipse away from peri.
        E = 2.*np.arctan(np.sqrt((1.0-e)/(1.+e))*np.tan(nu/2.)) # ecc. anomaly
        M = E - e*np.sin(E) # mean anomaly
        periPhase = M/(2.*math.pi)
        tS = tP + self._p_pl_orbper*periPhase + self._p_pl_orbper
        otherside = np.where(periPhase>0.0) # correct for aliases
        tS[otherside] = tP[otherside] + self._p_pl_orbper[otherside]*periPhase[otherside]

        # Create sampling of stellar parameters to calculate errors on compounding derived parameters
        # Teq
        Teff = np.random.normal(loc=st_teff,scale=st_teff_err1,size=len(flatchain))
        Teq = Teff*np.sqrt(1./(2.*aRs))
        # a-au
        R_s = np.random.normal(loc=st_rad,scale=st_raderr1,size=len(flatchain)) # radius of star
        aAU = aRs * R_s * aconst.R_sun.value / aconst.au.value
        # r_e
        R_e = RpRs * R_s * aconst.R_sun.value / aconst.R_earth.value
        R_j = RpRs * R_s * aconst.R_sun.value / aconst.R_jup.value

        self.df_post = pd.DataFrame(zip(t0,self._p_pl_orbper,RpRs,R_e,R_j,depth,aRs,aAU,incdeg,b,e,w,Teq,t14,tau,tS),
                                 columns=["t0","per","RpRs","R_e","R_j","depth","aRs","aAU","inc","b","e","w","Teq","t14","tau","tS"])

        # ugh
        nump = tS.size
        self._p_pl_orbper = self._p_pl_orbper.reshape((nump,1))
        tS = tS.reshape((nump,1))
        depth = depth.reshape((nump,1))
        aRs = aRs.reshape((nump,1))
        incdeg = incdeg.reshape((nump,1))
        b = b.reshape((nump,1))
        Tfwhm = Tfwhm.reshape((nump,1))
        tau = tau.reshape((nump,1))
        t14 = t14.reshape((nump,1))
        ecosw = ecosw.reshape((nump,1))
        esinw = esinw.reshape((nump,1))
        e = e.reshape((nump,1))
        w = w.reshape((nump,1))

        df = mcFunc.calc_medvals2(self.df_post)
        print(len(df))
        df["values"] =  [mcFunc.latex_mean_low_up(df.medvals[i],df.minus[i],df.plus[i]) for i in range(len(df))]
        df["Labels"] =  self.latex_labels
        df["Description"] = self.latex_description
        self.df_medvals = df
        return self.df_medvals
    
    def get_latex_table(self,outputfile=None):
        """
        Prints a nice latex table. Can save output too.

        NOTE:
         look at self.df_medvals.values if something is being truncated (...).
         
         table = self.df_medvals[["Labels","Description","values"]].to_latex(index=False,escape=False)
        """
        table = self.df_medvals[["Labels","Description","values"]].to_latex(index=False,escape=False)
        print(table)
        if outputfile!=None:
            f = open(outputfile,"wr")
            f.write(table)
            f.close()
            print("Saved table to file:",outputfile)
        return table

    def plot_lc(self,pv,times=None):
        """
        Plot the light curve for a given set of parameters pv
        
        INPUT:
        pv - an array containing a sample draw of the parameters defined in self.lpf.ps
        
        EXAMPLE:
        
        """
        self.scaled_flux   = self.lpf.data["flux"]/pv[self.lpf.number_pv_baseline]
        self.scaled_flux_no_trend = self.scaled_flux - self.lpf.detrend(pv)
        self.model_trend   = self.lpf.compute_lc_model(pv)
        self.model_no_trend= self.lpf.compute_transit(pv)
        self.residual      = self.scaled_flux - self.model_trend
        self.scaled_error  = self.lpf.data["error"]/pv[self.lpf.number_pv_baseline]
        
        nrows = 3
        self.fig, self.ax = plt.subplots(nrows=nrows,sharex=True)
        self.ax[0].errorbar(self.lpf.data["time"],self.scaled_flux,yerr=self.scaled_error,elinewidth=0.3,lw=0,alpha=0.5,marker="o",markersize=4,label="Data with trend")
        self.ax[0].plot(self.lpf.data["time"],self.model_trend,label="Model with trend")
        
        self.ax[1].errorbar(self.lpf.data["time"],self.scaled_flux_no_trend,yerr=self.scaled_error,elinewidth=0.3,lw=0,alpha=0.5,marker="o",markersize=4,label="Data, no trend")
        self.ax[1].plot(self.lpf.data["time"],self.model_no_trend,label="Model no trend")

        self.ax[2].plot(self.lpf.data["time"],self.residual,label="residual, std="+str(np.std(self.residual)),lw=0,marker="o",ms=3)
        [self.ax[i].minorticks_on() for i in range(nrows)]
        [self.ax[i].legend(loc="lower left",fontsize=8) for i in range(nrows)]
        self.ax[-1].set_xlabel("Time (BJD)")
        [self.ax[i].set_ylabel("Rel Flux") for i in range(nrows)]
        self.ax[0].set_title("Light curve")

    
    def plot_lc_fit(self):
        """
        Plot the best fit
        """
        #if hasattr(self,'min_pv'):
        #    self.lpf.plot_lc(self.min_pv)
        #else:
            #self.minimize()
        self.plot_lc(self.min_pv)

    def plot_lc_mcmc_fit(self,times=None): 
        df = self.get_mean_values_mcmc_posteriors()
        self.plot_lc(df.medvals.values)      

    def plot_lc_mcmc_fit_with_custom_times(self,times): 
        df = self.get_mean_values_mcmc_posteriors()
        self.pv_mcmc_min = df.medvals.values



        self.plot_lc(df.medvals.values)      
    

    def gelman_rubin(self,chains=None,burn=0,thin=1):
        """
        Calculates the gelman rubin statistic.

        # NOTE: 
        Should be close to 1
        """
        if chains==None:
            chains = self.sampler.chain[:,burn::thin,:]
        grarray = mcFunc.gelman_rubin(chains)
        return grarray
        
    def get_mean_values_mcmc_posteriors(self):
        df_list = [mcFunc.get_mean_values_for_posterior(self.sampler.flatchain[:,i],label,label) for i,label in enumerate(self.lpf.ps.descriptions)]
        return pd.concat(df_list) 

    def get_mean_value_for_posterior(self,posteriror,label):
        return mcFunc.get_mean_values_for_posterior(posteriror,label,label)
        
    def __call__(self,plot_lc_fit=False,plot_lc_mcmc_fit=True,plot_chains=False,plot_corner=False):
        self.minimize_PyDE(mcmc=True)
        if plot_lc_fit:
            print("Plotting PyDE optimized transit")
            self.plot_lc_fit()
        if plot_lc_mcmc_fit:
            print("Plotting MCMC optimized transit")
            self.plot_lc_mcmc_fit()
        if plot_chains:
            print("Plotting chains")
            self.plot_chains()
        if plot_corner:
            self.plot_corner()
            
