import everest
import numpy as np
import matplotlib.pyplot as pl
import astropy
import sigma_clipping

class EverestMOD(everest.Everest):
    """
    Inheriting the everest class to create modified functions. Otherwise works exactly like the Everest package
    """

    def __init__(self,ID):
        everest.Everest.__init__(self,ID)
        
    def get_folded_transit(self, t0, period, dur = 0.2,sigma=5.,plot=True):
        '''
        Extending plot_folded() from the original EVEREST package.
        
        INPUT:
            t0 - the midpoint of the transit
            period - period of the transit in days
            dur - duration of window to use in Everest.compute()
            sigma - numbers of sigma to clip (outliers)
            plot - if == True, plot a plot_folded plot

        OUTPUT:
            self.time_phased - the phased time
            self.flux_phased - the phased flux
            
        NOTES:
        Can access:
        self.time_phased, self.flux_phased - the time and flux (phased)
        '''
        # Mask the planet
        self.mask_planet(t0, period, dur)

        # Whiten
        gp = everest.gp.GP(self.kernel, self.kernel_params, white = False)
        gp.compute(self.apply_mask(self.time), self.apply_mask(self.fraw_err))
        med = np.nanmedian(self.apply_mask(self.flux))
        y, _ = gp.predict(self.apply_mask(self.flux) - med, self.time)
        fwhite = (self.flux - y)
        fwhite /= np.nanmedian(fwhite)

        # Fold
        tfold = (self.time - t0 - period / 2.) % period - period / 2. 

        # Crop
        inds = np.where(np.abs(tfold) < 2 * dur)[0]
        
        #self.time_masked, self.flux_flattened_masked - the time and flux_flattened (not phased)
        #m = astropy.stats.sigma_clip(fwhite,sigma=sigma).mask
        #self.time_masked = self.time[~m]
        #self.flux_flattened_masked = fwhite[~m]
        self.time_phased = tfold[inds]
        self.flux_phased = fwhite[inds]
        
        # sigma clip
        m = sigma_clipping.sigma_clip(self.flux_phased,sig=sigma).mask
        self.flux_phased = self.flux_phased[~m]
        self.time_phased = self.time_phased[~m]
        
        if plot:
            # Plot
            fig, ax = pl.subplots(1, figsize = (9, 5))
            fig.subplots_adjust(bottom = 0.125)
            ax.plot(self.time_phased, self.flux_phased, 'k.', alpha = 0.5)

            # Get ylims
            yfin = np.delete(self.flux_phased, np.where(np.isnan(self.flux_phased)))
            lo, hi = yfin[np.argsort(yfin)][[3,-3]]
            pad = (hi - lo) * 0.1
            ylim = (lo - pad, hi + pad)
            ax.set_ylim(*ylim)

            # Appearance
            ax.set_xlabel(r'Time (days)', fontsize = 18)
            ax.set_ylabel(r'Normalized Flux', fontsize = 18)
            fig.canvas.set_window_title('%s %d' % (self._mission.IDSTRING, self.ID))

            pl.show()
            
        return self.time_phased, self.flux_phased
