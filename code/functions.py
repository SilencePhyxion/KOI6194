import pandas as pd, numpy as np, lightkurve as lk, exoplanet as xo, astropy.units as u, multiprocessing as mp
import matplotlib.pyplot as plt, matplotlib.pylab as pylab, matplotlib.gridspec as gridspec
import arviz as az, pickle, corner, csv
import pymc3 as pm, pymc3_ext as pmx, aesara_theano_fallback.tensor as tt

from astropy.io import fits
from scipy import signal
from sinusoid import SinusoidModel
from astropy.timeseries import LombScargle
from astropy.table import Table
from functions import * # <-- self-defined functions
from IPython.display import display, Math
from celerite2.theano import terms, GaussianProcess
from PIL import Image

def model_sampling(model, map_soln, tune, draws, cores, chains, seed):
    with model:
        trace = pm.sample(
            tune=tune,
            draws=draws,
            initvals=map_soln,
            cores=cores,
            chains=chains,
            target_accept=0.95,
            return_inferencedata=True,
            random_seed = seed, #[203771098, 203775000],
            mp_ctx=mp.get_context("fork"),
            init="adapt_full",
        )
    return trace

def Annotate(txt, x,y,fontsize):
    ax.annotate(
    txt,
    (0, 0),
    xycoords="axes fraction",
    xytext=(x,y),
    textcoords="offset points",
    ha="left",
    va="bottom",
    fontsize=fontsize,
    bbox=dict(facecolor='white')
    )      

#from lightkurve documentation
nu_maxsolar = 3090
Dnu_solar = 135.1
T_eff_solar = 5777.2

def f_Dnu(Dnu,numax,Teff,mh):
    fdnu =  4.015+0.168*np.log10(numax/3090)-0.186*np.log10(Dnu/135.1) - \
        10.234*(Teff/5777.2)+11.432*(Teff/5777.2)**2-4.2*(Teff/5777.2)**3+0.001*mh
    return fdnu

def MassRelation(Dnu, numax,T,f_dnu):
    return (numax/nu_maxsolar)**3 * (Dnu/(f_dnu*Dnu_solar))**(-4) * (T/T_eff_solar)**(3/2)

def MassUnc(Dnu,Dnu_err,numax,numax_err,T,T_err, f_dnu):
    return MassRelation(Dnu,numax,T,f_dnu) * np.sqrt((3* Dnu_err/Dnu)**2 + (4 * numax_err/numax)**2 + ((3/2) * T_err/T)**2)

def RadRelation(Dnu, numax,T,f_dnu):
    return (numax/nu_maxsolar) * (Dnu/(f_dnu*Dnu_solar))**(-2) * (T/T_eff_solar)**(1/2)

def RadUnc(Dnu,Dnu_err,numax,numax_err,T,T_err, f_dnu):
    return RadRelation(Dnu,numax,T,f_dnu) * np.sqrt(( Dnu_err/Dnu)**2 + (2 * numax_err/numax)**2  + ((1/2) * T_err/T)**2)

def DensityRel(Dnu,f_dnu):
    return ((Dnu/(f_dnu*Dnu_solar))**2) * (1.4)

def DensUnc(Dnu,Dnu_err, f_dnu):
    return DensityRel(Dnu,f_dnu) * np.sqrt((2*Dnu_err/Dnu)**2)

def GravRelation(numax,T):
    return ((numax/nu_maxsolar) * np.sqrt(T/T_eff_solar)) * 28.02

def GravUnc(numax,numax_err,T,T_err):
    return GravRelation(numax,T) * np.sqrt((numax_err/numax)**2 + ((1/2) * T_err/T)**2)

def GaussianChecking(Lower_xlim, Upper_xlim, PlotNumber, figX, figY,mask,extras0,map_soln0,save=True,name=""):
    if mask is None:
        mask = np.ones(len(time_cull), dtype=bool)
    else:
        mask = mask
    cols = int(PlotNumber/2)
    rows = 2
    p = map_soln0["period"][0]# type: ignore 
    t0 = map_soln0["t0"][0] # type: ignore 
    Initial_Limits = np.array([Lower_xlim,Upper_xlim]) # reference point for the first transit epoch
    gp_mod = extras0["gp_pred"] + map_soln0["mean"] # type: ignore

    fig, axes = plt.subplots(cols,rows, figsize=(figX,figY), sharex=False, sharey=True)
    for index in range(PlotNumber):
        col = index // rows
        row = index % rows
        j = index
        if index == 3: # data is chopped too late for this one, skip it
            j += 1 
        ax = axes[col, row]
        ax.plot(time_cull[mask], flux_cull[mask], ".k", label="data")
        ax.plot(time_cull[mask], gp_mod, color="C2", label="gp model")
        ax.plot(
        time_cull[mask],(map_soln0["light_curves"][:,0])[mask]+1,lw=2,label="Transit Model",color="r" # type: ignore
        )
        ax.legend(loc='lower right',fontsize=15)
        ax.set_xlim(Initial_Limits[0] + j *p, Initial_Limits[1] + j *p)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
    fig.text(0.53, 0.04, "Time [BKJD - 2454833]", ha='center', fontsize=25)
    fig.text(-0.015, 0.5, "Relative flux", va='center', rotation='vertical', fontsize=25)

    plt.tight_layout(rect=[0, 0.05, 1, 1]) 
    if save==True:
        fig.savefig(name, facecolor='w',dpi=100,bbox_inches='tight')
    return

def plot_light_curve(data_arr, lightcurve, day_filter, og_map_soln, map_soln, extras, mask=None,save=True):
    time_cull, flux_cull, flux_err_cull = data_arr
    if mask is None:
        mask = np.ones(len(time_cull), dtype=bool)

    p_map0 = og_map_soln["period"][0]# type: ignore 
    t0_map0 = og_map_soln["t0"][0] # type: ignore 
    other_map0 =  og_map_soln["light_curves"][:,0]# type: ignore
    gp_mod = extras["gp_pred"] + map_soln["mean"]

    figure, axes = plt.subplots(1,2, figsize=(20, 8), sharex=False)
    
    # Plot the folded data
    ax = axes[0]
    x_fold_map0= (time_cull - t0_map0 + 0.5 * p_map0) % p_map0 - 0.5 * p_map0
    ax.scatter(x_fold_map0, flux_cull)
    
    #binned data
    binned = lightcurve.fold(period=p_map0, epoch_time=t0_map0).bin(.008).remove_outliers()
    flux_binned = binned.flux
    time_binned = binned.time.value
    ax.scatter(time_binned,flux_binned, label='binned', zorder=2)
    
    #plotting folded model
    inds = np.argsort(x_fold_map0) #organises the time data in increasing order
    inds = inds[np.abs(x_fold_map0)[inds] < day_filter]
    ax.plot(x_fold_map0[inds], (other_map0+1)[inds], label="model",zorder=1000,color="C3",lw=4)

    ax.legend(loc="lower right",fontsize=35)
    ax.tick_params(axis='both', which='major', labelsize=35)
    ax.set_xlim(-1,1)
    ax.set_ylim(0.9985,1.0015)

    # Plot the folded data
    ax = axes[1]
    gp_scaled = flux_cull-gp_mod
    ax.scatter(x_fold_map0, gp_scaled+1)
    #binned data
    lcgp = lk.LightCurve(time=time_cull, flux=gp_scaled,flux_err=flux_err_cull-gp_mod).fold(period=p_map0, epoch_time=t0_map0).bin(0.008) # type: ignore
    flux_gp = lcgp.flux.value
    time_gp = lcgp.time.value
    ax.scatter(time_gp,flux_gp+1, label='binned', zorder=2)

    #plotting folded model
    inds = np.argsort(x_fold_map0) #organises the time data in increasing order
    inds = inds[np.abs(x_fold_map0)[inds] < day_filter]
    ax.plot(x_fold_map0[inds], (other_map0+1)[inds], label="model",zorder=1000,color="C3",lw=4)
    # ax.set_xlabel('Phase')
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.set_xlim(-1,1)
    ax.set_ylim(0.9985,1.0015)
    ax.tick_params(axis='both', which='major', labelsize=35)
    ax.legend(loc="lower right",fontsize=20)
    figure.text(-0.03, 0.5, "Relative flux", va='center', rotation='vertical', fontsize=35)
    figure.text(0.53, -0.03, "Phase (Days)", ha='center', fontsize=35)
    plt.tight_layout()
    if save==True:
        figure.savefig('ASTERO+Transit+GPTransit.png', facecolor='w',bbox_inches='tight')
    return 

def MAPFitting(dat_arr,rv_dat_arr,map_soln,extras,mask):
    time_cull, flux_cull, flux_err_cull = dat_arr
    t_rv, t_BKJD, rv, rv_err = rv_dat_arr
    if mask is None:
        mask = np.ones(len(time_cull), dtype=bool)

    #plotting simulated data against maximum posteriori model
    fig, axes = plt.subplots(2,2,figsize=(13, 10),sharey=False, sharex=False)
    p = map_soln["period"]# type: ignore 
    t0 = map_soln["t0"] # type: ignore 

    ax = axes[0,0]
    ax.plot(time_cull,flux_cull,".k",ms=4,label="data")
    ax.plot(
        time_cull,map_soln["light_curves"][:,0]+1,lw=2,label="'Candidate' Planet b", # type: ignore
    )
    ax.set_xlabel("Time [days]")
    ax.legend(fontsize=10)
    _ = ax.set_title("Map Model")
    ax.set_ylabel("Relative flux")

    # Compute the median of posterior estimate of the contribution from
    # the other planet. Then we can remove this from the data to plot
    # just the planet we care about.
    other =  map_soln["light_curves"][:,0]# type: ignore

    # Plot the folded data
    ax = axes[0,1]
    gp_mod = extras["gp_pred"] + map_soln["mean"] # type: ignore
    gp_scaled = flux_cull[mask]-gp_mod[mask]

    x_fold= (time_cull - t0[0] + 0.5 * p[0]) % p[0] - 0.5 * p[0]
    ax.scatter(x_fold, gp_scaled + 1)
    #binned data
    lcgp = lk.LightCurve(time=time_cull,
                          flux=gp_scaled,
                          flux_err=flux_err_cull-gp_mod).fold(period=p[0], # type: ignore
                                                               epoch_time=t0[0]).bin(0.008).remove_outliers() # type: ignore
    flux_binned = lcgp.flux
    time_binned = lcgp.time.value
    flux_err_binned = lcgp.flux_err
    ax.scatter(time_binned,flux_binned+1, label='binned', zorder=2)
    #plotting folded model
    inds = np.argsort(x_fold) #organises the time data in increasing order
    inds = inds[np.abs(x_fold)[inds] < 2]
    ax.plot(x_fold[inds], (other+1)[inds], label="model",zorder=1000,color="C3",lw=2)

    ax.legend(fontsize=10, loc=4)
    ax.set_xlabel("time since transit [days]")
    ax.set_ylabel("relative flux")
    ax.set_title("KOI-6194.01 - GP")
    _ = ax.set_xlim(-1, 1)

    #plotting RV model
    ax = axes[1,0]
    ax.errorbar(t_BKJD, rv, yerr=rv_err, fmt=".k")
    ax.plot(t_rv, map_soln["vrad_pred"], "--k", label="Planet induced RV",alpha=0.5) # type: ignore # RV induced by planets 
    ax.plot(t_rv, map_soln["bkg_pred"], ":r", label="Background")  # type: ignore # RV induced by background sources
    ax.plot(t_rv, map_soln["rv_model_pred"], label="model") # type: ignore
    ax.legend(fontsize=10)
    _ = ax.set_xlabel("BKJD [days]")
    ax.set_ylabel("radial velocity [m/s]")

    ax = axes[1,1]
    other1 =  map_soln["vrad"][:,0]# type: ignore

    # Plot the folded data
    x_fold1= (t_BKJD - t0[0] + 0.5 * p[0]) % p[0] - 0.5 * p[0] # type: ignore
    ax.scatter(x_fold1, rv-(map_soln["vrad"][:,1]+map_soln["vrad"][:,2]),label="data") # type: ignore
    inds = np.argsort(x_fold1) #organises the time data in increasing order
    #plotting folded model
    ax.plot(x_fold1[inds], other1[inds]-map_soln["mean_rv"], label="model",zorder=1000,color="C3",lw=2) # type: ignore
    ax.legend()
    ax.set_ylabel('radial velocity [m/s]')
    _ = ax.set_xlabel('Phase')
    fig.savefig('ASTERO+MAP TransitRV.png', facecolor='w')
    # plt.close()

    #plotting the RVs only
    fig_RV, axes = plt.subplots(1,figsize=(16, 10))
    ax_rv = axes
    ax_rv.scatter(t_BKJD, rv, s=100,c='k',zorder=100)
    ax_rv.plot(t_rv, map_soln["vrad_pred"], "--k", label="Planet induced RV",alpha=0.5,linewidth=2.0) # type: ignore # RV induced by planets 
    ax_rv.plot(t_rv, map_soln["bkg_pred"], ":r", label="Background",linewidth=2.0)  # type: ignore # RV induced by background sources
    ax_rv.plot(t_rv, map_soln["rv_model_pred"], label="model",linewidth=2.0) # type: ignore
    ax_rv.legend(fontsize=14, loc='upper right')
    ax_rv.set_xlabel("BKJD [days]",fontsize=14)
    ax_rv.set_ylabel("Radial Velocity [m/s]",fontsize=14)
    ax_rv.set_title("4 Years of Keck/HIRES Data",fontsize=14)
    ax_rv.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for the xlabel
    fig_RV.savefig('MAP_RV.png',dpi=100, facecolor='w',bbox_inches='tight')

    #RV model and plots
    fig, axes = plt.subplots(4,1,figsize=(16, 12),sharey=False, sharex=False)
    ax = axes[0]
    ax.errorbar(t_BKJD, rv, yerr=rv_err, fmt=".k")
    ax.plot(t_rv, map_soln["vrad_pred"], "--k", label="Planet induced RV",alpha=0.5,linewidth=2.0) # type: ignore # RV induced by planets 
    ax.plot(t_rv, map_soln["bkg_pred"], ":r", label="Background",linewidth=2.0)  # type: ignore # RV induced by background sources
    ax.plot(t_rv, map_soln["rv_model_pred"], label="model",linewidth=2.0) # type: ignore
    ax.legend(fontsize=14, loc='lower right')
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for the xlabel
    ax.set_xlabel("BKJD [days]",fontsize=14)
    ax.set_ylabel("Radial Velocity [m/s]",fontsize=14)
    ax.set_title("MAP RV",fontsize=20)

    other1 =  map_soln["vrad"][:,0]# type: ignore
    ax = axes[1]
    # Plot the folded data
    x_fold1= (t_BKJD - t0[0] + 0.5 * p[0]) % p[0] - 0.5 * p[0] # type: ignore
    ax.scatter(x_fold1, rv-(map_soln["vrad"][:,1]+map_soln["vrad"][:,2]),label="data") # type: ignore
    inds = np.argsort(x_fold1) #organises the time data in increasing order
    #plotting folded model
    ax.plot(x_fold1[inds], other1[inds]-map_soln["mean_rv"], label="model",zorder=1000,color="C3",lw=2) # type: ignore
    ax.legend()
    ax.set_ylabel('Radial Velocity [m/s]')

    ax = axes[2]
    other2 =  map_soln["vrad"][:,1]# type: ignore
    # Plot the folded data
    x_fold2= (t_BKJD - t0[1] + 0.5 * p[1]) % p[1] - 0.5 * p[1] # type: ignore
    ax.scatter(x_fold2, rv-(map_soln["vrad"][:,0]+map_soln["vrad"][:,2]),label="data") # type: ignore
    inds = np.argsort(x_fold2) #organises the time data in increasing order
    #plotting folded model
    ax.plot(x_fold2[inds], other2[inds]-map_soln["mean_rv"], label="model",zorder=1000,color="C3",lw=2) # type: ignore
    ax.legend()
    ax.set_ylabel('radial velocity [m/s]')

    ax = axes[3]
    other3 =  map_soln["vrad"][:,2]# type: ignore
    # Plot the folded data
    x_fold3= (t_BKJD - t0[2] + 0.5 * p[2]) % p[2] - 0.5 * p[2] # type: ignore
    ax.scatter(x_fold3, rv-(map_soln["vrad"][:,1]+map_soln["vrad"][:,0]),label="data") # type: ignore
    inds = np.argsort(x_fold3) #organises the time data in increasing order
    #plotting folded model
    ax.plot(x_fold3[inds], other3[inds]-map_soln["mean_rv"], label="model",zorder=1000,color="C3",lw=2) # type: ignore
    ax.legend()
    ax.set_ylabel('radial velocity [m/s]')
    ax.set_xlabel('Phase')
    fig.savefig('ASTERO+MAP RV.png', facecolor='w')
    return 

def GaussianChecking(data_arr,Lower_xlim, Upper_xlim, PlotNumber, figX, figY,mask,extras0,map_soln0,save=True,name=""):
    time_cull, flux_cull, flux_err_cull = data_arr
    if mask is None:
        mask = np.ones(len(time_cull), dtype=bool)
    else:
        mask = mask
    cols = int(PlotNumber/2)
    rows = 2
    p = map_soln0["period"][0]
    t0 = map_soln0["t0"][0]  
    Initial_Limits = np.array([Lower_xlim,Upper_xlim]) # reference point for the first transit epoch
    gp_mod = extras0["gp_pred"] + map_soln0["mean"] 

    fig, axes = plt.subplots(cols,rows, figsize=(figX,figY), sharex=False, sharey=True)
    for index in range(PlotNumber):
        col = index // rows
        row = index % rows
        j = index
        if index == 3: # data is chopped too late for this one, skip it
            j += 1 
        ax = axes[col, row]
        ax.plot(time_cull[mask], flux_cull[mask], ".k", label="data")
        ax.plot(time_cull[mask], gp_mod, color="C2", label="gp model")
        ax.plot(
        time_cull[mask],(map_soln0["light_curves"][:,0])[mask]+1,lw=2,label="Transit Model",color="r" 
        )
        ax.legend(loc='lower right',fontsize=15)
        ax.set_xlim(Initial_Limits[0] + j *p, Initial_Limits[1] + j *p)
        ax.tick_params(axis='both', which='major', labelsize=20)
    fig.text(0.53, 0.04, "Time [BKJD - 2454833]", ha='center', fontsize=25)
    fig.text(-0.015, 0.5, "Relative flux", va='center', rotation='vertical', fontsize=25)

    plt.tight_layout(rect=[0, 0.05, 1, 1]) 
    if save==True:
        fig.savefig(name, facecolor='w',dpi=100,bbox_inches='tight')
    return

def plot_og_data(time, flux, t_BKJD, rv, rv_err):

    fig_data, axes = plt.subplots(1,figsize=(50, 10))

    font = 40
    ax2 = axes.twinx()
    p1, = axes.plot(time,flux, ".k",markersize=20.0,label='Kepler')
    axes.set_ylabel('Normalized Flux',fontsize=font)
    axes.set_xlabel('Time [BKJD - 2454833]',fontsize=font)

    p2, = ax2.plot(t_BKJD, rv,'.b',markersize=20.0,label='Keck/HIRES',zorder=100)
    ax2.errorbar(t_BKJD, rv, yerr=rv_err,fmt='None',elinewidth=3.0)
    ax2.set_ylabel("Radial Velocity [m/s]",fontsize=font)
    axes.tick_params(axis='both', which='major', labelsize=font)
    ax2.tick_params(axis='both', which='major', labelsize=font)

    lines = [p1,p2]
    _ = axes.legend(lines, [l.get_label() for l in lines],fontsize=font)
    fig_data.savefig('Data.png',facecolor='w',dpi=100, bbox_inches='tight')