# Multifrequency imager for M87 simulation images
# Observations at 213,215,227,229 GHz
# Image a single spectral index map over the full range
# Produce results for figures 9-10
# Andrew Chael, October 2022

from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import ehtim as eh
import sys
from ehtim.calibrating import self_cal as sc
from ehtim.image import get_specim, blur_mf
import time

plt.close('all')

# image parameters
fov = 200*eh.RADPERUAS
npix = 200
prior_fwhm = 60*eh.RADPERUAS
ep = 1.e-6

# data and regularizer terms    
data_term = {'cphase':20, 'logcamp':20}
data_term_2 = {'amp':20, 'cphase':20, 'logcamp':5}

reg_term_0 = {'l1':1, 'simple':0.1,'flux':10}
reg_term_1 = {'l1':1,'simple':0.1, 'tv':2,'flux':10}
                
reg_term_mf0 = {'l1':1, 'simple':.1, 'flux':10,
                'l2_alpha':5.,'tv_alpha':5.}
reg_term_mf1 = {'l1':1,'simple':.1, 'tv':2, 'flux':10,
                'l2_alpha':5.,'tv_alpha':5.}
                              
# which models to image
models = ['MAD','SANE']

# which imaging strategy to run
image_nomf = True
image_mf = True

for model in models:
    # output directory
    outdir = './output_' + model + '/' 
    
    #######################################################
    # Load the synthetic observations
    #######################################################
    print("loading data from uvfits")

    obs213 = eh.obsdata.load_uvfits('./data_' + model + '/uvfits/datafile_213GHz.uvfits')
    obs215 = eh.obsdata.load_uvfits('./data_' + model + '/uvfits/datafile_215GHz.uvfits')
    obs227 = eh.obsdata.load_uvfits('./data_' + model + '/uvfits/datafile_227GHz.uvfits')
    obs229 = eh.obsdata.load_uvfits('./data_' + model + '/uvfits/datafile_229GHz.uvfits')
        
    rflist = [obs213.rf,obs215.rf,obs227.rf,obs229.rf]
    labellist = ['213','215','227','229']
    reffreq = obs227.rf 
        
    #######################################################
    # flux, resolution and prior
    #######################################################
    
    # zero baseline flux (assumed from unresolved observations)  
    if model == 'MAD':
        zbl213 = 0.546
        zbl215 = 0.542
        zbl227 = 0.518
        zbl229 = 0.515
        
    elif model=='SANE':
        zbl213 = 0.538
        zbl215 = 0.533
        zbl227 = 0.505
        zbl229 = 0.501    
    zbllist = [zbl213, zbl215, zbl227, zbl229]
    refzbl = zbl227                                                              
    
    # Resolution
    beamparams = obs227.fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
    res = obs227.res() # nominal array resolution, 1/longest baseline
    print("Nominal Resolution (227 GHz): " ,res)

    # Construct an initial image
    emptyprior = eh.image.make_square(obs227, npix, fov)
    emptyprior.rf = reffreq
    gaussprior = emptyprior.add_gauss(refzbl, (prior_fwhm, prior_fwhm,0))
    gaussprior = gaussprior.blur_circ(res)

    #####################################################################################
    # Image four frequencies independently 
    #####################################################################################
    if image_nomf:
        for kk,obs in enumerate([obs213,obs215,obs227,obs229]):
            zbl = zbllist[kk]
            label = labellist[kk]          
            rprior = gaussprior
            rprior.rf = rflist[kk]
            rprior.imvec *= zbl / rprior.total_flux()
                 
            # set up the imager with closure phases and closure amplitudes
            imgr  = eh.imager.Imager(obs, rprior, rprior, zbl,
                                     data_term=data_term,
                                     reg_term=reg_term_0,
                                     show_updates=False,norm_reg=True,
                                     maxit=200, ttype='nfft')
            imgr.regparams['epsilon_tv'] = ep
            
            # blur and reimage
            for i in range(3): 
                imgr.make_image_I(mf=False,show_updates=False)
                out = imgr.out_last()
                imgr.reg_term = reg_term_1
                init_next = out.blur_circ(res)
                imgr.init_next = init_next
                imgr.maxit_next *=2

            # self-calibrate visibility amplitudes
            im_selfcal = out
            im_selfcal.imvec *= zbllist[kk]/im_selfcal.total_flux() # fix the total flux
            obs_sc = eh.self_cal.self_cal(obs,im_selfcal,method='both',
                                          processes=10,ttype='nfft',use_grad=True)
                                          
            # save self-calibrated dataset
            obs_sc.save_uvfits(outdir + model+'_'+label+'_nomf_selfcal.uvfits')
                
            # reimage with visibility amplitudes
            imgr  = eh.imager.Imager(obs_sc, rprior, rprior, zbl,
                                     data_term=data_term_2,
                                     reg_term=reg_term_0,
                                     show_updates=False,norm_reg=True,
                                     maxit=200, ttype='nfft')
            imgr.regparams['epsilon_tv'] = ep
            
            for i in range(5): 
                imgr.make_image_I(mf=False,show_updates=False)
                out = imgr.out_last()
                imgr.reg_term = reg_term_1
                init_next = out.blur_circ(res)
                imgr.init_next = init_next
                imgr.maxit_next *= 2    

            # save results
            out.save_fits(outdir+model+'_'+label+'_nomf.fits')            

    #####################################################################################
    # Image four frequencies together with spectral index 
    #####################################################################################

    if image_mf:

        # determine the unresolved spectral index
        xfit = np.log(np.array(rflist) / reffreq)
        yfit = np.log(np.array(zbllist))
        coeffs = np.polyfit(xfit,yfit,1)
        alpha1 = coeffs[0]
        
        # add the unresolved spectral index to the initial image
        rprior = gaussprior
        rprior = rprior.add_const_mf(alpha1,0)

        # set up the imager
        imgr  = eh.imager.Imager([obs213, obs215,obs227, obs229], rprior, rprior, refzbl,
                                 data_term=data_term,
                                 reg_term=reg_term_mf0,
                                 mf_which_solve=(1,1,0),
                                 show_updates=False,norm_reg=True,
                                 maxit=200, ttype='nfft')
        imgr.regparams['epsilon_tv'] = ep
            
        # blur and reimage
        for i in range(3): 
            imgr.make_image_I(mf=True,show_updates=False)
            out = imgr.out_last()
            imgr.reg_term = reg_term_mf1
            init_next = blur_mf(out, rflist, res, fit_order=1)
            imgr.init_next = init_next
            imgr.maxit_next *=2

        # self-calibrate visibility amplitudes
        obs_sc_list = []
        for kk,obs in enumerate([obs213, obs215,obs227, obs229]):
        
            im_selfcal = out.get_image_mf(obs.rf)
            im_selfcal.imvec *= zbllist[kk]/im_selfcal.total_flux() # fix the total flux
            obs_sc = eh.self_cal.self_cal(obs,im_selfcal,method='both',
                                          processes=10,ttype='nfft',use_grad=True)
            obs_sc_list.append(obs_sc)
            
            # save self-calibrated dataset
            obs_sc.save_uvfits(outdir+model+'_'+labellist[kk]+'_nomf_selfcal.uvfits')
            
        # reimage with visibiilty amplitudes
        imgr  = eh.imager.Imager(obs_sc_list, rprior, rprior, refzbl,
                                 data_term=data_term_2,
                                 reg_term=reg_term_mf0,
                                 mf_which_solve=(1,1,0),
                                 show_updates=False,norm_reg=True,
                                 maxit=200, ttype='nfft')
        imgr.regparams['epsilon_tv'] = ep
                                             
        for i in range(5): # blur and reimage
            imgr.make_image_I(mf=True,show_updates=False)
            out = imgr.out_last()
            imgr.reg_term = reg_term_mf1
            init_next = blur_mf(out, rflist, res, fit_order=1)
            imgr.init_next = init_next
            imgr.maxit_next *=2
                                                     
        # save results
        out.get_image_mf(obs213.rf).save_fits(outdir+model+'_213.fits')
        out.get_image_mf(obs215.rf).save_fits(outdir+model+'_215.fits')
        out.get_image_mf(obs227.rf).save_fits(outdir+model+'_227.fits')
        out.get_image_mf(obs229.rf).save_fits(outdir+model+'_229.fits')

