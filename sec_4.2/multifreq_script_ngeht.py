# Multifrequency imager for M87 simulation images
# Observations at 86, 230, 345 GHz
# Produce results for figures 3-6
# Andrew Chael, October 2022

from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import ehtim as eh
import sys
from ehtim.calibrating import self_cal as sc
from ehtim.image import get_specim, blur_mf

plt.close('all')

# image parameters
fov86  = 700 # 600
fov230 = 700 # 600
fov345 = 700 # 600
npix   = 200 # 200

prior_fwhm_uas = 45*eh.RADPERUAS
disk_rad_uas = 100*eh.RADPERUAS

# imager parameters
ep = 1.e-6
STOP=1.e-8
maxit = 5000
nloops = 3

data_term = {'logcamp':20, 'cphase':20}  
data_term_2 = {'amp':20, 'cphase':20}      

reg_term0 = {'l1':1,'flux':10}
reg_term1 = {'l1':1,'tv':1.,'flux':10}

reg_term_mf = {'l1':1,'tv':1,'flux':10,
               'l2_alpha':20,'tv_alpha':20,
               'l2_beta':30,'tv_beta':30}

# which models to image
models = ['Chael'] #['Chael','Mizuno']

# which imaging strategy to run
image_nomf = False
image_mf = True
    
for model in models:
    # output directory
    outdir = './output_' + model + '/' 

    #######################################################
    # Load the synthetic observations
    #######################################################
    print("loading data from uvfits")
    if model=='Chael':
        obs86 = eh.obsdata.load_uvfits('./data_M87_Chael/uvfits/datafile_86GHz.uvfits')
        obs230 = eh.obsdata.load_uvfits('./data_M87_Chael/uvfits/datafile_230GHz.uvfits')
        obs345 = eh.obsdata.load_uvfits('./data_M87_Chael/uvfits/datafile_345GHz.uvfits')
        
    elif model=='Mizuno':
        obs86 = eh.obsdata.load_uvfits('./data_M87_Mizuno/uvfits/datafile_86GHz.uvfits')
        obs230 = eh.obsdata.load_uvfits('./data_M87_Mizuno/uvfits/datafile_230GHz.uvfits')
        obs345 = eh.obsdata.load_uvfits('./data_M87_Mizuno/uvfits/datafile_345GHz.uvfits')
        
    #######################################################
    # flux, resolution and prior
    #######################################################

    # zero baseline flux (assumed from unresolved observations)      
    if model=='Chael':
        zbl86 = 2.53
        zbl230 = 1.98
        zbl345 = 1.54
        
    elif model=='Mizuno':
        zbl86 = 1.23
        zbl230 = 1.17
        zbl345 = 0.99
    else:
        raise Exception()
        
 
    # Resolution
    beamparams86 = obs86.fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) 
    res86 = obs86.res() # nominal array resolution, 1/longest baseline
    beamparams230 = obs230.fit_beam() 
    res230 = obs230.res() 
    beamparams345 = obs345.fit_beam() 
    res345 = obs345.res() 
    
    print("Nominal Resolution: " ,res345,res230,res86)

    # Construct initial geometric images
    emptyprior86 = eh.image.make_square(obs86, npix, fov86*eh.RADPERUAS)
    gaussprior86 = emptyprior86.add_gauss(.6*zbl86, (prior_fwhm_uas, prior_fwhm_uas, 0, 100*eh.RADPERUAS, 0))    
    tophatprior86 = emptyprior86.add_gauss(.4*zbl86, (3*disk_rad_uas,3*disk_rad_uas,0,-80*eh.RADPERUAS,0))
    prior86 = gaussprior86.copy()
    prior86.imvec += tophatprior86.imvec
    prior86 = prior86.blur_circ(30*eh.RADPERUAS)
    
    emptyprior230 = eh.image.make_square(obs230, npix, fov230*eh.RADPERUAS)
    gaussprior230 = emptyprior230.add_gauss(.8*zbl230, (prior_fwhm_uas, prior_fwhm_uas, 0, 100*eh.RADPERUAS, 0))    
    tophatprior230 = emptyprior230.add_tophat(.2*zbl230, 2*disk_rad_uas).shift([0,int(80*eh.RADPERUAS/emptyprior230.psize)])
    prior230 = gaussprior230.copy()
    prior230.imvec += tophatprior230.imvec
    prior230 = prior230.blur_circ(30*eh.RADPERUAS)
    
    emptyprior345 = eh.image.make_square(obs345, npix, fov345*eh.RADPERUAS)
    gaussprior345 = emptyprior345.add_gauss(.9*zbl345, (prior_fwhm_uas, prior_fwhm_uas, 0, 100*eh.RADPERUAS, 0))    
    tophatprior345 = emptyprior345.add_tophat(.1*zbl345, disk_rad_uas).shift([0,int(0*eh.RADPERUAS/emptyprior345.psize)])
    prior345 = gaussprior345.copy()
    prior345.imvec += tophatprior345.imvec
    prior345 = prior345.blur_circ(30*eh.RADPERUAS)
    
    ###################################################################################
    # final prep before imaging
    #####################################################################################
    
    # flag zero baselines
    obs86 = obs86.flag_uvdist(uv_min=2.e8,output='kept')
    obs230 = obs230.flag_uvdist(uv_min=2.e8,output='kept')
    obs345 = obs345.flag_uvdist(uv_min=2.e8,output='kept')

    # useful data
    obslist = [obs86,obs230,obs345]
    rflist = [obs86.rf, obs230.rf, obs345.rf]
    zbllist = [zbl86, zbl230, zbl345]
    labellist = ['86','230','345']
            
    #####################################################################################
    ## Image all frequencies independently
    #####################################################################################

    if image_nomf:
       for kk,obs in enumerate(obslist):
            print("\nImaging %0.2f GHz"%obs.rf)
            zbl = zbllist[kk]
            label = labellist[kk]
            rf = rflist[kk]
                   
            # initial image        
            if kk==0: 
                rprior = prior86
            elif kk==1:
                rprior = prior230
            elif kk==2:    
                rprior = prior345                        
            else:
                raise Exception()
                
            rprior.rf = rf
                        
            #set up and run imager
            imgr  = eh.imager.Imager(obs, rprior, rprior, zbl, norm_reg=True,transform='log',
                                     data_term=data_term,
                                     reg_term=reg_term0,
                                     maxit=250, ttype='nfft',stop=STOP)
            imgr.regparams['epsilon_tv'] = ep
            imgr.make_image_I(show_updates=False)
            out = imgr.out_last()
            
            # blur and reimage
            for i in range(nloops): 
                imgr.reg_term_next = reg_term1
                imgr.maxit_next=maxit
                imgr.init_next = out.blur_circ(40*eh.RADPERUAS)    
                imgr.make_image_I(show_updates=False)
                out = imgr.out_last()
            
            # self calibrate
            im_selfcal = out.copy()
            im_selfcal.imvec *= zbl/im_selfcal.total_flux()            
            obs_sc = eh.self_cal.self_cal(obs,im_selfcal,method='both',
                                          processes=10,ttype='nfft',use_grad=True)
            obs_sc = eh.self_cal.self_cal(obs_sc,im_selfcal,method='both',
                                          processes=10,ttype='nfft',use_grad=True)

            # save self-calibrated dataset
            obs_sc.save_uvfits(outdir + model+'_ngeht_'+label+'_nomf_selfcal.uvfits')
                        
            # reimage with calibrated amplitudes
            rprior = im_selfcal.blur_circ(50*eh.RADPERUAS)
            imgr  = eh.imager.Imager(obs_sc, rprior, rprior, zbl,
                                     norm_reg=True,transform='log',
                                     data_term=data_term_2,
                                     reg_term=reg_term0,
                                     maxit=250, ttype='nfft',stop=STOP)
            imgr.regparams['epsilon_tv'] = ep                    
            imgr.make_image_I(show_updates=False)
            out = imgr.out_last()
            
            # blur and reimage
            for i in range(nloops): 
                imgr.reg_term_next = reg_term1
                imgr.maxit_next=maxit
                imgr.init_next = out.blur_circ(40*eh.RADPERUAS) 
                imgr.make_image_I(show_updates=False)
                out = imgr.out_last()
              
            # save results                
            out.save_fits(outdir + model +'_ngeht_'+label+'_nomf.fits')
            
    #####################################################################################
    # Image all frequencies together with spectral index
    #####################################################################################

    if image_mf:
        print("Multifrequency Imaging")
        
        # reference frequency
        reffreq = obs230.rf; refzbl = zbl230
        
        # get unresolved spectral index
        xfit = np.log(np.array([obs86.rf,obs230.rf,obs345.rf]) / reffreq)
        yfit = np.log(np.array([zbl86,zbl230,zbl345]))
        coeffs = np.polyfit(xfit,yfit,2)
        alpha0 = coeffs[1]
        beta0 = 0
        
        # Start with the single-frequency 86 GHZ image with a constant spectral index
        out86 = eh.image.load_fits(outdir + model +'_ngeht_86_nomf.fits')   
        rprior = out86.blur_circ(50*eh.RADPERUAS)        
        rprior.imvec *= refzbl/rprior.total_flux()
        rprior.rf = reffreq
        rprior = rprior.add_const_mf(alpha0,beta0)

        #set up and run imager
        imgr  = eh.imager.Imager([obs86,obs230,obs345], rprior, rprior, refzbl,
                                 data_term=data_term,
                                 reg_term=reg_term_mf,
                                 mf_which_solve=(1,1,1),
                                 show_updates=False,norm_reg=True,
                                 maxit=250, ttype='nfft',stop=STOP)    
        imgr.regparams['epsilon_tv'] = ep                                         
        imgr.make_image_I(mf=True,show_updates=False)
        out = imgr.out_last()     
        
        # blur and reimage                                           
        for i in range(nloops): 
            imgr.init_next = blur_mf(out, rflist, 40*eh.RADPERUAS, fit_order=2)
            imgr.maxit_next = maxit 
            imgr.make_image_I(mf=True,show_updates=False)
            out = imgr.out_last()
                      
        # self-calibrate
        obs_sc_list = []
        for kk,obs in enumerate([obs86,obs230,obs345]):
            zbl = zbllist[kk]
            label = labellist[kk]
            im_selfcal = out.get_image_mf(obs.rf)
            im_selfcal.imvec *= zbl/im_selfcal.total_flux()
            obs_sc = eh.self_cal.self_cal(obs,im_selfcal,method='both',
                                          processes=10,ttype='nfft',use_grad=True)
            obs_sc = eh.self_cal.self_cal(obs_sc,im_selfcal,method='both',
                                          processes=10,ttype='nfft',use_grad=True)
            obs_sc_list.append(obs_sc)
            
            # save self-calibrated dataset
            obs_sc.save_uvfits(outdir + model+'_ngeht_'+label+'_mf_selfcal.uvfits')
                     
        # reimage with calibrated amplitudes
        rprior = out.get_image_mf(reffreq).blur_circ(50*eh.RADPERUAS)
        rprior = rprior.add_const_mf(alpha0,beta0)
        imgr  = eh.imager.Imager(obs_sc_list, rprior, rprior, refzbl,
                                 data_term=data_term_2,
                                 reg_term=reg_term_mf,
                                 mf_which_solve=(1,1,1),
                                 show_updates=False,norm_reg=True,
                                 maxit=250, ttype='nfft',stop=STOP)                                        
        imgr.regparams['epsilon_tv'] = ep                                                                       
        imgr.make_image_I(mf=True,show_updates=False)
        out = imgr.out_last()                                  
        
        # blur and reimage
        for i in range(nloops): 
            imgr.init_next = blur_mf(out, rflist, 40*eh.RADPERUAS, fit_order=2)
            imgr.maxit_next = maxit 
            imgr.make_image_I(mf=True,show_updates=False)
            out = imgr.out_last()
                                   
        # save final results
        out.get_image_mf(obs86.rf).save_fits(outdir+model+'_ngeht_86_mf.fits')
        out.get_image_mf(obs230.rf).save_fits(outdir+model+'_ngeht_230_mf.fits')
        out.get_image_mf(obs345.rf).save_fits(outdir+model+'_ngeht_345_mf.fits')

plt.close('all')

