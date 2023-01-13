# Multifrequency imager on MOJAVE datasets
# Produce results for figures 11-12
# Andrew Chael, October 2022

from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import ehtim as eh
from ehtim.calibrating import self_cal as sc
from ehtim.image import get_specim, blur_mf
import time


plt.close('all')

# data and regularizer terms
ep = 1.e-8

data_term={'cphase':20, 'logcamp':20,'amp':1}
data_term2={'cphase':20, 'logcamp':1,'amp':10}
data_term3={'cphase':20, 'amp':20}

reg_term_mf1 = {'l1':1,'l2_alpha':0,'tv_alpha':.75}
reg_term_mf2 = {'l1':1,'tv':1,'l2_alpha':0,'tv_alpha':.75}
mf_which_solve = (1,1,0)

# which sources to image
sources = ['0212','1730']

for source in sources:
    # output directory
    outdir = './output/' 
    
    #######################################################
    # Load the observations
    #######################################################
    if source=='0212':
        obsX = eh.obsdata.load_uvfits('./mojave_data/0212+735.x.2006_07_07.uvf')
        obsY = eh.obsdata.load_uvfits('./mojave_data/0212+735.y.2006_07_07.uvf')
        obsJ = eh.obsdata.load_uvfits('./mojave_data/0212+735.j.2006_07_07.uvf')

        # re-scale the noise to ensure correct statistics on closure triangles
        # rescaling factors can be obtained from obs.estimate_noise_rescale_factor()
        obsX = obsX.rescale_noise(1)
        obsY = obsY.rescale_noise(1.52)
        
        # apply post-priori amplitude scaling to the 12.1 GHz data
        # Source: Matt Lister, private communication
        obsJ.data['vis'] *= 1.1
        obsJ.data['qvis'] *= 1.1
        obsJ.data['uvis'] *= 1.1        
        obsJ.data['vvis'] *= 1.1      
        obsY = obsJ.rescale_noise(1.1)
            
        # shift the visibilitity phases/the image centroid
        shift =  -10000*eh.RADPERUAS/2.
        obsX.data['vis'] *= np.exp(1j*2*np.pi*shift*obsX.data['u'])
        obsY.data['vis'] *= np.exp(1j*2*np.pi*shift*obsY.data['u'])
        obsJ.data['vis'] *= np.exp(1j*2*np.pi*shift*obsJ.data['u'])


    elif source=='1730':
        obsX = eh.obsdata.load_uvfits('./mojave_data/1730-130.x.2006_07_07.uvf')
        obsY = eh.obsdata.load_uvfits('./mojave_data/1730-130.y.2006_07_07.uvf')
        obsJ = eh.obsdata.load_uvfits('./mojave_data/1730-130.j.2006_07_07.uvf')

        # re-scale the noise to ensure correct statistics on closure triangles
        # rescaling factors can be obtained from obs.estimate_noise_rescale_factor()
        obsX = obsX.rescale_noise(1)
        obsY = obsY.rescale_noise(2.21)
        
        # apply post-priori amplitude scaling to 12.1 GHz data
        # Source: Matt Lister, private communication
        obsJ.data['vis'] *= 1.1
        obsJ.data['qvis'] *= 1.1
        obsJ.data['uvis'] *= 1.1        
        obsJ.data['vvis'] *= 1.1      
        obsJ = obsJ.rescale_noise(1.1)
        
        # shift the visibilitity phases/the image centroid
        shift =  -20000*eh.RADPERUAS/2.
        obsX.data['vis'] *= np.exp(1j*2*np.pi*shift*obsX.data['v'])
        obsY.data['vis'] *= np.exp(1j*2*np.pi*shift*obsY.data['v'])
        obsJ.data['vis'] *= np.exp(1j*2*np.pi*shift*obsJ.data['v'])   
    else:
        raise Exception()
        
    rflist = [obsX.rf, obsY.rf, obsJ.rf]
    labellist = ['x','y','j']
    
    #######################################################
    # flux, resolution and prior
    #######################################################

    # zero baseline flux from short-baseline visibility median
    zblJ = np.median(obsJ.flag_uvdist(uv_min=1.e7,output='flagged').unpack(['amp'])['amp'])
    zblY = np.median(obsY.flag_uvdist(uv_min=1.e7,output='flagged').unpack(['amp'])['amp'])
    zblX = np.median(obsX.flag_uvdist(uv_min=1.e7,output='flagged').unpack(['amp'])['amp'])
    zbllist = [zblX,zblY,zblJ]
    reffreq = obsY.rf
        
    # resolution
    beamparamsX = obsX.fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
    resX = obsX.res() # nominal array resolution, 1/longest baseline
    beamparamsY = obsY.fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
    resY = obsY.res() # nominal array resolution, 1/longest baseline
    beamparamsJ = obsJ.fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
    resJ = obsJ.res() # nominal array resolution, 1/longest baseline
    print("Nominal Resolution: " ,resJ,resY,resX)

    # Construct an intial image
    if source=='0212':
        fov = 25000
        npix = 256 

        flatprior =  eh.image.make_square(obsY, npix, fov*eh.RADPERUAS)
        flatprior.imvec += 1.e-6
        flatprior = flatprior.add_gauss(zblY,(2000*eh.RADPERUAS,2000*eh.RADPERUAS,0,shift,0))
        flatprior = flatprior.add_gauss(.2*zblY,(5000*eh.RADPERUAS,5000*eh.RADPERUAS,0, -shift,0))
        flatprior.imvec *= np.sum(zblY)/np.sum(flatprior.imvec)

        # mask upper and lower bands of the image
        bandsize_pixel = 80
        mask = flatprior.imarr()
        mask[0:bandsize_pixel,:]=0
        mask[npix-bandsize_pixel:npix,:]=0
        flatprior.imvec = mask.flatten()
    elif source=='1730':
        fov = 40000
        npix = 256 

        flatprior =  eh.image.make_square(obsY, npix, fov*eh.RADPERUAS)
        flatprior.imvec += 1.e-6
        flatprior = flatprior.add_gauss(zblY,(10000*eh.RADPERUAS,5000*eh.RADPERUAS,0,0,shift))
        flatprior = flatprior.add_gauss(.2*zblY,(8000*eh.RADPERUAS,8000*eh.RADPERUAS,0,0., 9000*eh.RADPERUAS))
        flatprior.imvec *= np.sum(zblY)/np.sum(flatprior.imvec)

        # mask left and right bands of the image
        bandsize_pixel = 80
        mask = flatprior.imarr()
        mask[:,0:bandsize_pixel]=0
        mask[:,npix-bandsize_pixel:npix]=0
        flatprior.imvec = mask.flatten()
    else:
        raise Exception()
            
    #####################################################################################
    # Image three frequencies together with spectral index
    #####################################################################################
    # determine the unresolved spectral index
    xfit = np.log(np.array(rflist) / reffreq)
    yfit = np.log(np.array(zbllist))
    coeffs = np.polyfit(xfit,yfit,1)
    alpha1 = coeffs[0]
       
    # add the unresolved spectral index to the initial image    
    rprior = flatprior
    #rprior = rprior.add_const_mf(0.,0.)
    rprior = rprior.add_const_mf(alpha1,0.)
    
    # set up the imager
    imgr  = eh.imager.Imager([obsX, obsY, obsJ], rprior, rprior, zblY,
                             data_term = data_term,
                             reg_term=reg_term_mf1,
                             mf_which_solve=mf_which_solve,
                             show_updates=False,norm_reg=True,
                             maxit=100, ttype='nfft',
                             clipfloor=0)
    imgr.regparams['epsilon_tv'] = ep
    
    # blur and reimage
    for i in range(5): 
        imgr.make_image_I(mf=True,show_updates=False)
        out = imgr.out_last()
        imgr.init_next = blur_mf(out, rflist, resX, fit_order=1)
        imgr.maxit_next *= 2
        imgr.maxit_next = np.min((imgr.maxit_next,10000))
        imgr.reg_term_next=reg_term_mf2
        
    # self-calibrate
    out0 = imgr.out_last()
    imX_sc = out0.get_image_mf(obsX.rf)
    imY_sc = out0.get_image_mf(obsY.rf)
    imJ_sc = out0.get_image_mf(obsJ.rf)

    obsX_sc = eh.self_cal.self_cal(obsX, imX_sc, method='both',
                                    processes=8,ttype='nfft',use_grad=True)
    obsY_sc = eh.self_cal.self_cal(obsY, imY_sc, method='both',
                                    processes=8,ttype='nfft',use_grad=True)
    obsJ_sc = eh.self_cal.self_cal(obsJ, imJ_sc, method='both',
                                    processes=8,ttype='nfft',use_grad=True)
    obs_sc_list = [obsX_sc, obsY_sc, obsJ_sc]

            
    # reimage with visibility amplitudes
    rprior2 = out0.blur_circ(resX)
    imgr  = eh.imager.Imager(obs_sc_list, rprior2, rprior2, zblY,
                            data_term=data_term2,
                            reg_term=reg_term_mf2,
                            mf_which_solve=mf_which_solve,
                            show_updates=False,norm_reg=True,
                            maxit=100, ttype='nfft',
                            clipfloor=0)
    imgr.regparams['epsilon_tv'] = ep
    
    for i in range(5): 
       imgr.make_image_I(mf=True,show_updates=False)
       out = imgr.out_last()
       imgr.init_next = blur_mf(out, rflist, resX, fit_order=1)
       imgr.maxit_next *= 2
       imgr.maxit_next = np.min((imgr.maxit_next,10000))

    # self-calibrate
    out1 = imgr.out_last()
    imX_sc = out1.get_image_mf(obsX.rf)
    imY_sc = out1.get_image_mf(obsY.rf)
    imJ_sc = out1.get_image_mf(obsJ.rf)

    obsX_sc = eh.self_cal.self_cal(obsX, imX_sc, method='both',
                                    processes=8,ttype='nfft',use_grad=True)
    obsY_sc = eh.self_cal.self_cal(obsY, imY_sc, method='both',
                                    processes=8,ttype='nfft',use_grad=True)
    obsJ_sc = eh.self_cal.self_cal(obsJ, imJ_sc, method='both',
                                    processes=8,ttype='nfft',use_grad=True)
    obs_sc_list = [obsX_sc, obsY_sc, obsJ_sc]

            
    # reimage with complex visibilities
    rprior3 = out1.blur_circ(resX)
    imgr  = eh.imager.Imager(obs_sc_list, rprior3, rprior3, zblY,
                            data_term=data_term3,
                            reg_term=reg_term_mf2,
                            mf_which_solve=mf_which_solve,
                            show_updates=False,norm_reg=True,
                            maxit=100, ttype='nfft',
                            clipfloor=0)
    imgr.regparams['epsilon_tv'] = ep
    
    for i in range(10): # blur and reimage
       print(imgr.maxit_next)
       imgr.make_image_I(mf=True,show_updates=False)
       out = imgr.out_last()
       imgr.init_next = blur_mf(out, rflist, resX, fit_order=1)
       imgr.maxit_next *= 2
       imgr.maxit_next = np.min((imgr.maxit_next,10000))
       
    # save final images
    out = imgr.out_last()    
    for jj in range(len(obs_sc_list)):
        out.get_image_mf(obs_sc_list[jj].rf).save_fits(outdir + source + '_' + labellist[jj] + '_mf.fits')

    # save final self-calibrated data
    for jj in range(len(obs_sc_list)):
        obs_sc_list[jj].save_uvfits(outdir + source + '_' + labellist[jj] + '_mf_selfcal.uvfits')
            
    plt.close('all')
