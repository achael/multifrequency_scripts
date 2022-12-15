# Multifrequency imager on HL Tau data
# 4 spws at band 6, band 7

from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import ehtim as eh
from ehtim.calibrating import self_cal as sc
from ehtim.image import get_specim, blur_mf


plt.close('all')

# image parameters
fov = 2.5*eh.RADPERAS
npix = 400
diskrad = 1.2*eh.RADPERAS
core_fwhm =  1*eh.RADPERAS
ep = 1.e-6

# data and regularizer terms
data_term_1 = {'vis':1}
data_term_2 = {'vis':1}
data_term_3 = {'vis':1}

reg_term_1 = {'simple':1,'tv_alpha':1,'l2_alpha':1,'flux':100}
reg_term_2 = {'simple':1,'tv':1,'tv_alpha':.5,'l2_alpha':1,'flux':100}
reg_term_3 = {'simple':1,'tv':1,'tv_alpha':.5,'l2_alpha':1,'flux':100}

# output directory
outdir = './output/'

#######################################################
# Load the observations
#######################################################
obs1 = eh.obsdata.load_uvfits("./ALMA_data/Band6/HLTau_Band6_spw01_polswap_avg.uvfits")
obs2 = eh.obsdata.load_uvfits("./ALMA_data/Band6/HLTau_Band6_spw02_polswap_avg.uvfits")
obs3 = eh.obsdata.load_uvfits("./ALMA_data/Band6/HLTau_Band6_spw03_polswap_avg.uvfits")
obs4 = eh.obsdata.load_uvfits("./ALMA_data/Band6/HLTau_Band6_spw04_polswap_avg.uvfits")

obs5 = eh.obsdata.load_uvfits("./ALMA_data/Band7/HLTau_Band7_spw01_polswap_avg.uvfits")
obs6 = eh.obsdata.load_uvfits("./ALMA_data/Band7/HLTau_Band7_spw02_polswap_avg.uvfits")
obs7 = eh.obsdata.load_uvfits("./ALMA_data/Band7/HLTau_Band7_spw03_polswap_avg.uvfits")
obs8 = eh.obsdata.load_uvfits("./ALMA_data/Band7/HLTau_Band7_spw04_polswap_avg.uvfits")

obslist_orig = [obs1,obs2,obs3,obs4,obs5,obs6,obs7,obs8]
obslist = [obs.copy() for obs in obslist_orig]
rflist = [obs.rf for obs in obslist]
reffreq = 287250600872.5

# re-scale the noise to ensure correct statistics on closure triangles
# rescaling factors can be obtained from obs.estimate_noise_rescale_factor()
# but this takes a very long time on these large datasets
noisefacs = [25.,25.,25.,25.,25.,25.,25.,25.] 
for i,obs in enumerate(obslist):
    noise_scale_factor = noisefacs[i]
    for d in obs.data:
        d[-4] = d[-4] * noise_scale_factor
        d[-3] = d[-3] * noise_scale_factor
        d[-2] = d[-2] * noise_scale_factor
        d[-1] = d[-1] * noise_scale_factor

# Now check the noise statistics on all closure phase triangles
check_statistics=False
max_diff = 1000.0 #seconds
if check_statistics:
    for obs in obslist[-1:]:
        c_phases = obs.c_phases(vtype='vis', mode='time', count='min', ang_unit='')
        all_triangles = []
        for scan in c_phases:
            for cphase in scan:
                all_triangles.append((cphase[1],cphase[2],cphase[3]))

        s_list_all = []
        for tri in set(all_triangles):
            all_tri = np.array([[]])
            for scan in c_phases:
                for cphase in scan:
                    if cphase[1] == tri[0] and cphase[2] == tri[1] and cphase[3] == tri[2]:
                        all_tri = np.append(all_tri, ((cphase[0], cphase[-2], cphase[-1])))
            if len(all_tri)<3: continue
            
            all_tri = all_tri.reshape(int(len(all_tri)/3),3)

            # Now go through and find studentized differences of adjacent points
            s_list = np.array([])
            for j in range(len(all_tri)-1):
                if (all_tri[j+1,0]-all_tri[j,0])*3600.0 < max_diff:
                    diff = (all_tri[j+1,1]-all_tri[j,1]) % (2.0*np.pi)
                    if diff > np.pi: diff -= 2.0*np.pi
                    s = diff/(all_tri[j,2]**2 + all_tri[j+1,2]**2)**0.5
                    s_list = np.append(s_list, s)
                    s_list_all = np.append(s_list_all,s)    
            if len(s_list) > 20:
                print(tri,np.std(s_list))
        print('***ALL TRIANGLES***', np.std(s_list_all))        


#######################################################
# flux, resolution and prior
#######################################################

# total flux data from ALMA 2015 paper table 1
rflist_almatable = np.array([233.0e9,287.2e9,343.5e9]) 
zbllist_almatable = np.array([0.7441,1.4415,2.1408])
weights = 1./np.array([1.5e-3,1.8e-3,3.7e-3])

xfit = np.log(np.array(rflist_almatable) / reffreq)
yfit = np.log(np.array(zbllist_almatable))
coeffs = np.polyfit(xfit,yfit,1)
specalma = coeffs[0]
zblalma = np.exp(coeffs[1])

def zbl_almapaper(rf):
    return zblalma * ((rf/reffreq)**specalma)
            
zbls = []
for obs in obslist:
    zbl_data = np.median(obs.flag_uvdist(uv_min=20000,output='flagged').unpack(['amp'])['amp'])
    zbl_alma = zbl_almapaper(obs.rf)
    zbl = zbl_alma
        
    scalefactor = zbl/zbl_data
    print(obs.rf,zbl_data,zbl,np.abs(1-scalefactor))
    obs.data['vis'] *= scalefactor
    obs.data['sigma'] *= scalefactor
    zbls.append(zbl)


# Resolution
beamparams = obslist[0].fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
res = obslist[0].res() # nominal array resolution, 1/longest baseline

# Construct an initial image
emptyprior = eh.image.make_square(obs, npix, fov)
gaussprior = emptyprior.add_gauss(.8*zbl, (core_fwhm, core_fwhm, 0, 0, 0))
gaussprior = gaussprior.add_flat(.2*zbl)
alpha0 = 1.

gaussprior = gaussprior.add_const_mf(alpha0,0)
gaussprior.rf = reffreq

#####################################################################################
# Image all frequencies together  
#####################################################################################

# Round 1
# Set up the imager
imgr  = eh.imager.Imager(obslist, gaussprior, gaussprior, zbl, norm_reg=True,transform='log',
                         data_term=data_term_1,                      
                         reg_term=reg_term_1,
                         maxit=500, ttype='nfft',
                         mf_which_solve=(1,1,0))

# blur and reimage
for i in range(2): 
    imgr.make_image_I(show_updates=False,mf=True)
    out = imgr.out_last()
    imgr.init_next = blur_mf(out, rflist, 4*res,fit_order=1)
    imgr.prior_next = imgr.init_next    

# shift the output image to the center
out0 = out.copy()
refim = emptyprior.add_gauss(zbl, (.1*core_fwhm, .1*core_fwhm, 0, 0, 0))
shift = np.array(refim.align_images([out0])[1][0]) 
outlist = np.array([out0.get_image_mf(rf) for rf in rflist])
outlist = refim.align_images(outlist,shift=[shift for rf in rflist])[0]
out1 = get_specim(outlist, reffreq, fit_order=1)

# self-calibrate
obs_sc_list = []
for kk,obs in enumerate(obslist):
    scim = out1.get_image_mf(obs.rf)
    scim.imvec *= zbls[kk] / scim.total_flux()
    obs_sc = eh.self_cal.self_cal(obs,scim,method='both',
                                  processes=10,ttype='nfft',use_grad=True)
    obs_sc_list.append(obs_sc)
    
# Round 2
# Set up the imager again with the self-calibrated data and initial image
imgr  = eh.imager.Imager(obs_sc_list, gaussprior, gaussprior, zbl, norm_reg=True,transform='log',
                         data_term=data_term_1,                      
                         reg_term=reg_term_1,
                         maxit=1000, ttype='nfft',
                         mf_which_solve=(1,1,0))
                         
# blur and reimage                         
for i in range(3): 
    imgr.make_image_I(show_updates=False,mf=True)
    out = imgr.out_last()
    imgr.init_next = blur_mf(out, rflist, 2*res,fit_order=1)
    imgr.prior_next = imgr.init_next    
    imgr.reg_term_next = reg_term_2

# self-calibrate
out2= out.copy()
obs_sc_list = []
for kk,obs in enumerate(obslist):
    scim = out2.get_image_mf(obs.rf)
    scim.imvec *= zbls[kk] / scim.total_flux()
    obs_sc = eh.self_cal.self_cal(obs,scim,method='both',
                                  processes=10,ttype='nfft',use_grad=True)
    obs_sc_list.append(obs_sc)
    
# Round 3  
# Set up the imager again with the self-calibrated data and blurred final image
out2B = blur_mf(out2, rflist, 2*res,fit_order=1)
out2B.specvec = np.ones(out2B.specvec.shape)*alpha0
imgr  = eh.imager.Imager(obs_sc_list, out2B, out2B, zbl, norm_reg=True,transform='log',
                         data_term=data_term_3,                      
                         reg_term=reg_term_3,
                         maxit=5000, ttype='nfft',
                         mf_which_solve=(1,1,0))
        
for i in range(2): 
    imgr.make_image_I(show_updates=False,mf=True)
    out = imgr.out_last()
    imgr.init_next = blur_mf(out, rflist, 2*res,fit_order=1)
    imgr.prior_next = imgr.init_next

# self-calibrate
out3= out.copy()
obs_sc_list = []
for kk,obs in enumerate(obslist):
    scim = out3.get_image_mf(obs.rf)
    scim.imvec *= zbls[kk] / scim.total_flux()
    obs_sc = eh.self_cal.self_cal(obs,scim,method='both',
                                  processes=10,ttype='nfft',use_grad=True)
    obs_sc_list.append(obs_sc)
    
# Round 4
# Set up the imager again with the self-calibrated data and blurred final image
out3B = blur_mf(out3, rflist, 2*res,fit_order=1)
out3B.specvec = np.ones(out2B.specvec.shape)*alpha0
imgr  = eh.imager.Imager(obs_sc_list, out3B, out3B, zbl, norm_reg=True,transform='log',
                         data_term=data_term_3,                      
                         reg_term=reg_term_3,
                         maxit=10000, ttype='nfft',
                         mf_which_solve=(1,1,0))
        
for i in range(2): 
    imgr.make_image_I(show_updates=False,mf=True)
    out = imgr.out_last()
    imgr.init_next = blur_mf(out, rflist, 2*res,fit_order=1)
    imgr.prior_next = imgr.init_next

# Save final images
for rf in rflist:            
    out.get_image_mf(rf).save_fits(outdir + 'hltau_%0.1f_mf.fits'%(rf/1.e9))

# Save final self-calibrated data
for kk, obs in enumerate(obs_sc_list):
    obs.save_uvfits(outdir + 'hltau_%0.1f_mf_selfcal.uvfits'%(rflist[kk]/1.e9))
 

