# multifrequency imaging scripts for Chael+ 2022
This repository hosts the RML image reconstruction scripts used to produce the images in the paper ``Multi-frequency black hole imaging with the next-generation Event Horizon Telescope.''

These scripts require installation of [eht-imaging](https://github.com/achael/eht-imaging) version 1.2.5, as well as some other python libraries. 

If you use want to use the included simulation images or datasets in your work, please send me an email at achael@princeton.edu. 

The scripts are in the following sub-directories corresponding to sections in the paper: 
### sec_4.2
Single- and Multi-frequency imaging of simulated ngEHT datasets from 86,230,345 GHz. These scripts were used to reconstruct images from the included datasets for figures 3-6.  The simulation models are from [Chael+ 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.486.2873C/abstract) and [Mizuno+ 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.506..741M/abstract).

### sec_4.3
Single- and Multi-frequency imaging of simulated ngEHT datasets from 213,215,227,229 GHz data. These scripts were used to reconstruct images from the included datasets for figures 3-6. These simulation models are from [Ricarte+ 2022](https://ui.adsabs.harvard.edu/abs/2022arXiv220202408R/abstract).

### sec_5.1
Multi-frequency imaging of MOJAVE project data of [S5 0212+73](https://www.cv.nrao.edu/MOJAVE/sourcepages/0212+735.shtm) and [NRAO 530](https://www.cv.nrao.edu/MOJAVE/sourcepages/1730-130.shtml) from VLBA observations at 8.1, 8.4, and 12.1 GHz.  These scripts were used to reconstruct images from the included datasets for figures 11-12. The included CLEAN reconstructions are from [Hovatta+ 2014](https://ui.adsabs.harvard.edu/abs/2014AJ....147..143H/abstract) and [Lister+ 2018](https://ui.adsabs.harvard.edu/abs/2018ApJS..234...12L). 

### sec_5.2 
Multi-frequency imaging of [ALMA science verification data](https://almascience.nrao.edu/alma-data/science-verification/science-verification-data) of HL Tau from ALMA bands 6 and 7. These scripts were used to reconstruct images from the included datasets for figure 13. The data and the CLEAN images were published in [ALMA Partnership+ 2015](https://ui.adsabs.harvard.edu/abs/2015ApJ...808L...3A/abstract).  The data and CLEAN images are publically available [here](https://almascience.nrao.edu/almadata/sciver/HLTauBand7/).
