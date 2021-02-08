# This file contains the functions and classes used for simsurvey.

import optparse

import simsurvey
import os
import numpy as np
import sncosmo
import pandas

import pickle
#import dill
import glob

from scipy.interpolate import RectBivariateSpline as Spline2d
from scipy.interpolate import interpolate as interp

from astropy.cosmology import Planck15

import time


# Writte the mejs and phis and a list or an array. If we are interested on a single value, we should write it as a list
#mejs = np.arange(0.01,0.101,0.01)
#phis = np.arange(15+15,75+1,15)
mejs = [0.06]
phis = [60]

__version__ = 1.0

def parse_commandline():
    """@Parse the options given on the command-line.
    """
    parser = optparse.OptionParser(usage=__doc__,version=__version__)
    #parser.add_option("-p", "--possisDir", help="possis directory",default="data/possis/")
    parser.add_option("--surveyDir", help="Observing logs directory",default="data/")
    parser.add_option("--dustDir", help="Dust map directory",default="data/sfddata-master/")
    parser.add_option("--outputDir", help="Output directory",default="output/")
    parser.add_option("--inputDir", help="Input directory",default="input/")
    parser.add_option("-m", "--modelType", help="(Bulla, DES)",default="Bulla")
    parser.add_option("-t", "--theta", help="Fixed theta", default=20, type=float)
    parser.add_option("-f", "--fixedtheta", action="store_true", default=False)
    parser.add_option("--showplot", action="store_true", default=False)
    parser.add_option("--mej", help="mej", default=0.04, type=float)
    parser.add_option("--opening_angle", help="phi", default=30, type=float)
    parser.add_option("--temp", help="temp", default=5000, type=float)
    parser.add_option("--zeropoint", help="zeropoint", default=30, type=float)
    parser.add_option("-r", "--rate", help="rate Mpc-3 yr-1", default=1e-4, type=float)
    parser.add_option("--ndet", help="Number detections required", default=1, type=float)
    parser.add_option("--threshold", help="SNR required", default=5, type=float)
    parser.add_option("--redshift", help="Max redshift for the simulations (it define the volume simulated)", default=0.1, type=float)
    parser.add_option("-s", "--survey", help="Observing log",default="surveyplan.csv")
    parser.add_option("-n", "--ntransient", help="int number", default=100, type=int)

    parser.add_option("-v", "--verbose", action="store_true", default=False,
                      help="Run verbosely. (Default: False)")

    opts, args = parser.parse_args()

    # show parameters
    if opts.verbose:
        print >> sys.stderr, ""
        print >> sys.stderr, "running gwemopt_run..."
        print >> sys.stderr, "version: %s"%__version__
        print >> sys.stderr, ""
        print >> sys.stderr, "***************** PARAMETERS ********************"
        for o in opts.__dict__.items():
          print >> sys.stderr, o[0]+":"
          print >> sys.stderr, o[1]
        print >> sys.stderr, ""

    return opts

def load_ztf_fields(filename='data/ZTF_Fields.txt', mwebv=False, galactic=False):
    """Load the ZTF fields propose by Eran from the included file.

    Parameters
    ----------
    filename: [str]
        File name of the ASCII file containing the field definitions

    mwebv: [bool]
        Include the Milky Way E(B-V) from the file in the output

    Return
    ------
    Dictionary of np.arrays with the field coordinates, IDs (and exinction)
    """
    fields = np.genfromtxt(filename, comments='%')

    out = {'field_id': np.array(fields[:,0], dtype=int),
           'ra': fields[:,1],
           'dec': fields[:,2]}

    if mwebv:
        out['mwebv'] = fields[:,3]
    if galactic:
        out['l'] = fields[:,4]
        out['b'] = fields[:,5]

    return out

def load_ztf_ccds(filename='data/ZTF_corners.txt', num_segs=16):
    """
    """
    ccd_corners = np.genfromtxt(filename, skip_header=True)
    ccds = [ccd_corners[4*k:4*k+4, :2] for k in range(num_segs)]
    return ccds


# AngularTimeSeriesSource classdefined to create an angle dependent time serie source.
class AngularTimeSeriesSource(sncosmo.Source):
    """A single-component spectral time series model.
        The spectral flux density of this model is given by
        .. math::
        F(t, \lambda) = A \\times M(t, \lambda)
        where _M_ is the flux defined on a grid in phase and wavelength
        and _A_ (amplitude) is the single free parameter of the model. The
        amplitude _A_ is a simple unitless scaling factor applied to
        whatever flux values are used to initialize the
        ``TimeSeriesSource``. Therefore, the _A_ parameter has no
        intrinsic meaning. It can only be interpreted in conjunction with
        the model values. Thus, it is meaningless to compare the _A_
        parameter between two different ``TimeSeriesSource`` instances with
        different model data.
        Parameters
        ----------
        phase : `~numpy.ndarray`
        Phases in days.
        wave : `~numpy.ndarray`
        Wavelengths in Angstroms.
        cos_theta: `~numpy.ndarray`
        Cosine of
        flux : `~numpy.ndarray`
        Model spectral flux density in erg / s / cm^2 / Angstrom.
        Must have shape ``(num_phases, num_wave, num_cos_theta)``.
        zero_before : bool, optional
        If True, flux at phases before minimum phase will be zeroed. The
        default is False, in which case the flux at such phases will be equal
        to the flux at the minimum phase (``flux[0, :]`` in the input array).
        name : str, optional
        Name of the model. Default is `None`.
        version : str, optional
        Version of the model. Default is `None`.
        """

    _param_names = ['amplitude', 'theta']
    param_names_latex = ['A', r'\theta']

    def __init__(self, phase, wave, cos_theta, flux, zero_before=True, zero_after=True, name=None,
                 version=None):
        self.name = name
        self.version = version
        self._phase = phase
        self._wave = wave
        self._cos_theta = cos_theta
        self._flux_array = flux
        self._parameters = np.array([1., 0.])
        self._current_theta = 0.
        self._zero_before = zero_before
        self._zero_after = zero_after
        self._set_theta()

    def _set_theta(self):
        logflux_ = np.zeros(self._flux_array.shape[:2])
        for k in range(len(self._phase)):
            adding = 1e-10 # Here we are adding 1e-10 to avoid problems with null values
            f_tmp = Spline2d(self._wave, self._cos_theta, np.log(self._flux_array[k]+adding),
                             kx=1, ky=1)
            logflux_[k] = f_tmp(self._wave, np.cos(self._parameters[1]*np.pi/180)).T

        self._model_flux = Spline2d(self._phase, self._wave, logflux_, kx=1, ky=1)

        self._current_theta = self._parameters[1]

    def _flux(self, phase, wave):
        if self._current_theta != self._parameters[1]:
            self._set_theta()
        f = self._parameters[0] * (np.exp(self._model_flux(phase, wave)))
        if self._zero_before:
            mask = np.atleast_1d(phase) < self.minphase()
            f[mask, :] = 0.
        if self._zero_after:
            mask = np.atleast_1d(phase) > self.maxphase()
            f[mask, :] = 0.
        return f

# Define MODEL
def model_(phase=0, wave=0, cos_theta=0, flux=0, viewingangle_dependece=True):
    '''phase, wave, cos_theta and flux are 1D arrays with the same length'''
    if viewingangle_dependece:
        source = AngularTimeSeriesSource(phase, wave, cos_theta, flux)
    else:
        source = sncosmo.TimeSeriesSource(phase, wave, flux)
    dust = sncosmo.CCM89Dust()
    return sncosmo.Model(source=source,effects=[dust, dust], effect_names=['host', 'MW'], effect_frames=['rest', 'obs'])

def Bullamodel(mej=0.06, phi=60, dataDir='kilonova_models/bns_m2_2comp/'):
    print('mej=',mej, ',phi=',phi)
    try:
        l = sorted(glob.glob(dataDir + 'nph1.0e+06_mej'+'{:.3f}'.format(mej)+'_phi'+'{:.0f}'.format(phi)+'.txt'))[0]
    except:
        os.system('git clone https://github.com/mbulla/kilonova_models.git')
        l = sorted(glob.glob(dataDir + 'nph1.0e+06_mej'+'{:.3f}'.format(mej)+'_phi'+'{:.0f}'.format(phi)+'.txt'))[0]

    f = open(l)
    lines = f.readlines()
    nobs = int(lines[0])
    nwave = int(lines[1])
    line3 = (lines[2]).split(' ')
    ntime = int(line3[0])
    t_i = float(line3[1])
    t_f = float(line3[2])
    cos_theta = np.linspace(0, 1, nobs)  # 11 viewing angles
    phase = np.linspace(t_i, t_f, ntime)  # epochs
    file_ = np.genfromtxt(l, skip_header=3)
    wave = file_[0:int(nwave),0]
    flux = []
    for i in range(int(nobs)):
        flux.append(file_[i*int(nwave):i*int(nwave)+int(nwave),1:])
    flux = np.array(flux).T

    if phi==0 or phi==90:
        model = model_(phase=phase, wave=wave, cos_theta=cos_theta, flux=flux, viewingangle_dependece=False) # If viewingangle_dependece=False,it won't take into account cos_theta
    else:
        model = model_(phase=phase, wave=wave, cos_theta=cos_theta, flux=flux, viewingangle_dependece=True) # If viewingangle_dependece=False,it won't take into account cos_theta
    return model

# Define random parameters
def transientprop_(modeltype='DES', fixedtheta=False, theta=20, model=0, viewingangle_dependece=True):

    def random_parameters(redshifts, model,r_v=2., ebv_rate=0.11,**kwargs):
        # Amplitude
        amp = []
        for z in redshifts:
            amp.append(10**(-0.4*Planck15.distmod(z).value))

        if modeltype=='DES' or viewingangle_dependece==False:
            return {
                'amplitude': np.array(amp),
                'hostr_v': r_v * np.ones(len(redshifts)),
                'hostebv': np.random.exponential(ebv_rate, len(redshifts))
                }
        elif modeltype=='Bulla' and fixedtheta:
            return {
                'amplitude': np.array(amp),
                'theta': np.array([theta]*len(redshifts)),
                'hostr_v': r_v * np.ones(len(redshifts)),
                'hostebv': np.random.exponential(ebv_rate, len(redshifts))
                }
        elif modeltype=='Bulla' and fixedtheta==False:
            return {
                'amplitude': np.array(amp),
                #'theta': np.random.uniform(0,90,size=len(redshifts)), # distribution of angles.Uniform distribution of theta (viewing angle) ($\^$
                'theta': np.arccos(np.random.random(len(redshifts))) / np.pi * 180, # distribution of angles. Sine distribution of theta (viewing angle) ($\^{circ}$)
                'hostr_v': r_v * np.ones(len(redshifts)),
                'hostebv': np.random.exponential(ebv_rate, len(redshifts))
                }

    return dict(lcmodel=model, lcsimul_func=random_parameters)

def ZTFsurvey(survey='.', dataDir='.', zeropoint=30):
    df = pandas.read_csv(dataDir+survey)
    #df = df[df.programid==2]

    try:
        maglim = df.limMag
    except:
        maglim = df.diff_maglim
    skynoise = 10**(-0.4 * (maglim - zeropoint)) / 5
    fields = load_ztf_fields()

    ccds = load_ztf_ccds(filename='data/ZTF_corners_rcid.txt', num_segs=64)
    zp = np.array([zeropoint]*len(df.jd.values))

    try:
        band = df.filterid#.astype(float).astype(str)
    except:
        band = df.fid.astype(float).astype(str)

    try:
        obs_field = df.fieldid.values
    except:
        obs_field = df.fieldId.values

    '''try:
        obs_ccd = df.chid.values
    except:
        obs_ccd = df.rcid.values'''

    '''try:
        comment = df.progid.values
    except:
        comment = df.programid.values'''

    return simsurvey.SurveyPlan(time=df.jd.values,
                                band=band,
                                skynoise=skynoise,
                                obs_field=obs_field,
                                #obs_ccd=obs_ccd,
                                zp=zp,
                                comment=df.comment,
                                fields=fields
                                #,ccds=ccds
                                )



################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

start = time.time()

opts = parse_commandline()

for mej in mejs:
    for phi in phis:
        opts.mej = mej
        opts.opening_angle = phi
        if opts.modelType == "Bulla":
            mej = opts.mej
            phi = opts.opening_angle
            temp = opts.temp
            model = Bullamodel(mej=mej, phi=phi)
            if phi==0 or phi==90:
                transientprop = transientprop_(modeltype=opts.modelType, fixedtheta=opts.fixedtheta, theta=opts.theta, model=model, viewingangle_dependece=False)
            else:
                transientprop = transientprop_(modeltype=opts.modelType, fixedtheta=opts.fixedtheta, theta=opts.theta, model=model, viewingangle_dependece=True)

        elif opts.modelType == "DES":
            model = DESmodel(dataDir=opts.DEStemplateDir)
            transientprop = transientprop_(modeltype=opts.modelType, model=model, viewingangle_dependece=False)


        plan = ZTFsurvey(survey=opts.survey, dataDir=opts.surveyDir, zeropoint=opts.zeropoint)
        rate = opts.rate
        z_max = opts.redshift

        mjd_range = (plan.pointings['time'].min()-3, plan.pointings['time'].min()-0.1)
        dec_range = (plan.pointings['Dec'].min()-4, plan.pointings['Dec'].max()+4)
        ra_range = (plan.pointings['RA'].min()-4, plan.pointings['RA'].max()+4)

        # path to galactic extinction data
        sfd98_dir = opts.dustDir
        ntransient = opts.ntransient
        trs = simsurvey.get_transient_generator([0, z_max],
                                               ra_range=ra_range,
                                               dec_range=dec_range,
                                               mjd_range=[mjd_range[0],
                                                     mjd_range[1]],
                                               ratefunc=lambda z: rate,
                                               ntransient=ntransient,
                                               transientprop=transientprop,
                                               sfd98_dir=sfd98_dir
                                              )

        survey = simsurvey.SimulSurvey(generator=trs, plan=plan, n_det=opts.ndet, threshold=opts.threshold)
        lcs = survey.get_lightcurves(
                                 progress_bar=True, notebook=True # If you get an error because of the progress_bar, delete this line.
                                 )

        import matplotlib.pyplot as plt

        def check(outputname='check', title=None):
            print('trs.ntransient: ', trs.ntransient)
            print('len(lcs.lcs): ', len(lcs.lcs))
            print('len(lcs.meta_rejected): ', len(lcs.meta_rejected))
            print('len(lcs.meta_full): ', len(lcs.meta_full))
            print('len(lcs.meta_notobserved): ', len(lcs.meta_notobserved))
            print('')
            print('efficiency: ', len(lcs.lcs)/len(lcs.meta_full))

            plt.figure(figsize=(10,10))

            # RA DEC plot
            plt.subplot(331)
            plt.scatter(lcs.meta_rejected['ra'], lcs.meta_rejected['dec'], color='red', label='rejected')
            if len(lcs.lcs)>0:
                plt.scatter(lcs.meta['ra'], lcs.meta['dec'], color='green', label='meta')
            plt.legend(loc=0)
            plt.xlabel('RA [degrees]')
            plt.ylabel('Declination [degrees]')
            plt.title(title)

            # Redshift
            plt.subplot(332)
            plt.hist(lcs.meta_full['z'], label='full')
            plt.hist(lcs.meta['z'], label='meta')
            plt.legend(loc=0)
            plt.xlabel('redshift')
            plt.ylabel('N')

            # t0
            plt.subplot(333)
            plt.hist(lcs.meta_full['t0'], label='full')
            plt.hist(lcs.meta['t0'], label='meta')
            plt.legend(loc=0)
            plt.xlabel('t0')
            plt.ylabel('N')

            # amplitude
            plt.subplot(334)
            plt.hist(lcs.meta_full['amplitude'], label='full')
            plt.hist(lcs.meta['amplitude'], label='meta')
            plt.legend(loc=0)
            plt.xlabel('amplitude')
            plt.ylabel('N')

            # hostr_v
            plt.subplot(335)
            plt.hist(lcs.meta_full['hostr_v'], label='full')
            plt.hist(lcs.meta['hostr_v'], label='meta')
            plt.legend(loc=0)
            plt.xlabel('hostr_v')
            plt.ylabel('N')

            # hostebv
            plt.subplot(336)
            plt.hist(lcs.meta_full['hostebv'], label='full')
            plt.hist(lcs.meta['hostebv'], label='meta')
            plt.legend(loc=0)
            plt.xlabel('hostebv')
            plt.ylabel('N')

            # mwebv
            plt.subplot(337)
            plt.hist(lcs.meta_full['mwebv'], label='full')
            plt.hist(lcs.meta['mwebv'], label='meta')
            plt.legend(loc=0)
            plt.xlabel('mwebv')
            plt.ylabel('N')

            # mwebv_sfd98
            plt.subplot(338)
            plt.hist(lcs.meta_full['mwebv_sfd98'], label='full')
            plt.hist(lcs.meta['mwebv_sfd98'], label='meta')
            plt.legend(loc=0)
            plt.xlabel('mwebv_sfd98')
            plt.ylabel('N')

            # idx_orig
            plt.subplot(339)
            plt.hist(lcs.meta_full['idx_orig'], label='full')
            plt.hist(lcs.meta['idx_orig'], label='meta')
            plt.legend(loc=0)
            plt.xlabel('idx_orig')
            plt.ylabel('N')

            plt.tight_layout()

            plt.savefig(outputname, dpi=500)

        import csv

        filename = opts.survey[:-4]+'_'+opts.modelType+'_mej'+'{:.0e}'.format(opts.mej)+'_phi'+'{:.0f}'.format(opts.opening_angle)+'_fixedthe'+str(opts.fixedtheta)+'_'+'{:.0f}'.format(trs.ntransient)
    
        if opts.showplot:
            check(outputname=opts.outputDir+filename)

        log = [['log:', filename]]
        log.append(['modelType' , opts.modelType])
        if opts.fixedtheta:
            log.append(['fixedtheta' , opts.theta])
        else:
            log.append(['fixedtheta' , opts.fixedtheta])
        log.append(['mej (M_sun)' , opts.mej])
        log.append(['opening_angle (º)' , opts.opening_angle])
        log.append(['temp (K)' , opts.temp])
        log.append(['zeropoint' , opts.zeropoint])
        log.append(['rate (Mpc-3 yr-1)' , opts.rate])
        log.append(['redshift' , opts.redshift])
        log.append(['survey' , opts.survey])
        log.append(['len(lcs)' , len(lcs.lcs)])
        log.append(['ntransient' , trs.ntransient])
        log.append(['ndet' , opts.ndet])
        log.append(['threshold' , opts.threshold])

        with open(opts.outputDir+filename+'.csv', 'w') as f:
            csv.writer(f).writerows(log)


        #dill.dump(trs, open(opts.outputDir+'tr_'+filename+'.pkl','wb'))
        pickle.dump(lcs, open(opts.outputDir+'lcs_'+filename+'.pkl','wb'))


        end = time.time()
        time_taken = end - start
        print('Execution time: ',time_taken)
