import numpy as np
import pandas as pd
import lmfit as lm
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.special import erf,erfinv
from comibis.utils import *
import warnings
warnings.filterwarnings("ignore")

############### Probabilities ###############

def p_to_nsigma(p_unpola):
    '''compute the n-sigma significance from the proba of false detection
    technically this is not correct since polarigram parameters (a,phi0) are not normally distributed but it gives a quick interpretation of results
    /!\ This n-sigma value SHOULD NOT BE PUBLISHED
    '''
    return np.sqrt(2)*erfinv(1-p_unpola)

############### Fit polarigram ###############

def sin_pola(x,C,a0,phi0):
    '''theoretical polarigram model'''
    return C * (1 + a0 * np.cos(2 * (x - phi0) * (np.pi / 180) ) )

class Polarigram:
    '''polarigram = flux per radiant as a function of azimuthal scattering angle, for every energy band found'''

    def __init__(self, df_scw, all_band_names, angle_max):
        self.angle_max, self.df_scw, self.all_band_names = angle_max, df_scw, all_band_names
        self.expo = self.df_scw.EXPO.sum()
        self.n_pola_bands = find_pola_bands(self.df_scw, self.all_band_names)
        self.ISOT_START, self.ISOT_END = self.df_scw.ISOT.min(), self.df_scw.ISOT.max()

    def import_prf(self, pulsefrac_dir='compton_responses', pulsefrac_file='comps-prf-1.txt'):
        '''import the polarization response file that contains a100 = a100(E)'''
        pulsefrac_path= '{}/{}'.format(pulsefrac_dir, pulsefrac_file)
        self.df_prf=pd.read_csv(pulsefrac_path, delim_whitespace=True, names=['E','compton_event','a100','a100_err'])

    def make_polar(self, spicorr, compnorm=1):
        self.spicorr = spicorr
        self.compnorm = compnorm
        all_pola_dico_comp = sum_scw_df(self.df_scw, self.all_band_names, self.spicorr, self.compnorm, n_pola_bands=self.n_pola_bands)
        self.pola_dico=all_pola_dico_comp['simple']
        spec_list = [ [int(e) for e in b.split('-')]+[np.sum(self.pola_dico[b][0])]+[np.sum(self.pola_dico[b][1])] for b in self.pola_dico]
        self.df_spec = pd.DataFrame(spec_list, columns=['Einf','Esup','Flux','Flux_err'])

    def combine_bands(self, Emin, Emax):
        '''finds the biggest interval inside [e_min, e_max] from pola_dico'''
        bands=[]
        for b in self.all_band_names:
            e1,e2=b.split('-')
            if int(e1)>=int(Emin) and int(e2)<=int(Emax): bands.append(b)
        if len(bands)==0:
            print('no energy bands found !')
        return bands

    def calc_a100(self, Emin, Emax):
        ''' calcul la moyenne de a100 pondérée par F(E)'''    
        df_spec_born = self.df_spec[(self.df_spec.Einf>=Emin)&(self.df_spec.Esup<=Emax)]
        a100, Ftot = 0, 0
        dE=self.df_prf.E.iloc[1]-self.df_prf.E.iloc[0] # we suppose that f(E)=a100 has constant step
        for spec in df_spec_born.values:
            E1, E2, F, F_err= spec
            Ftot += F*(E2-E1)
            a100 += F * self.df_prf[(self.df_prf.E>=E1)&(self.df_prf.E<E2)].a100.sum() # convolution of a100 with F
        return (a100*dE)/Ftot, df_spec_born
    
    def a0_upper_lim(self, result, a100, p=0.01, verbose=True):
        '''gives an estimate of upper-limits by inverting the probability formula (see Bouchet+2024, A&A)'''
        c,sig_c = result.best_values['C'], result.data.std()#result.params.get('C').stderr
        z2 = (self.n_pola_bands*c**2)/(2*sig_c**2)
        a_up = np.sqrt(-(1/z2)*np.log(p + np.exp(-z2*a100**2)))
        if verbose: print('upper-limits: a0:{0:.3f}, PF:{1:.3f}'.format(a_up, a_up/a100))
        return a_up
    
    def phi_to_pa(self, phi_0, pa_mod):
        '''convert raw phi value into Polarization Angle (PA), with different references
        ref=0 : -90° < PA < 90°
        ref=90 : 0° < PA < 180°
        '''
        PA = phi_0 - 90
        if pa_mod=='PA_ref0': return PA - 180*(PA>90) # substract 180° if above 90°
        elif pa_mod=='PA_ref90': return PA + 180*(PA<0) # add 180° if below 0°
        else: return PA 

    def find_polar_param(self, result, a100, pa_mod, p_higher, p_uplim=0.01):
        '''returns Polarization Angle and Fraction from fit result, and upper-limit flag'''
        # if sigma_higher>sigma_threshold: # if polarization is detected, a0/phi0 from fit are kept
        if p_higher < p_uplim: # if polarization is detected, a0/phi0 from fit are kept
            uplim=0
            a_0=result.best_values['a0']
            phi_0=result.best_values['phi0']
        else: # otherwise use upper-limit
            uplim=1
            a_0 = self.a0_upper_lim(result, a100, p=p_uplim, verbose=0)
            phi_0 = np.nan
        PA, PA_err = self.phi_to_pa(phi_0, pa_mod), result.params.get('phi0').stderr
        PF, PF_err = a_0/a100, result.params.get('a0').stderr/a100
        return PA, PA_err, PF, PF_err, uplim
    
    def proba_higher(self, result, a100, verbose=True):
        ''' finds the probability to have higher a0 then the one fitted, knowing the source is unpolarized (a0=0)
        P(a > a0 measured | a0 source = 0) '''
        c,sig_c = result.best_values['C'], result.data.std()
        a0 = result.best_values['a0']
        z2 = (self.n_pola_bands*c**2)/(2*sig_c**2)
        p_higher = (np.exp(-z2*a0**2) - np.exp(-z2*a100**2))
        sigma_higher = p_to_nsigma(p_higher)
        # print('a0= {0} a100= {1}'.format(a0, a100))
        if verbose:
            print('p unpola=', p_higher)
            print('{0:.2f}-sigma detection'.format(sigma_higher))
        return p_higher, sigma_higher
    
    def fit_pola(self, bands, p_uplim=0.01, folded=1, weighted=True, pa_mod=None, verbose=True, article=False):
        '''fit the polarigram and deduce PA/PF values, given a list of energy bands'''
        
        Emin, Emax = bands[0].split('-')[0],bands[-1].split('-')[1] # find min/max energy of the bands
        ### add the different bands together
        y=np.zeros(self.n_pola_bands)
        y_err=np.zeros(self.n_pola_bands)
        for b in bands:
            y+=self.pola_dico[b][0]
            y_err+=np.array(self.pola_dico[b][1])**2
        y_err=np.sqrt(y_err)

        # dividing count-rates by bin angle width allows to have binning-invariant polarigrams (similar to flux/keV for spectra)
        # folded=1 is the default, folded=0 means the polarigram was not folded on [0,pi] during ic_PolaIma computation (only for testing purpose)
        bin_width= (2-folded)*np.pi/self.n_pola_bands # bin width in radian
        y = y/(bin_width*(1+folded))
        y_err = y_err/(bin_width*(1+folded))

        SNR = y.sum()/np.sqrt(np.sum(y_err**2))
        a100, df_spec_born = self.calc_a100(int(Emin), int(Emax)) # compute a100 for the given energy range
        pola_width = ((2-folded)*180/self.n_pola_bands) # 180° if folded, 360° if not 
        x_pola = np.array([pola_width*(i+1/2) for i in range(self.n_pola_bands)]) # convert pola bin to angle phi
        #y_test=np.random.normal(0.1, np.mean(pola_dico[b][0]), self.n_pola_bands)

        ### fit polarigram
        mean_y=np.mean(y)
        par_pola = lm.Parameters()
        par_pola.add('C',value=mean_y)
        if folded: par_pola.add('a0',value=(np.max(y)-mean_y)/mean_y)
        else: par_pola.add('a0',value=(np.max(y)-mean_y)/mean_y,min=0)
        max_phi = x_pola[np.argmax(y)] # angle of maximum flux
        par_pola.add('phi0',value=max_phi) # no constraint on phi0, this is preferred
        
        gmodel = lm.Model(sin_pola)
        
        ### fit with constant to compare Chi2
        par_nopola= lm.Parameters()
        par_nopola.add('c',value=mean_y)
        gmodel_nopola = lm.models.ConstantModel()
        if weighted: # does not change result since y_err is pretty much constant, but gives chi2 value
            result = gmodel.fit(y, par_pola, x=x_pola, weights=1/y_err)
            result_nopola = gmodel_nopola.fit(y, par_nopola, x=x_pola, weights=1/y_err)
            print('reduced chi2: pola={0}, no pola={1}'.format(result.redchi, result_nopola.redchi))
        else: result = gmodel.fit(y, par_pola, x=x_pola)

        # find all relevant polarization parameters and save them in dico
        p_higher, sigma_higher = self.proba_higher(result, a100, verbose=False) # find probability and n-sigma estimate
        flux, flux_err = df_spec_born.Flux.sum(), np.sqrt(np.sum(df_spec_born.Flux_err**2))
        PA, PA_err, PF, PF_err, uplim = self.find_polar_param(result, a100, pa_mod, p_higher, p_uplim)

        pola_param = {'Emin':Emin,'Emax':Emax, 'PA':PA, 'PA_err':PA_err, 'PF':PF, 'PF_err':PF_err, 'uplim':uplim,\
                      'SNR':SNR, 'redchi2':result.redchi, 'p_higher':p_higher, 'sigma_higher':sigma_higher}

        if verbose:
            print('Polarized flux in '+Emin+'-'+Emax+' keV band')
            # plot the result
            x_model=np.linspace(np.min(x_pola), np.max(x_pola), num=500)
            x_model_360=np.linspace(np.min(x_pola), 360, num=500)
            fig, ax = plt.subplots(figsize=(8,5))
            label_font=15
            label_fit='PA = {:.1f} ± {:.1f} °\nPF = {:.2f} ± {:.2f} %\nSNR = {:.1f}'.format(PA, PA_err, PF*100, PF_err*100, SNR)
            label_data=''
            ax.axhline(y=result.best_values['C'],color='grey',ls='--')
            ax.axhline(y=0,color='k')
            if folded: # duplicates the [0,pi] polarigram to be on [0,2*pi]
                ax.errorbar(x=np.concatenate((x_pola,(x_pola+180))), y=np.concatenate((y,y)), 
                            yerr=np.concatenate((y_err,y_err)), xerr=pola_width/2, fmt='k.', label=label_data) 
                ax.plot(x_model_360, result.eval(x=x_model_360), 'r-', label=label_fit)
            else:
                ax.errorbar(x=x_pola, y=y, yerr=y_err,xerr=pola_width/2, fmt='.', label=label_data)
                ax.plot(x_model, result.eval(x=x_model), 'r-', label=label_fit)
            ax.xaxis.label.set_size(label_font);ax.yaxis.label.set_size(label_font);ax.tick_params(which='both', labelsize=label_font)
            ax.set_xlabel('$\phi$ (°)');ax.set_ylabel(r'Counts s$^{-1}$ rad$^{-1}$')
            plt.legend(loc='best',fontsize=14);plt.grid(True);plt.show()

            print('mean flux = {0} ± {1} ct/s/rad'.format(np.mean(y), np.std(y)),' a0*C={0}'.format(result.best_values['a0']*result.best_values['C']))
            print('total flux = {0:.4f} ± {1:.4f} ct/s'.format(flux, flux_err))
            print('SNR = {0:.1f} P(no pola) = {1:.5f} -> {2:.1f}-sigma detection'.format(SNR, p_higher, sigma_higher))
            # print('a0 = {0:.3f},  a100 = {1:.3f}'.format(result.best_values['a0'],a100))
            print('PA = {0:.1f} ± {1:.1f}\nPF = {0:.2f} ± {1:.2f}'.format(PA, PA_err, PF, PF_err))

        return pola_param
    
    def pola_espectrum(self, energy_bands, pa_mod='PA_ref90', SNR_threshold=12, p_uplim=0.01, verbose=0):
        '''find best fit PA/PF in given energy bands if detection is significant enough, and returns df for plot
        energy_bands = [['e0 - e1'], ['e2 - e3'], etc...]
        '''

        if energy_bands=='all': energy_bands = self.all_band_names # if using original energy bands
        list_pola_param=[] # list of polar parameters dictionaryies
        for band in energy_bands:
            try:
                if verbose: print('\n\n*** -------------------------------------------------------------------------\n')
                Emin, Emax = band.split('-')
                bands_inside = self.combine_bands(Emin, Emax) # find the smaller energy band(s) inside the given energy band
                pola_param = self.fit_pola(bands_inside, p_uplim=0.01, verbose=1, article=0, weighted=0, pa_mod=pa_mod, folded=1)

                if  pola_param['SNR'] > SNR_threshold: # if the SNR is sufficient for meaningful detection
                    list_pola_param.append(pola_param)

            except ValueError: # this can happen for empty energy bands or negative flux
                print('nan value in fit for {} - {} keV band'.format(Emin, Emax))
                continue

        # create df with parameters of each polarigram
        self.df_pola_param=pd.DataFrame(list_pola_param) # build data frame from list of polar dict
        self.df_pola_param['E_mean'] = self.df_pola_param.apply(lambda x:(int(x.Emin)+int(x.Emax))/2, axis=1 )
        self.df_pola_param['dE'] = self.df_pola_param.apply(lambda x:(int(x.Emax)-int(x.Emin))/2, axis=1 )
        self.df_pola_param['PF_pct'] = self.df_pola_param.PF*100 # convert to percent
        self.df_pola_param['PF_err_pct'] = self.df_pola_param.PF_err*100
        self.df_pola_param['uplim_pct'] = self.df_pola_param.uplim*100
        return self.df_pola_param

    def plot_polar_espectrum(self, plot_scale='lin', plot_percent=True, plot_grid=0, with_snr=0, fmt='k.'):
        '''plot the PA/PF as a function of energy'''
        if with_snr: fig,ax=plt.subplots(3,1,figsize=(10,10))
        else: fig,ax=plt.subplots(2,1,figsize=(10,8))
        PA_mean, PF_mean = self.df_pola_param[['PA','PF']].mean()
        PA_std, PF_std = self.df_pola_param[['PA','PF']].std()
        # Polarization Angle (PA) = upper plot
        ax[0].errorbar('E_mean', 'PA', yerr='PA_err', xerr='dE', fmt=fmt, data=self.df_pola_param, label=None)
        ax[0].axhline(PA_mean, linestyle='--', c='k', label='Average PA = {0:.1f} ± {1:.1f} °'.format(PA_mean,PA_std))
        ax[0].set_ylabel('PA (°)')
        # ax[0].tick_params(labelbottom=False) 
        ax[0].legend(loc='upper right')#;ax[1].legend()#

        if plot_percent: ax[1].errorbar('E_mean','PF_pct',yerr='PF_err_pct',xerr='dE',fmt=fmt,  data=self.df_pola_param,label=None, uplims=self.df_pola_param['uplim_pct'])
        else: ax[1].errorbar('E_mean','PF',yerr='PF_err',xerr='dE',fmt=fmt,  data=self.df_pola_param,label=None, uplims=self.df_pola_param['uplim'])

        #ax[1].axhline(PF_mean,linestyle='--',c='k',label='Average PF = {0:.2f} ± {1:.2f}'.format(PF_mean, PF_std))
        ax[1].set_ylabel('PF' + plot_percent*' (%)')
        # ax[1].set_ylim(bottom=0,top=1.)
        ax[0].set_xlim(ax[1].get_xlim()) # to have the same x axis in case there are upper-limits

        if with_snr: # plot the SNR of each energy band
            ax[2].errorbar('E_mean','SNR',xerr='dE',fmt=fmt,  data=self.df_pola_param)
            ax[2].set_ylabel('SNR');ax[2].set_xlabel('E (keV)')
            ax[2].legend()
        else: ax[1].set_xlabel('E (keV)')

        for i in range(len(ax)): # other plot parameters
            ax[i].xaxis.label.set_size(16);ax[i].yaxis.label.set_size(16)
            ax[i].tick_params(which='both', labelsize=16)
            if plot_grid==True: ax[i].grid(True);ax[i].grid(True)
            if plot_scale=='log': ax[i].set_xscale('log');ax[i].set_xscale('log')

