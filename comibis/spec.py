"""
Classes and functions for IBIS/Compton mode spectral analysis
"""

from comibis.utils import *
from datetime import datetime
import lmfit as lm
from astropy.io import fits
from astropy.table import Table
from matplotlib.ticker import LogFormatter

############### Constants ###############

# dico containing the units, multiplicative factors and energy exponents (ex: F(erg/s/cm2) = keV_to_erg * F**2)
spec_dico = {'RATE':[r'Count s$^{-1}$ keV$^{-1}$',1,0], 'FLUX':[r'ph cm$^{-2}$ s$^{-1}$ keV$^{-1}$',1,0], 'EFLUX':[r'ph cm$^{-2}$ s$^{-1}$',1,1], 'EEFLUX':[r'keV cm$^{-2}$ s$^{-1}$',1,2], 'ERG':[r'erg cm$^{-2}$ s$^{-1}$',kev_to_erg,2]}
res_dico = {'RES':[0,r'$\sigma$'],'REDCHI2':[1,r'$\chi^2_{red}$']}

############### Classes ###############

class Response:
    ''' import all the useful response info (RMF, ARF, energy bins) into a single object'''

    def __init__(self, resp_dir, rmf_ebound_ext, rmf_matrix_ext, rmf_ver='6', arf_ver='6',
                 rmf_name='comps-rmf-', arf_name='comps-arf-'):
        ''' energy of each channel of incident/true photons = J (281 usually) = size of ARF
            detected/reconstructed photons of simu = I (3321 without rebin)
        '''
        self.rmf_file_name = f'{resp_dir}/{rmf_name}{rmf_ver}.fits'
        self.arf_file_name = f'{resp_dir}/{arf_name}{arf_ver}.fits'
        hdul_rmf = fits.open(f'{self.rmf_file_name}') # original RMF
        hdul_arf = fits.open(f'{self.arf_file_name}')
        self.rmf_mat = hdul_rmf[rmf_matrix_ext].data
        self.rmf_ebd = hdul_rmf[rmf_ebound_ext].data
        self.I = (self.rmf_ebd['E_MAX'] + self.rmf_ebd['E_MIN'])/2
        self.dI = self.rmf_ebd['E_MAX'] - self.rmf_ebd['E_MIN']
        self.J = (self.rmf_mat['ENERG_HI'] + self.rmf_mat['ENERG_LO'])/2
        self.dJ = self.rmf_mat['ENERG_HI'] - self.rmf_mat['ENERG_LO']
        self.arf = hdul_arf[1].data['SPECRESP'] # ARF (cm2) as a function of channel (J)
        # self.arf = self.hdul_arf['COMP-SARF-RSP'].data['SPECRESP'] # only works for IBIS/Compton

    def make_rbn_mat(self, E_bounds_rbn):
        '''creates a bool matrix that tells if channel I is inside [Ei, Ei+1] of Compton spec'''
        self.rbn_IE_matrix = np.array([(self.I>erbn[0])&(self.I<=erbn[1]) for erbn in E_bounds_rbn]).T
        # return self.rbn_IE_matrix
    
    def plot_rmf(self, vmax=4e-3, cmap='magma'):
        I_min, I_max, J_min, J_max = self.I[0], self.I[-1], self.J[0], self.J[-1]
        fig, ax= plt.subplots(1,1,figsize=(10,8))
        cb=ax.imshow(self.rmf_mat['MATRIX'], extent=[I_min, I_max, J_min, J_max ],
                    origin='lower', aspect= (I_min - I_max)/(J_min - J_max), vmax=vmax, cmap=cmap)
        ax.set_xlabel('Reconstructed Energy (keV)')
        ax.set_ylabel('True Energy (keV)')
        plt.colorbar(cb)
        return ax
    
    def plot_arf(self, color='k'):
        fig, ax= plt.subplots(1,1,figsize=(8,6))
        ax.step(self.J, self.arf, color=color,where='mid' )
        ax.set_xlabel('Energy (keV)')
        ax.set_ylabel(r'Effective area (cm$^2$)')
        return ax


class Spectrum:
    '''
    Spectrum superclass
    so far it is used to build spectrum of: IBIS/Compton, ECLAIRs,
    '''
    # def __init__(self, spec_sys_error):
    #     self.sys_error=spec_sys_error
    
    def make_fit_ebd(self, e_min_fit, e_max_fit):
        self.E_bounds_fit = (self.E>e_min_fit)&(self.E<=e_max_fit) # bool vector to select E bounds for fit

    def plot_raw_spec(self, logscale=True):
        plt.errorbar(x=self.E, xerr=self.dE/2, y = self.rate, yerr=self.rate_err, fmt='.')
        plt.xlabel('E (keV)')
        plt.ylabel('Rate (Counts/s/keV)')
        if logscale: plt.loglog()
        
    def save_spec_fits(self, saved_spec_dir, spec_name, spec_temp_name, resp=Response, add_header_info={}):
        '''
        saved the spectrum as '.fits' in COUNT/S using a template
        '''

        # add_header_info = {'SPICORR': self.spicorr, 'OFFAXIS': self.angle_max}
        # ADD RMF CREATION

        # init default header infos
        head_dico_spec = { 'DETCHANS':self.bin_num, 'TLMAX1':int(self.bin_num)-1,'DATE':datetime.now().isoformat()}
        if resp:
            head_dico_spec['RESPFILE'] = resp.rmf_file_name
            head_dico_spec['ANCRFILE'] = resp.arf_file_name
            # print(f'rbnrmf infile={resp.rmf_file_name} ')

        head_dico_spec.update({'EXPOSURE':self.expo, 'POISSERR':0.,})
        
        with fits.open(spec_temp_name) as hdul_model:
            for head_name in head_dico_spec: hdul_model[1].header[head_name] = head_dico_spec[head_name] # fill spectrum header
            t=Table([np.arange(self.bin_num, dtype='int16'), # Channel
                     self.rate * self.dE, self.rate_err*self.dE, # rate and stat error, converted from ct/s/keV to ct/s
                     np.ones(dtype='float32',shape=self.bin_num) * self.sys_error, # sys error
                     np.zeros(dtype='float32',shape=self.bin_num), # 
                     np.ones(dtype='int16' ,shape=self.bin_num)],
                names=('CHANNEL', 'RATE','STAT_ERR','SYS_ERR','QUALITY','GROUPING'))
            hdul_model[1].data = fits.BinTableHDU(t).data # replace the data with a Table
            hdul_model.writeto(f'{saved_spec_dir}/{spec_name}', overwrite=True)
        return
    
class ComptonSpectrum(Spectrum):
    '''use the polarization fluxes to build a ct/s spectrum, then a ct/s/keV spectrum'''
    def __init__(self, df_scw, band_names, spec_sys_error, angle_max):
        self.sys_error=spec_sys_error 
        self.angle_max, self.df_scw, self.band_names = angle_max, df_scw, band_names
        self.expo = self.df_scw.EXPO.sum()
        self.n_pola_bands = find_pola_bands(self.df_scw, self.band_names)
        self.E_bounds_rbn = np.array([np.int64(b.split('-')) for b in self.band_names]) # energy bounds of Compton spec
        self.E = self.E_bounds_rbn.mean(axis=1) # mean energies
        self.dE = np.diff(self.E_bounds_rbn).flatten() # energy widths
        self.bin_num = len(self.E)
        self.ISOT_START, self.ISOT_END = self.df_scw.ISOT.min(), self.df_scw.ISOT.max()

    def make_rate(self, spicorr, compnorm=1, df_spicorr_rbn=None):
        self.spicorr = spicorr
        self.compnorm = compnorm
        all_pola_dico_comp = sum_scw_df(self.df_scw, self.band_names, self.spicorr, self.compnorm, self.n_pola_bands, df_spicorr_rbn)
        pola_dico=all_pola_dico_comp['simple']
        comp_count= [np.sum(pola_dico[e][0]) for e in pola_dico.keys()] # sum over pola bin
        comp_count_err=[np.sum(pola_dico[e][1]) for e in pola_dico.keys()]
        self.rate, self.rate_err = comp_count/self.dE, comp_count_err/self.dE # spectrum (ct/s/keV)
        self.rate_err = np.sqrt((self.sys_error * self.rate)**2 + self.rate_err**2)  # add systematic error in quadrature, as a percent of the flux
        return

    def make_spec_df(self, resp, result, model):
        ''' compute the flux (in ph/s/cm2/keV) of Compton spec from the count-rate
            this is done by estimating the inverse response: M = Fm * R => R^-1 = Fm / M, where Fm and M are the flux and count-rate of the model
        '''
        self.df_spec = pd.DataFrame({'E':self.E, 'E_ERR':self.dE/2, 'RATE':self.rate, 'RATE_ERR':self.rate_err})
        flux_rate_ratio = calc_model_flux(self.E, model.model_parameters, model.model_name) / ((model.rate * resp.dI)@resp.rbn_IE_matrix/self.dE) # equivalent to an inverse response
        self.df_spec['FLUX'] = flux_rate_ratio * self.df_spec.RATE # ph/s/cm2/keV
        self.df_spec['FLUX_ERR'] = flux_rate_ratio *  self.df_spec.RATE_ERR 

        # create another df only inside the fit energy bounds
        self.df_spec_fit = self.df_spec[self.E_bounds_fit].copy()
        self.df_spec_fit['RES'] = -result.residual
        self.df_spec_fit['RES_ERR'] = 1.
        if result.nfree==0:
            print('No degree of freedom in fit!')
            return 0
        else:
            self.df_spec_fit['REDCHI2'] = (result.residual)**2*(result.ndata/result.nfree)
            self.df_spec_fit['REDCHI2_ERR'] = 0.
            return 1
        
    def save_spec(self, saved_spec_dir, spec_name, spec_temp_name='comp_spec_template.fits', resp=None):
        ''' saved the compton spectrum as '.fits' in COUNT/S using a template
            NB: this will require a re-binned RMF if used in Xspec because the spectrum is in channel and not energy!
        '''
        head_dico_spec = {'EXPOSURE':self.expo, 'RESPFILE':f'comps-rmf-6-rbn{self.bin_num}.fits', 'ANCRFILE': 'comps-arf-6.fits',
            'POISSERR':0., 'DETCHANS':self.bin_num, 'TLMAX1':int(self.bin_num)-1, 'SPICORR': self.spicorr, 'OFFAXIS': self.angle_max, 'DATE':datetime.now().isoformat()}
        with fits.open(spec_temp_name) as hdul_model:
            for head_name in head_dico_spec: hdul_model[1].header[head_name] = head_dico_spec[head_name] # fill spectrum header
            t=Table([np.arange(self.bin_num, dtype='int16'), self.rate * self.dE, self.rate_err*self.dE, # convert ct/s/keV to ct/s
                np.zeros(dtype='float32',shape=self.bin_num), # np.ones(dtype='float32',shape=self.bin_num) * self.sys_error,
                np.zeros(dtype='float32',shape=self.bin_num), np.ones(dtype='int16' ,shape=self.bin_num)],
                names=('CHANNEL', 'RATE','STAT_ERR','SYS_ERR','QUALITY','GROUPING'))
            hdul_model[1].data = fits.BinTableHDU(t).data
            hdul_model.writeto(f'{saved_spec_dir}/{spec_name}', overwrite=True)

        # ADD RMF CREATION
        if resp:
            print(f'rbnrmf infile={resp.rmf_file_name} ')
        return

class Model:
    '''Model object contains model name/parameters and is used for fitting procedure'''
    def __init__(self,  model_name): # model_parameters
        self.model_name = model_name

    def calc_rate(self, resp):
        '''convert flux spec into rates with the response'''
        model_flux = calc_model_flux(resp.J, self.model_parameters, self.model_name) # flux (ph/cm2/s/keV) is always calculated using the RMF energies (J)
        self.rate = ((model_flux * resp.dJ * resp.arf)@resp.rmf_mat['MATRIX'] / resp.dI) # conversion flux->count/s with instrument response (RMF+ARF) and energy bin size (rmf_dI, rmf_dJ)
        return self.rate
    
    def residual_compton(self, fit_params, spectrum, resp):
        ''' return residual array on desired energy bounds'''
        self.model_parameters = fit_params
        self.model_spec_rbn = (self.calc_rate(resp) * resp.dI)@resp.rbn_IE_matrix/spectrum.dE # model rate and matrix re-binning operation
        return ((self.model_spec_rbn - spectrum.rate) / spectrum.rate_err)[spectrum.E_bounds_fit] # the length of this array will be used for the reduced chi2
    
    def residual_alpha(self, fit_params, spectrum, resp):
        ''' ONLY FOR CALIBRATION (used to fit SpiCorr)
        '''
        self.model_parameters = fit_params
        self.model_spec_rbn = (self.calc_rate(resp) * resp.dI)@resp.rbn_IE_matrix/spectrum.dE # model rate and matrix re-binning operation
        spectrum.make_rate(spicorr=fit_params['spicorr'], compnorm = fit_params['compnorm'],) # re-make the spectrum with new spicorr
        return ((self.model_spec_rbn - spectrum.rate) / spectrum.rate_err)[spectrum.E_bounds_fit] # the length of this array will be used for the reduced chi2
    
    def make_spec_df(self, resp, result):
        self.model_parameters = result.params # update params after fit
        self.df_spec =  pd.DataFrame({'E':resp.I, 'E_ERR':resp.dI/2, 'RATE':self.calc_rate(resp)})
        self.df_spec['FLUX'] = calc_model_flux(resp.I, self.model_parameters, self.model_name)

    def rebin_spec_fact(self, rebin_fact=5, rbn_type= 'linear'):
        ''' rebin the spectrum by grouping channels to obtain N/rebin_fact channels
            linear: group rebin_fact adjacent, log: group on log scale to have larger grouping towards higher energies
            returns a df with the same unit for RATE (here dE = 2 * E_ERR)
        '''
        if rbn_type=='linear': self.df_spec['RBN'] = self.df_spec.index//rebin_fact # assign a re-binning number for each channel
        if rbn_type=='log':  self.df_spec['RBN'] = np.int64(np.logspace(np.log10(len(self.df_spec)//rebin_fact), 0, len(self.df_spec)))
        group_rbn = self.df_spec.groupby('RBN') # group df by re-binning number
        group_rbn_sum, group_rbn_mean = group_rbn.sum(), group_rbn.mean()
        return pd.DataFrame({'E':group_rbn_mean.E, 'E_ERR':group_rbn_sum.E_ERR, 'RATE':group_rbn_sum.RATE/(2 * group_rbn_sum.E_ERR),
                'FLUX':group_rbn_sum.FLUX/(2 * group_rbn_sum.E_ERR)})
    
    def calc_flux_model(self, e_flux_min, e_flux_max, flux_type='eeuf', N_flux_bin=100, verbose=1):
        ''' sum (uf) model over a given energy band to obtain the flux
            can be in 'euf'= ph/cm2/s, or 'eeuf' = erg/cm2/s 
        '''
        flux_E = np.linspace(e_flux_min, e_flux_max, N_flux_bin) # energy bins for integration
        dE = np.diff(flux_E)
        uf = calc_model_flux(flux_E, self.model_parameters, self.model_name)[:-1] # ph/cm2/s/keV remove last one for "left rectangle" integral
        euf = kev_to_erg * uf * flux_E[:-1] # erg/cm2/s/keV
        f_dico = {key: np.sum(f * dE) for (key,f) in zip(['euf','eeuf'],[uf,euf])} 
        if verbose:
            print('flux in {} - {} keV: {:.2e} erg/cm2/s , {:.2e} ph/cm2/s'.format(e_flux_min, e_flux_max, f_dico['eeuf'],f_dico['euf']))
        return f_dico[flux_type]

def calc_model_flux(E, par, model_name):
    ''' contains simple models (based on Xspec definitions)'''
    if model_name=='powerlaw': return par['K'] * (E**(-par['gamma'])) # K in ph/cm2/s/keV @ 1 keV, so E must be in keV
    if model_name=='cutoffpl': return par['K'] * E**(-par['alpha']) * np.exp(-E/par['beta'])
    if model_name=='const': return par['K'] * np.ones_like(E)
    if model_name=='bknpower':
        lowE, highE = E[E <= par['Eb']], E[E > par['Eb']]
        return np.concatenate((par['K'] * lowE**(-par['gamma1']),\
                               par['K'] * par['Eb']**(par['gamma2'] - par['gamma1']) * highE**(-par['gamma2'])))
    if model_name=='grbm':
        dalpha = par['alpha1'] - par['alpha2']
        lowE, highE = E[E < par['Ec'] * dalpha], E[E >= par['Ec'] * dalpha]
        return np.concatenate((par['K'] * (lowE/100.)**par['alpha1'] * np.exp(-lowE/par['Ec']),\
                               par['K'] * (dalpha * par['Ec']/100.)**dalpha * (highE/100.)**par['alpha2'] * np.exp(-dalpha)))
    
    if model_name=='powerlaw_flux':
        # variant of powerlaw model, where K is the integrated flux between e1 and e2, in ph/cm2/s
        return par['K'] * ((par['gamma']-1) * E**(-par['gamma'])) / (par['e1']**(1-par['gamma']) - par['e2']**(1-par['gamma']))
    if model_name=='powerlaw_erg':
        # variant of powerlaw model, where K is the integrated flux between e1 and e2, in erg/cm2/s
        return (par['K']/kev_to_erg) * (par['gamma'] - 2) * E**(-par['gamma']) / (par['e1']**(2-par['gamma']) - par['e2']**(2-par['gamma']))
    else: raise ValueError(f'{model_name} model is not implemented!')

############### Fit functions ###############

def plot_spec_model(spectrum, model, e_min_fit, e_max_fit, spec_type='RATE', res_type='REDCHI2', rebin_fact_model=False, rbn_type='log',yscale=None):
    show_residuals = 1 if res_type else 0 # used both as bool and int
    if rebin_fact_model: df_spec_model = model.rebin_spec_fact(rebin_fact_model, rbn_type)
    else: df_spec_model = model.df_spec
    df_spec_model_fit = df_spec_model[(df_spec_model.E>e_min_fit)&(df_spec_model.E<e_max_fit)].copy()
    df_spec_compton_fit = spectrum.df_spec_fit

    spec_type_param = spec_dico[spec_type]
    fig, axes = plt.subplots(1+show_residuals, 1, figsize=(9,7), height_ratios=[2,1]*show_residuals+[1]*(1-show_residuals), squeeze=False) # squeeze=0 makes it an array
    if spec_type=='RATE': 
        axes[0,0].plot('E', spec_type, 'r', label='Best fit', data=df_spec_model_fit)
        axes[0,0].errorbar(x='E', xerr='E_ERR', y=spec_type, yerr=spec_type+'_ERR', fmt='k.', label='Data spectrum', data=df_spec_compton_fit)
    else:
        axes[0,0].plot(df_spec_model_fit.E, spec_type_param[1] * df_spec_model_fit['FLUX'] * (df_spec_model_fit.E**spec_type_param[2]), 'r', label='Best fit')
        axes[0,0].errorbar(x = df_spec_compton_fit.E, xerr = df_spec_compton_fit.E_ERR, \
                           y = spec_type_param[1] * df_spec_compton_fit['FLUX'] * (df_spec_compton_fit['E']**spec_type_param[2]), \
                           yerr = spec_type_param[1] * df_spec_compton_fit['FLUX_ERR'] * (df_spec_compton_fit['E']**spec_type_param[2]),\
                           fmt='k.', label='Data spectrum')
    axes[0,0].set_yscale('log');axes[0,0].legend();axes[0,0].set_ylabel(spec_dico[spec_type][0]);axes[0,0].set_xscale('log')
    if yscale: axes[0,0].set_ylim(yscale[0],yscale[1])
    if show_residuals:
        axes[0,0].set_xticklabels([]);axes[0,0].set_xticklabels([],minor=True) # remove energy label for minor and major ticks
        axes[1,0].errorbar(x='E', xerr='E_ERR', y=res_type, yerr=res_type+'_ERR', fmt='k.', data=df_spec_compton_fit)
        axes[1,0].axhline(res_dico[res_type][0], color='grey', linestyle='--')
        axes[1,0].set_ylabel(res_dico[res_type][1]);axes[1,0].set_xscale('log')
        
    # to show the energies on (some) minor ticks, and with the full numbers instead of power of 10
    formatter = LogFormatter(labelOnlyBase=False, minor_thresholds=(3, 0.4))
    axes[0,0].yaxis.set_minor_formatter(formatter)
    axes[0,0].yaxis.set_major_formatter(formatter)
    axes[show_residuals,0].xaxis.set_minor_formatter(formatter)
    axes[show_residuals,0].xaxis.set_major_formatter(formatter)
    axes[show_residuals,0].set_xlabel('E (keV)')
    return axes

def fit_spec(spectrum, model, resp, lmfit_params, e_min_fit, e_max_fit):
    '''fit the spectrum with model and response'''
    spectrum.make_fit_ebd(e_min_fit, e_max_fit)
    minner = lm.Minimizer(model.residual_compton, lmfit_params, fcn_args=(spectrum, resp))
    result = minner.minimize(method='leastsq')
    model.make_spec_df(resp, result)
    spectrum.make_spec_df(resp, result, model)
    return result, minner

def fit_spec_alpha(spectrum, model, resp, lmfit_params, e_min_fit, e_max_fit):
    '''
    fit the spectrum with a varying SpiCorr and response 
    ONLY FOR CALIBRATION
    '''
    spectrum.make_fit_ebd(e_min_fit, e_max_fit)
    minner = lm.Minimizer(model.residual_alpha, lmfit_params, fcn_args=(spectrum, resp))
    result = minner.minimize(method='leastsq')
    model.make_spec_df(resp, result)
    spectrum.make_spec_df(resp, result, model)
    return result, minner
