"""
Classes and functions for IBIS/Compton mode polarization analysis
"""

from comibis.utils import *
import lmfit as lm
from scipy.stats import norm, chi2 # norm to convert p-value to sigma, chi2 to find contour levels
from scipy.integrate import dblquad, quad
from scipy.optimize import root_scalar, least_squares
from scipy.stats import rice # Rice distribution
from arviz import hdi # error computation from marginalized posterior
# error computation from likelihood
try:
    no_iminuit = False
    from iminuit import Minuit 
except:
    no_iminuit = True
    print("iminuit library not found! likelihood error won't be available")

import warnings
warnings.filterwarnings("ignore")

rad_to_deg = 180/np.pi
deg_to_rad = np.pi/180

############### Proba/stat ###############

# n-sigma values: 1-sigma -> 68.2%, 2-sigma -> 95.4%, 3-sigma -> 99.7% 

def make_nll_polar(phi0, a0, k2):
    '''returns the -log(likelihood) function, given the data point (phi0, a0) and k2
    phi0 and phi_s are in radian
    '''
    return lambda phi_s, a_s: (k2/2) * (a0**2 + a_s**2 - 2 * a_s * a0 * np.cos(2 * (phi0 - phi_s)))

def make_posterior_polar(pa0, pf0, sig2):
    '''returns the posterior function, given the data point (pa0, pf0) and sig2
    pa0/pa_s are in radian, pf/pf_s in [0,1]
    '''
    return lambda pa_s, pf_s: (pf0 / (np.pi*sig2)) * np.exp(- (pf0**2 + pf_s**2 - 2*pf0*pf_s * np.cos(2 * (pa0 - pa_s)))/(2 * sig2))

def p_to_nsigma(p_unpola):
    '''compute the n-sigma significance from the proba of false detection
    technically this is not correct since polarigram parameters (a,phi0) are not normally distributed but it gives a familiar interpretation of results
    '''
    return norm.isf(p_unpola/2)

def find_error_interval(x_grid, pdf, x_mode, p=.68):
    '''symmetric error at p confidence around the mode
    the cummulative sum is taken above the mode, since the PDF does not have a start for cyclic variables like PA
    (should be re-written with cyclic behavior)
    '''
    upper_grid = x_grid > x_mode # to select upper part of PDF
    upper_pdf_cumsum = np.cumsum(pdf[upper_grid])
    idx_best_p = np.argmin(np.abs(upper_pdf_cumsum - p/2)) # find index where cumsum is half of p
    if idx_best_p == len(upper_pdf_cumsum) - 1:
        print('Warning: The PA error is stuck at upper-bound!')
    return x_grid[upper_grid][idx_best_p] - x_mode # returns distance from the mode

############### Fit polarigram ###############

def phi_to_pa(phi_0, pa_ref):
    '''convert raw phi (in deg) value into Polarization Angle (PA, in deg), with different references:
    PA_ref0 -> ref=0 : -90° < PA < 90°
    PA_ref90 -> ref=90 : 0° < PA < 180°
    '''
    PA = (phi_0 - 90)%180
    if pa_ref=='PA_ref0': return PA - 180 * (PA > 90) # substract 180° if above 90°
    elif pa_ref=='PA_ref90': return PA
    else: return PA

def convert_corr_type(compnorm, spicorr):
    '''naming convention for file saving'''
    if spicorr== 'auto':
        if compnorm == 'auto': return 'alpha'
        else: return 'beta'
    else: return spicorr

def polarigram_model(x, C, a0, phi0):
    '''theoretical polarigram model with phi in degrees'''
    return C * (1 + a0 * np.cos(2 * (x - phi0) * deg_to_rad))

class Polarigram:
    '''polarigram = flux per radian as a function of azimuthal scattering angle, for every energy band found'''

    def __init__(self, df_scw, all_band_names, angle_max):
        self.angle_max, self.df_scw, self.all_band_names = angle_max, df_scw, all_band_names
        self.expo = self.df_scw.EXPO.sum()
        self.n_pola_bands = find_pola_bands(self.df_scw, self.all_band_names)
        self.E_bounds_rbn = np.array([np.int64(b.split('-')) for b in self.all_band_names]) # energy bounds of Compton spec
        self.E_mean = self.E_bounds_rbn.mean(axis=1)
        self.ISOT_START, self.ISOT_END = self.df_scw.ISOT.min(), self.df_scw.ISOT.max()
        self.ISOT_DIFF = self.ISOT_END - self.ISOT_START
        self.ISOT_MID = self.ISOT_START + self.ISOT_DIFF/2 # average of ISOT date in 2 steps
        self.MJD_START, self.MJD_END = self.df_scw.MJD.min(), self.df_scw.MJD.max()
        self.MJD_DIFF, self.MJD_MID = self.MJD_END - self.MJD_START,  (self.MJD_END + self.MJD_START)/2

    def import_prf(self, pulsefrac_dir='compton_responses', pulsefrac_file='comps-prf-1.txt'):
        '''import the polarization response file that contains a100 = a100(E)'''
        pulsefrac_path = f'{pulsefrac_dir}/{pulsefrac_file}'
        self.df_prf = pd.read_csv(pulsefrac_path, delim_whitespace=True, names=['E','compton_event','a100','a100_err'])

    def make_polar(self, spicorr, compnorm=1, df_spicorr_rbn=None, alpha_err=.14):
        '''average all scw to create the polarigrams: 1 for each energy band found
        this will raise an error if empty df
        '''
        self.spicorr = spicorr
        self.compnorm = compnorm
        self.alpha_err = alpha_err
        self.df_spicorr_rbn = df_spicorr_rbn
        all_pola_dico_comp = sum_scw_df(self.df_scw, self.all_band_names, self.spicorr, self.compnorm, self.n_pola_bands, self.df_spicorr_rbn, self.alpha_err)
        self.pola_flux_dico = all_pola_dico_comp['simple']
        spec_list = [[int(e) for e in b.split('-')] + [np.sum(self.pola_flux_dico[b][0])] + [np.sum(self.pola_flux_dico[b][1])] for b in self.pola_flux_dico]
        self.df_spec = pd.DataFrame(spec_list, columns=['Einf','Esup','Flux','Flux_err'])

    def combine_bands(self, Emin, Emax):
        '''find the list of energy intervals between Emin and Emax'''
        bands=[]
        for b in self.all_band_names:
            e1,e2=b.split('-')
            if int(e1)>=int(Emin) and int(e2)<=int(Emax): bands.append(b)
        if len(bands)==0:
            print('No energy bands found !')
        return bands

    def calc_a100(self, Emin, Emax):
        ''' find mean of a100 weighted by F(E)
        this should be re-written with linear algebra a100(E) = a100(E')@B(E',E).F(E).dE / B(E',E)@F(E).dE
        with B(E',E) the re-binning matrix between PRF and df_spec_born, with 0s outise [Emin, Emax]
        '''    
        df_spec_born = self.df_spec[(self.df_spec.Einf>=Emin)&(self.df_spec.Esup<=Emax)]
        a100, Ftot = 0, 0
        dE=self.df_prf.E.iloc[1]-self.df_prf.E.iloc[0] # we suppose that f(E)=a100 has constant step
        for spec in df_spec_born.values:
            E1, E2, F, F_err= spec
            Ftot += F*(E2-E1)
            a100 += F * self.df_prf[(self.df_prf.E>=E1)&(self.df_prf.E<E2)].a100.sum() # convolution of a100 with F
        return (a100*dE)/Ftot, df_spec_born
    
    ##################################################################################################
    #########################################   STATISTICS   #########################################
    ##################################################################################################

    def proba_higher(self, result, a100, verbose=True):
        ''' finds the probability to have higher a0 then the one fitted, knowing the source is unpolarized (a0=0) for any phi
            P(a > a0 measured | a0 source = 0)
        '''
        a0 = result.best_values['a0']
        p_higher = np.exp(- self.pola_param['k2'] * a0**2 / 2)
        sigma_higher = p_to_nsigma(p_higher) # only for checking purpose
        if verbose:
            print('p-value=', p_higher)
        return p_higher, sigma_higher

    def pf_margin(self, pf0, sig2, n_grid=1e5, n_sample=1e7, p_err = .68):
        '''find the asymmetric high/low errors of PF from the marginalized proba
        errors on phi0 are symmetric, but not for a0
        '''
        pf_grid = np.linspace(0, 1, int(n_grid))
        rice_dist = rice.pdf(pf0/np.sqrt(sig2), pf_grid/np.sqrt(sig2)) # x is const, b is variable
        pf_margin_dist = rice_dist/rice_dist.sum()
        pf_mode = pf_grid[np.argmax(pf_margin_dist)]
        # find HDI (= posterior errors)
        samples = np.random.choice(pf_grid, size = int(n_sample), p = pf_margin_dist)
        pf_lo, pf_hi = hdi(samples, hdi_prob = p_err)
        return pf_mode, pf_mode - pf_lo, pf_hi - pf_mode # pf_lo_err, pf_hi_err
    
    def pa_margin(self, pa0, pf0, sig2, pa_ref, n_grid_pa = 3600, n_grid_pf = 100, p_err=.68):
        '''find the symmetric errors of PA from marginalized proba
        PF does not need a fine binning, as it gets summed
        '''
        post_weiss = make_posterior_polar(pa0*deg_to_rad, pf0, sig2) # posterior likelihood function
        if pa_ref=='PA_ref90': pa_grid= np.linspace(0, np.pi, n_grid_pa)
        if pa_ref=='PA_ref0': pa_grid= np.linspace(-np.pi/2, np.pi/2, n_grid_pa)
        pf_grid = np.linspace(0, 1, n_grid_pf)
        post_grid = np.array([[post_weiss(pa, pf) for pa in pa_grid] for pf in pf_grid])
        pa_margin_dist = post_grid.sum(axis=0) 
        pa_margin_dist /= pa_margin_dist.sum() # poster PDF marginalized on a = poster of phi
        pa_mode = pa_grid[np.argmax(pa_margin_dist)] # mode should be same as phi0
        pa_err = find_error_interval(pa_grid, pa_margin_dist, pa_mode, p_err)
        return pa_mode * rad_to_deg, pa_err * rad_to_deg
    
    def error_nll(self, a0, phi0, k2, a100):
        '''find the correct high/low errors (at 1-sigma) from the likelihood
        errors on phi0 are symmetric, but not for a0
        phi0 is converted in rad here
        '''
        m = Minuit(make_nll_polar(phi0 * deg_to_rad, a0, k2), phi_s=phi0*deg_to_rad, a_s = a0)
        m.limits['phi_s'] = (0, 2*np.pi)
        m.limits['a_s'] = (0, a100) # a_s is the true modulation of the source, which cannot exceed a100
        m.errordef = Minuit.LIKELIHOOD # required for likelihood fits
        m.migrad()  # run minimization
        m.minos()  # compute asymmetric errors
        self.pola_param['fval'] = m.fval # record the NLL minimum value, used to plot contours
        return m.merrors # lower errors are negatives
    
    def pf_upper_lim(self, p=0.01):
        '''gives an estimate of upper-limits by inverting the probability formula (Weisskopf+2006)'''
        pf_up = np.sqrt(-2 * np.log(p) * self.pola_param['sig2'])
        return pf_up
    
    def a0_upper_lim(self, p=0.01):
        '''gives an estimate of upper-limits by inverting the probability formula (Weisskopf+2006)'''
        a0_up = np.sqrt(-2 * np.log(p) / self.pola_param['k2'])
        return a0_up

    ##################################################################################################
    #########################################   POLARIGRAM   #########################################
    ##################################################################################################

    def find_polar_param_likelihood(self, result, a100, pa_ref, p_higher, p_det=0.01, p_lim=0.05):
        '''returns Polarization Angle and Fraction from fit result, and upper/lower-limit flags
        use the likelihood function and the iminuit python library
        '''

        a0, phi0, k2 = result.best_values['a0'], result.best_values['phi0'], self.pola_param['k2']
        if p_higher < p_det: # if polarization is detected, (a0,phi0) from fit are kept
            uplim = 0
            merrors_fit = self.error_nll(a0, phi0, k2, a100)
            a0_lo_err, a0_hi_err = -merrors_fit['a_s'].lower, merrors_fit['a_s'].upper
            phi0_rad_err = (merrors_fit['phi_s'].upper - merrors_fit['phi_s'].lower)/2 # take avg because errors on phi are symmetric
            lolim = merrors_fit['a_s'].at_upper_limit # "at_upper_limit" checks if the param hits the upper bound (here a100) = what scientists call a lower-limit
            if lolim:
                a0 = a0 - a0_lo_err # a0 is now defined as the lower bound at 68%
        
        else: # otherwise use upper-limit, which does not need MLE
            uplim=1
            lolim= 0
            a0 = self.a0_upper_lim(p=p_lim)
            a0_lo_err, a0_hi_err = a100/10, a100/10 # to have the length of upper-limit arrows as 10% when plotting 
            phi0, phi0_rad_err = np.nan, 0. # in order to not plot PA for non-detection
        
        PA, PA_err = phi_to_pa(phi0, pa_ref), phi0_rad_err * rad_to_deg
        PF, PF_lo_err, PF_hi_err = a0/a100, a0_lo_err/a100, a0_hi_err/a100
        return PA, PA_err, PF, PF_lo_err, PF_hi_err, uplim, lolim


    def find_polar_param_margin(self, result, a100, pa_ref, p_higher, p_det=0.01, p_lim=0.05):
        '''returns Polarization Angle and Fraction and upper/lower-limit flags
        use the marginalized posterior probability (longer computation time)
        '''
        pa0 = phi_to_pa(result.best_values['phi0'], pa_ref)
        pf0 = result.best_values['a0']/a100
        print(f'initial PF = {pf0*100:.2f} %')

        if p_higher < p_det: # if polarization is detected, (a0,phi0) from fit are kept
            uplim = 0
            PF, PF_lo_err, PF_hi_err = self.pf_margin(pf0, self.pola_param['sig2'])
            PA, PA_err = self.pa_margin(pa0, pf0, self.pola_param['sig2'], pa_ref)
            lolim = PF + PF_hi_err > 1 - 1e-3 # upper-error stuck at 1
            if lolim:
                PF = PF - PF_lo_err # PF is now defined as the lower bound at 68%

        else: # otherwise use upper-limit
            uplim=1
            lolim= 0
            PF = self.pf_upper_lim(p=p_lim)
            PF_lo_err, PF_hi_err = .1, .1 # to have the length of upper-limit arrows as 10% when plotting 
            PA, PA_err = np.nan, 0. # in order to not plot PA for non-detection
        
        return PA, PA_err, PF, PF_lo_err, PF_hi_err, uplim, lolim


    def fit_pola(self, bands, p_det=0.01, p_lim=0.05, folded=1, weighted=True, pa_ref=None, verbose=True, article=False, plot_0=True, errortype='margin'):
        '''fit the total polarigram given by a list of energy bands, then deduce PA/PF values
        the best parameters and errors
        '''
        self.pa_ref = pa_ref
        
        ### build polarigram
        Emin, Emax = bands[0].split('-')[0], bands[-1].split('-')[1] # find min/max energy of the bands
        if verbose: print(f'Polarized flux in {Emin} - {Emax} keV band')
        # add the different bands together
        y=np.zeros(self.n_pola_bands)
        y_err=np.zeros(self.n_pola_bands)
        for b in bands:
            y+=self.pola_flux_dico[b][0]
            y_err+=np.array(self.pola_flux_dico[b][1])**2
        y_err=np.sqrt(y_err)

        # dividing count-rates by bin angle width allows to have binning-invariant polarigrams (similar to flux/keV for spectra)
        # folded=1 is the default, folded=0 means the polarigram was not folded on [0,pi] during ic_PolaIma computation (for testing purpose)
        bin_width= (2-folded)*np.pi/self.n_pola_bands # bin width in radian
        y = y/(bin_width*(1+folded))
        y_err = y_err/(bin_width*(1+folded))

        SNR = y.sum()/np.sqrt(np.sum(y_err**2))
        a100, df_spec_born = self.calc_a100(int(Emin), int(Emax)) # compute a100 for the given energy range

        pola_width = ((2-folded)*180/self.n_pola_bands) # 180° if folded, 360° if not 
        x_pola = np.array([pola_width*(i+1/2) for i in range(self.n_pola_bands)]) # convert pola bin to angle phi

        ### fit polarigram
        mean_y=np.mean(y)
        par_pola = lm.Parameters()
        par_pola.add('C',value=mean_y)
        if folded: par_pola.add('a0',value=(np.max(y)-mean_y)/mean_y)
        else: par_pola.add('a0',value=(np.max(y)-mean_y)/mean_y,min=0)
        max_phi = x_pola[np.argmax(y)] # angle of maximum flux
        par_pola.add('phi0',value=max_phi) # no constraint on phi0
        gmodel = lm.Model(polarigram_model)
        
        # fit with constant to compare Chi2
        par_nopola= lm.Parameters()
        par_nopola.add('c',value=mean_y)
        gmodel_nopola = lm.models.ConstantModel()

        if weighted: # does not change result since y_err is pretty much constant, but gives chi2 value
            result = gmodel.fit(y, par_pola, x=x_pola, weights=1/y_err)
            result_nopola = gmodel_nopola.fit(y, par_nopola, x=x_pola, weights=1/y_err)
            print(f'reduced chi2: pola={result.redchi:.2f}, no pola={result_nopola.redchi:.2f}')
        else: result = gmodel.fit(y, par_pola, x=x_pola)
        self.last_result = result # store last polarigram fit for de-bugging and testing
        # print(f'y = {mean_y:.2e} ± {y_err.mean():.2e}')
        self.pola_param = {'SNR':SNR, 'redchi2':result.redchi, 'k2':(self.n_pola_bands * (mean_y/y_err.mean())**2)/2, 'a100':a100}
        self.pola_param['sig2'] = 1/(self.pola_param['k2'] * a100**2)
        # find all relevant polarization parameters and save them in pola_param dico
        p_higher, sigma_higher = self.proba_higher(result, a100, verbose=False) # find probability and n-sigma estimate
        flux, flux_err = df_spec_born.Flux.sum(), np.sqrt(np.sum(df_spec_born.Flux_err**2))
        mdp99 = 4.29 / (a100 * np.sqrt(flux * self.expo)) # MDP (minimum detected polarization) at 99% confidence

        ### best values and error calculation
        if errortype=='margin' or no_iminuit:
            if no_iminuit: print('no iminuit library imported, reverting to marginalized errors')
            PA, PA_err, PF, PF_lo_err, PF_hi_err, uplim, lolim = self.find_polar_param_margin(result, a100, pa_ref, p_higher, p_det, p_lim)
        else:
            PA, PA_err, PF, PF_lo_err, PF_hi_err, uplim, lolim = self.find_polar_param_likelihood(result, a100, pa_ref, p_higher, p_det, p_lim)
        PF_err = (PF_lo_err + PF_hi_err)/2 # for convinience, as the low/high error are usually close

        self.pola_param.update({'PA':PA, 'PA_err':PA_err, 'PF':PF, 'PF_err':PF_err, 'uplim':uplim, 'lolim':lolim, 'mdp99':mdp99,\
                      'p_higher':p_higher, 'sigma_higher':sigma_higher, 'PF_hi_err':PF_hi_err, 'PF_lo_err':PF_lo_err})

        ### Plot the polarigram and print fit parameters
        if verbose: 
            x_model = np.linspace(np.min(x_pola), np.max(x_pola), num=300)
            x_model_360 = np.linspace(np.min(x_pola), 360, num=300)
            fig, ax = plt.subplots(figsize=(8,5))
            label_font=15

            if plot_0:
                # ax.axhline(y=0,color='k')
                ax.axhline(y=result.best_values['C'],color='grey',ls='--')
                if uplim:
                    label_fit = f"PF < {PF*100:.1f} %" # upper-limit = no PA
                else:
                    label_fit = f"PA = {PA:.1f} ± {PA_err:.1f} °\n"
                    if lolim: label_fit += f"PF > {PF*100:.1f} %"
                    else: label_fit += f"PF = {PF*100:.1f} ± {PF_err*100:.1f} %" # \nSNR = {SNR:.1f}
            else:
                label_fit = fr"$\phi_0$ = {result.best_values['phi0']:.1f} ± {result.params.get('phi0').stderr:.1f} °"+\
                '\n'+ rf"$a_0$ = {result.best_values['a0']*100:.1f} ± {result.params.get('a0').stderr*100:.1f} %"

            label_data=''
            
            if folded: # duplicates the [0,pi] polarigram to be on [0,2*pi]
                ax.errorbar(x=np.concatenate((x_pola,(x_pola+180))), y=np.concatenate((y,y)), 
                            yerr=np.concatenate((y_err,y_err)), xerr=pola_width/2, fmt='k.', label=label_data) 
                ax.plot(x_model_360, result.eval(x=x_model_360), 'r-', label=label_fit)
            else:
                ax.errorbar(x=x_pola, y=y, yerr=y_err,xerr=pola_width/2, fmt='.', label=label_data)
                ax.plot(x_model, result.eval(x=x_model), 'r-', label=label_fit)
            ax.xaxis.label.set_size(label_font);ax.yaxis.label.set_size(label_font);ax.tick_params(which='both', labelsize=label_font)
            ax.set_xlabel(r'$\phi$ (°)');ax.set_ylabel(r'Counts s$^{-1}$ rad$^{-1}$')
            plt.legend(loc='upper right',fontsize=label_font)

            # print(f'mean flux = {np.mean(y):.1e} ± {np.std(y):.1e} ct/s/rad ') # 'a0*C={result.best_values['a0']*result.best_values['C']}'
            print(f'total flux = {flux:.1e} ± {flux_err:.1e} ct/s')
            print(f'SNR = {SNR:.1f} p-value = {p_higher:.1e} ({sigma_higher:.1f}-sigma)')
            # print(f"a0 = {result.best_values['a0']:.3f},  a100 = {a100:.3f}")
            if uplim: print(f'PF < {PF*100:.1f} % (upper-limit at {(1 - p_lim)*100:.3g} %)')
            if lolim: print(f'PA = {PA:.1f} ± {PA_err:.1f} °\nPF > {PF*100:.1f} % (lower-limit at 68 %)')
            else: print(f'PA = {PA:.1f} ± {PA_err:.1f} °\nPF = {PF*100:.1f} (+{PF_hi_err*100:.1f}/-{PF_lo_err*100:.1f}) %')

        else:
            ax = None
        return self.pola_param, ax
    

    def plot_nll(self, n_grid = 400, list_p_contour=[0.68, 0.99], cmap='magma', vmax_fact=1., polar_plot=False, only_contour=False):
        '''plot the (delta) negativa log-likelihood (NLL) shifted by K
        add contours based on NLL/chi2 relation
        smaller vmax_fact will increase contrast around the NLL minimum
        '''
        a0, phi0, k2 = self.last_result.best_values['a0'], self.last_result.best_values['phi0'], self.pola_param['k2']
        a100 = self.pola_param['a100']
        nll_fct = make_nll_polar(self.pola_param['PA'] * deg_to_rad, a0, k2) # make new nll centered on PA instead of phi
        if self.pa_ref =='PA_ref90': phi_min, phi_max = 0, np.pi
        elif self.pa_ref =='PA_ref0': phi_min, phi_max = -np.pi/2, np.pi/2
        phi_grid = np.linspace(phi_min, phi_max, n_grid)
        a_grid = np.linspace(0, a100, n_grid)
        x1, x2 = np.meshgrid(phi_grid, a_grid)
        NLL_grid = nll_fct(x1, x2)

        # compute contour levels at p, based on formula: DeltaNLL = 0.5 * DeltaChi2, for 2 dof (see https://statproofbook.github.io/P/ci-wilks.html)
        nll_level_list = [NLL_grid.min() + 0.5 * chi2.ppf(cl, df = 2) for cl in list_p_contour]
        # nll_level_list = [self.pola_param['fval'] + 0.5 * chi2.ppf(cl, df = 2) for cl in list_p_contour]

        if polar_plot: # radar plot (à la IXPE)
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(projection='polar') # works in radian
            ax.set_thetalim(phi_min, phi_max)
            if self.pa_ref =='PA_ref90':ax.set_theta_zero_location('W')
            elif self.pa_ref =='PA_ref0':ax.set_theta_zero_location('N')
            
            if only_contour:
                cnt = ax.contour(phi_grid, 100*a_grid/a100 , NLL_grid, levels=nll_level_list, colors='black', linestyles='--')
            else:
                cnt = ax.contour(phi_grid, 100*a_grid/a100 , NLL_grid, levels=nll_level_list, colors='white', linestyles='--')
                cb = ax.pcolormesh(phi_grid, 100*a_grid/a100 , NLL_grid, edgecolors='face', cmap=cmap, vmin=0, vmax=vmax_fact*np.mean(NLL_grid))
                plt.colorbar(cb, ax=ax, label=r'$\Delta$NLL', orientation='horizontal', location='top', pad=-.1,)

            ax.clabel(cnt, inline=True, fmt={nll_level_list[i]: f" {list_p_contour[i] * 100:.5g} % " for i in range(len(list_p_contour))}, fontsize=10)
            # ax.scatter(self.pola_param['PA']*deg_to_rad, 100 * self.pola_param['PF'], marker='+', color='g')
            ax.errorbar(x=self.pola_param['PA']*deg_to_rad, y=100 * self.pola_param['PF'], yerr=100 * self.pola_param['PF_err'], xerr=self.pola_param['PA_err']*deg_to_rad,
                            fmt='.', ecolor='g', c='g')

            ax.set_theta_direction(-1) # clock-wise
            # ax.set_xlabel('PA');ax.xaxis.set_label_coords(.5, .85)
            ax.set_ylabel('PF (%)', rotation=0);ax.yaxis.set_label_coords(0.5,.15)
            ax.yaxis.set_tick_params(labelright=True, labelleft=True)
            ax.xaxis.set_tick_params(pad=10) # move theta ticks away from border
            ax.grid(alpha=.5)

        else: # square plot
            fig, ax=plt.subplots(1,1,figsize=(8,8))
            cb=ax.imshow(NLL_grid, extent=[0, 180, 0., 100], aspect=1.8, origin='lower', cmap=cmap, vmax = vmax_fact * NLL_grid.mean())
            cnt = ax.contour(phi_grid*rad_to_deg, 100*a_grid/a100 , NLL_grid, levels=nll_level_list, colors='w', linestyles='--')
            ax.clabel(cnt, inline=True, fmt={nll_level_list[i]: f" {list_p_contour[i] * 100:.5g} % " for i in range(len(list_p_contour))}, fontsize=10)
            ax.scatter(self.pola_param['PA'], 100 * self.pola_param['PF'], marker='+', color='g')
            ax.set_xlabel('PA (°)');ax.set_ylabel('PF (%)')
            plt.colorbar(cb, ax=ax, label=r'$\Delta$NLL') # (\phi_s, a_s)
        return ax

    #########################################   Polarization by energy   #########################################

    def pola_espectrum(self, energy_bands, pa_ref='PA_ref90', SNR_threshold=12, p_det=0.01, p_lim=0.05, weighted=1, verbose=0, errortype='margin'):
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
                pola_param, _ = self.fit_pola(bands_inside, p_det=p_det, p_lim=p_lim, folded=1, weighted=weighted, pa_ref=pa_ref, verbose=1, article=0, errortype=errortype)
                pola_param.update({'Emin':Emin, 'Emax':Emax})
                plt.show()

                if  pola_param['SNR'] > SNR_threshold: # if the SNR is sufficient for meaningful detection
                    list_pola_param.append(pola_param)

            except ValueError: # this can happen for empty energy bands or negative flux
                print(f'nan value in fit for {Emin} - {Emax} keV band')
                continue

        # create df with parameters of each polarigram
        self.df_pola_param=pd.DataFrame(list_pola_param) # build data frame from list of polar dict
        self.df_pola_param['E_mean'] = self.df_pola_param.apply(lambda x:(int(x.Emin)+int(x.Emax))/2, axis=1 )
        self.df_pola_param['dE'] = self.df_pola_param.apply(lambda x:(int(x.Emax)-int(x.Emin))/2, axis=1 )
        # convert PFs to percent
        self.df_pola_param[['PF_pct','PF_pct_err','PF_pct_hi_err','PF_pct_lo_err']] = self.df_pola_param[['PF','PF_err','PF_hi_err','PF_lo_err']] * 100
        self.df_pola_param['inst'] = 'IBIS'
        return self.df_pola_param
    
    def save_pola_espectrum(self, save_dir, save_file):
        path = f'{save_dir}/{save_file}.csv'
        self.df_pola_param.to_csv(path)
        print(f'polar spectrum saved at {path}')

    def plot_pola_espectrum(self, plot_scale='lin', plot_percent=True, plot_grid=0, with_snr=0, fmt='k.'):
        '''plot the PA/PF as a function of energy'''
        if with_snr: fig,ax=plt.subplots(3,1,figsize=(10,10))
        else: fig,ax=plt.subplots(2,1,figsize=(10,8))
        PA_mean, PF_mean = self.df_pola_param[['PA','PF']].mean()
        PA_std, PF_std = self.df_pola_param[['PA','PF']].std()
        # Polarization Angle (PA) = upper plot
        ax[0].errorbar('E_mean', 'PA', yerr='PA_err', xerr='dE', fmt=fmt, data=self.df_pola_param, label=None)
        ax[0].axhline(PA_mean, linestyle='--', c='k', label=f'Average PA = {PA_mean:.1f} ± {PA_std:.1f} °')
        ax[0].set_ylabel('PA (°)')
        # ax[0].tick_params(labelbottom=False) 
        ax[0].legend(loc='upper right')#;ax[1].legend()#

        if plot_percent:
            ax[1].errorbar('E_mean','PF_pct', yerr = self.df_pola_param[['PF_pct_lo_err', 'PF_pct_hi_err']].to_numpy().T, xerr='dE', fmt=fmt, data=self.df_pola_param,\
                label=None, uplims=self.df_pola_param['uplim'], lolims=self.df_pola_param['lolim'])
            # ax[1].errorbar('E_mean','PF_pct',yerr='PF_pct_err',xerr='dE',fmt=fmt, data=self.df_pola_param,label=None, uplims=self.df_pola_param['uplim'], lolims=self.df_pola_param['lolim'])
        else:
            ax[1].errorbar('E_mean','PF',yerr=self.df_pola_param[['PF_lo_err', 'PF_hi_err']].to_numpy().T, xerr='dE', fmt=fmt, 
                           data=self.df_pola_param,label=None, uplims=self.df_pola_param['uplim'], lolims=self.df_pola_param['lolim'])

        #ax[1].axhline(PF_mean,linestyle='--',c='k',label=f'Average PF = {PF_mean:.2f} ± {PF_std:.2f}')
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
            if plot_grid==True: ax[i].grid(True)
            if plot_scale=='log': ax[i].set_xscale('log')


##################################################################################################
#####################################  Selection functions   #####################################
##################################################################################################

def pola_select(df_scw, src_info_list, band_names, angle_max, column_name, select_list, spicorr, compnorm, alpha_err, prf_file_name, df_spicorr_rbn, resp_dir,
                errortype='margin', SNR_threshold=12, weighted=1, p_det=0.01, p_lim=0.05, verbose=1):
    ''' find best fit PA/PF for each selection given in the select_list and returns df with all parameters
        ex for column_name='YEAR': select_list= [[2010,2011], [2012,2013]]
    '''
    Emin, Emax = src_info_list[0]
    states_dico = src_info_list[1]
    pa_ref=src_info_list[2]
    states_mjd = src_info_list[3]
    list_pola_param = [] # list of polar parameters dictionaryies
    df_scw = scw_selection(df_scw, angle_max=angle_max)

    for i,sel in enumerate(select_list):
        df_scw_sel=df_scw[df_scw[column_name].isin(sel)] # select scw from the i-th selection sub-list
        # create polarigram from df
        all_polarigrams = Polarigram(df_scw_sel, band_names, angle_max) # initialize polarigram
        all_polarigrams.import_prf(pulsefrac_dir=resp_dir, pulsefrac_file=prf_file_name) # import polarization response (a100)

        if verbose:
            print('\n\n****** ------------------------------------------------------------------------- ******\n')
            print('Polarigram of {} {} - {}'.format(column_name, sel[0], sel[-1]))
            print(f'Exposure = {int(all_polarigrams.expo/1000)} ks')
            print(f'Start = MJD {all_polarigrams.MJD_START} Stop = MJD {all_polarigrams.MJD_END}')
            if spicorr=='auto': print('Average SpiCorr = {:.3f}'.format(df_scw_sel.spicorr.mean()))
        try: # average all scw and create polarigrams
            all_polarigrams.make_polar(spicorr, compnorm, df_spicorr_rbn, alpha_err)
        except Exception:
            print('Empty data frame !')
            continue

        try: # fit polarigram and save parameters
            bands_inside = all_polarigrams.combine_bands(Emin, Emax) # find the smaller energy band(s) inside the given energy band
            print(bands_inside)
            pola_param, _ = all_polarigrams.fit_pola(bands_inside, p_det=p_det,p_lim=p_lim, folded=1, weighted=weighted, pa_ref=pa_ref, verbose=verbose, article=0, errortype=errortype)
            
            plt.show()
            pola_param.update({'ISOT_MID':all_polarigrams.ISOT_MID, 'ISOT_ERR':all_polarigrams.ISOT_DIFF/2,\
                               'MJD_MID':all_polarigrams.MJD_MID, 'MJD_ERR':all_polarigrams.MJD_DIFF/2, 'EXPO':all_polarigrams.expo})
            if  pola_param['SNR'] > SNR_threshold: # if the SNR is sufficient for meaningful detection
                list_pola_param.append(pola_param)
        except ValueError: # this can happen for empty energy bands or negative flux
            print('Nan value in fit')

    df_pola_param=pd.DataFrame(list_pola_param) # build data frame from list of polar dict
    df_pola_param[['PF_pct','PF_pct_err','PF_pct_hi_err','PF_pct_lo_err']] = df_pola_param[['PF','PF_err','PF_hi_err','PF_lo_err']] * 100
    return df_pola_param

    
def plot_pola_select(df_pola_param, src_info_list, plot_percent=True, with_proba=0, with_snr=0, with_expo=0, jet_angle=None, date_type='MJD'):
    '''plot polarization as a function of time'''

    Emin, Emax = src_info_list[0]
    states_mjd=src_info_list[3]
    PA_mean, PF_mean = df_pola_param[['PA','PF']].mean()
    PA_std, PF_std = df_pola_param[['PA','PF']].std()

    fig,ax=plt.subplots(2+with_snr+with_proba+with_expo, 1, figsize=(10,8))
    ax[0].errorbar(x=f'{date_type}_MID', xerr=f'{date_type}_ERR', y='PA', yerr='PA_err', fmt='k.', data=df_pola_param, label=f'{Emin} - {Emax} keV')
    # ax[0].axhline(PA_mean, linestyle='--', c='k', label=f'Average PA = {PA_mean:.1f} ± {PA_std:.1f} °')
    if jet_angle: ax[0].axhspan(jet_angle[0], jet_angle[1] ,color='green', label='Radio jet axis', alpha=0.4)
    ax[0].set_ylabel('PA (°)')
    ax[0].tick_params(labelbottom=False)
    ax[0].xaxis.label.set_size(15);ax[0].yaxis.label.set_size(15)
    ax[0].tick_params(which='both', labelsize=15)
    # ax[0].set_xticks(rotation=30);ax[1].set_xticks(rotation=30);ax[2].set_xticks(rotation=30)

    if plot_percent:
        ax[1].errorbar(x=f'{date_type}_MID', xerr=f'{date_type}_ERR', y='PF_pct', yerr=df_pola_param[['PF_pct_lo_err', 'PF_pct_hi_err']].to_numpy().T,\
                       uplims=df_pola_param['uplim'], lolims=df_pola_param['lolim'], fmt='k.', data=df_pola_param,label=None)
    else:
        ax[1].errorbar(x=f'{date_type}_MID', xerr=f'{date_type}_ERR', y='PF', yerr=df_pola_param[['PF_lo_err', 'PF_hi_err']].to_numpy().T,\
                       uplims=df_pola_param['uplim'], lolims=df_pola_param['lolim'], fmt='k.', data=df_pola_param,label=None)
    #ax[1].axhline(df_pola_param.PF.mean(),linestyle='--',c='r',label='Average PF = {0:.2f} ± {1:.2f}'.format(df_pola_sel.PF.mean(),df_pola_sel.PF.std()))
    ax[1].xaxis.label.set_size(17);ax[1].yaxis.label.set_size(15)
    ax[1].tick_params(which='both', labelsize=15)
    ax[1].set_ylabel('PF' + plot_percent*' (%)')

    if states_mjd:
        for state in states_mjd.keys():
            state_infos=states_mjd[state]
            if state_infos[0]:
                ax[0].axvline(x=state_infos[0], color='red', ls='--', alpha=0.8)
                ax[1].axvline(x=state_infos[0], color='red', ls='--', alpha=0.8)
            ax[0].text(state_infos[1],PA_mean+10,state,fontsize=15,color='r')

    if with_snr:
        ax[2].errorbar(x=f'{date_type}_MID', xerr=f'{date_type}_ERR', y='SNR', fmt='gs', data=df_pola_param, label=None)
        ax[2].set_ylabel('SNR')
        ax[2].legend()#;ax[2].grid(True)
    
    if with_proba:
        ax[1].tick_params(labelbottom=False)
        ax[2].errorbar(x=df_pola_param[f'{date_type}_MID'], xerr=df_pola_param[f'{date_type}_ERR'], y=df_pola_param['p_higher'] * 100, fmt='gs', label=None)
        ax[2].set_ylabel('p-value (%)')
        ax[2].set_xlim(ax[1].get_xlim())
        ax[2].set_ylim(bottom=0.)
    
    if with_expo:
        ax[2].errorbar(x=f'{date_type}_MID', xerr=f'{date_type}_ERR', y='Expo (ks)',fmt='gs', data=df_pola_param,label=None)
        ax[2].set_ylabel('Exposure (ks)')
        ax[2].set_xlim(ax[1].get_xlim())
        ax[2].grid(True)
    ax[-1].set_xlabel(f'Date ({date_type})')

    plt.xticks(rotation=30)
    fig.tight_layout()
    #ax[0].set_title('{0} in the {1}-{2} keV band'.format(src,Emin,Emax))
    ax[0].set_xlim(ax[1].get_xlim())
    ax[1].set_ylim(bottom=0.)
    ax[0].legend();ax[1].legend()#;ax[0].grid(True);ax[1].grid(True)
    return ax