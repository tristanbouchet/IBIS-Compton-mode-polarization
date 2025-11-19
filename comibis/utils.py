"""
Functions useful for spectro/polar/lightcurve analysis of IBIS Compton mode
(this should eventually be re-written as a superclass for polar and spec)
"""

from astropy.time import Time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

kev_to_erg = 1.60218e-9 # erg/keV

def iso_to_mjd(row):
    '''convert ISOT to MJD'''
    date=Time(str(row.ISOT), format='iso') 
    date.format='mjd' # convert to ISO (YYYY-MM-DD)
    return date.value # convert to plot-able dates
    
def mjd_to_isot(row):
    '''convert MJD values into plot-able ISO dates'''
    isot=Time(row.MJD, format='mjd') # charge MJD dates with astropy
    isot.format='isot' # convert to ISO (YYYY-MM-DD)
    return np.datetime64(isot.value) # convert to plot-able dates

############### DataFrame managment ###############

# class DataIBIS:
#     def __init__(self, df_scw, all_band_names, angle_max):
#         self.angle_max, self.df_scw, self.all_band_names = angle_max, df_scw, all_band_names
#         self.expo = self.df_scw.EXPO.sum()
#         self.n_pola_bands = find_pola_bands(self.df_scw, self.all_band_names)
#         self.E_bounds_rbn = np.array([np.int64(b.split('-')) for b in self.all_band_names]) # energy bounds of Compton spec
#         self.ISOT_START, self.ISOT_END = self.df_scw.ISOT.min(), self.df_scw.ISOT.max()
#         self.ISOT_DIFF = self.ISOT_END - self.ISOT_START
#         self.ISOT_MID = self.ISOT_START + self.ISOT_DIFF/2 # average of ISOT date in 2 steps
#         self.MJD_START, self.MJD_END = self.df_scw.MJD.min(), self.df_scw.MJD.max()
#         self.MJD_DIFF, self.MJD_MID = self.MJD_END - self.MJD_START,  (self.MJD_END + self.MJD_START)/2


def combine_df(src, scw_file_name_list, saved_pola_folder,  save_name=None):
    df_pola_scw=pd.concat([pd.read_csv(f'{saved_pola_folder}/{src}_{scw_file_name}_pola.csv',
                    dtype={'SCW':'str','REV':'str'}) for scw_file_name in scw_file_name_list]) # charge all df in a list and concat them
    if save_name: df_pola_scw.to_csv(f'{saved_pola_folder}/{src}_{save_name}_pola.csv')
    return df_pola_scw

def save_pre_2020(src, scw_file_name_list, saved_pola_folder, save_name,year=2020):
    '''useful to remove data with wrong IC'''
    df_pola_scw=combine_df(src,scw_file_name_list, save_name=None)
    df_pola_scw['ISOT']=df_pola_scw.apply(lambda x:np.datetime64(x.ISOT),axis=1)
    df_pola_scw['YEAR']=df_pola_scw.apply(lambda x:x.ISOT.year,axis=1)
    df_pre2020 = df_pola_scw[df_pola_scw.YEAR<year]
    df_pre2020.to_csv(f'{saved_pola_folder}/{src}_{save_name}_pola.csv')

def charge_df(src, scw_file_name_list, saved_pola_folder='saved_pola_df', df_spicorr=None,
              plot=True, verbose=True):
    '''charge csv files of a source into dataframe'''
    df_pola_scw = combine_df(src, scw_file_name_list, saved_pola_folder, save_name=None)
    all_band_names=pd.unique([c.split('_')[0] for c in df_pola_scw.columns if '-' in c])
    df_pola_scw['REVINT']=np.int64(df_pola_scw.REV)
    df_pola_scw['scw_id']=df_pola_scw.apply(lambda x:x.SCW[:-4],axis=1)
    df_pola_scw['SCW_ORDER']=df_pola_scw.apply(lambda x:int(x.SCW[4:8]),axis=1)
    df_pola_scw['ISOT']=df_pola_scw.apply(lambda x:np.datetime64(x.ISOT),axis=1)
    df_pola_scw['YEAR']=df_pola_scw.apply(lambda x:x.ISOT.year,axis=1)
    df_pola_scw['MONTH']=df_pola_scw.apply(lambda x:x.ISOT.month,axis=1)
    if df_spicorr is not None: # add a time-dependent spicorr and compnorm value in the df according to the year
        calib_col = [c.split('_')[0] for c in df_spicorr.columns if '_START' in c][0] # infer the column (REV, YEAR, ...) used for calibration validity
        df_pola_scw['spicorr']= df_pola_scw.apply(lambda x:df_spicorr.spicorr[(x[calib_col]>=df_spicorr[calib_col+'_START'])&(x[calib_col]<=df_spicorr[calib_col+'_END'])].iloc[0],axis=1)
        df_pola_scw['spicorr_err']= df_pola_scw.apply(lambda x:df_spicorr.spicorr_err[(x[calib_col]>=df_spicorr[calib_col+'_START'])&(x[calib_col]<=df_spicorr[calib_col+'_END'])].iloc[0],axis=1)
        # df_pola_scw['compnorm']= df_pola_scw.apply(lambda x:df_spicorr.compnorm[(x[calib_col]>=df_spicorr[calib_col+'_START'])&(x[calib_col]<=df_spicorr[calib_col+'_END'])].iloc[0],axis=1)
        # find index of df_spicorr with right time interval
        df_pola_scw['calib_idx'] = df_pola_scw.apply(lambda x:df_spicorr[(x[calib_col]>=df_spicorr[calib_col+'_START'])&(x[calib_col]<=df_spicorr[calib_col+'_END'])].index[0], axis=1)
    if verbose:
        print(f'{len(df_pola_scw)} scw found')
        print(all_band_names)
    if plot:
        if len(df_pola_scw.YEAR.unique())<3: df_pola_scw.groupby('REV').REV.count().plot(kind="bar") # if only 1 year of data, plots the revs histogram
        else: df_pola_scw.groupby('YEAR').YEAR.count().plot(kind="bar")
        plt.ylabel('Number of scw')
        plt.plot()
    return df_pola_scw, all_band_names

def rebin_compcorr(df, all_band_names):
    '''update calibration dataframe with re-binned compcorr values to match data energy bands'''
    E_bounds_rbn = np.array([np.int64(b.split('-')) for b in all_band_names])
    E_mean = E_bounds_rbn.mean(axis=1)
    compcorr_col = [c for c in df.columns if 'compcorr_' in c]
    E_compcorr_bounds = np.array([np.int64(c.split('_')[1:]) for c in compcorr_col])
    # rbn_compcorr_matrix = np.array([(E_mean>=erbn[0])&(E_mean<erbn[1]) for erbn in E_compcorr_bounds]).astype('int')

    # build re-binning matrix
    rbn_compcorr_matrix=np.zeros(shape=(len(E_compcorr_bounds),len(E_bounds_rbn)))
    for i in range(len(E_compcorr_bounds)):
        for j in range(len(E_bounds_rbn)):
            if E_bounds_rbn[j][0]>=E_compcorr_bounds[i][0] and E_bounds_rbn[j][1]<=E_compcorr_bounds[i][1]:
                rbn_compcorr_matrix[i][j] = 1
            elif E_bounds_rbn[j][0]<=E_compcorr_bounds[i][0] and E_bounds_rbn[j][1]>=E_compcorr_bounds[i][1]:
                rbn_compcorr_matrix[i][j] = (E_compcorr_bounds[i][1] - E_compcorr_bounds[i][0]) / (E_bounds_rbn[j][1] - E_bounds_rbn[j][0])

    compcorr_mat = df[compcorr_col].to_numpy()
    compcorr_mat_rbn = np.array([cc@rbn_compcorr_matrix for cc in compcorr_mat]).T

    for i in range(len(all_band_names)):
        df['rbn_compcorr_'+all_band_names[i]] = compcorr_mat_rbn[i]
    return df, rbn_compcorr_matrix

def find_pola_bands(df, all_band_names):
    '''infer the number of pola bands from the column of a df'''
    return np.max([int(c.split('_')[1]) for c in df.columns if all_band_names[0] in c])+1

def scw_selection(df, angle_max=None, rev_list=None, start_date=None, end_date=None, date_type='ISOT'):
    df = df.sort_values('ISOT',ignore_index=True)
    if angle_max: df=df[df.ANGLE<angle_max]
    if start_date: df=df[df[date_type]>=start_date]
    if end_date: df=df[df[date_type]<=end_date]
    if rev_list: df=df[df['REV'].isin(rev_list)]
    return df

def select_by_scw(df, scw_folder='', scw_file_list=[]):
    ''' select scw in the df by their names, using an external file containing a list of scw (with or without the .001)'''
    all_scw=[] 
    for scw_file in scw_file_list:
        with open(f'{scw_folder}/{scw_file}','r') as f:
            all_scw+=f.read().splitlines()
    all_scw=[scw.split('.')[0] for scw in all_scw]
    return df[df.scw_id.isin(all_scw)]

def count_error_auto(x, name, compcorr_spec, alpha_err):
    '''propagate the errors on Compton flux = alpha * C - beta * S'''
    C_sig = compcorr_spec[x.calib_idx] * x[name+'_err'] * x.EXPO
    alpha_sig = alpha_err * compcorr_spec[x.calib_idx] * x[name] * x.EXPO
    S_sig = x['spicorr'] * x[name+'_spurf_err'] * x.ISGRI_EXPO
    beta_sig = x['spicorr_err'] * x[name+'_spurf'] * x.ISGRI_EXPO
    # if '-250' in name: print(f"{x.SCW}: C_sig = {C_sig:.2e} alpha_sig = {alpha_sig:.2e} S_sig = {S_sig:.2e} beta_sig = {beta_sig:.2e}")
    return np.sqrt(C_sig**2 + alpha_sig**2 + S_sig**2 * beta_sig**2)

def sum_scw_df(df, all_band_names, spicorr, compnorm, n_pola_bands, df_spicorr_rbn=None, alpha_err=.14):
    ''' compute the average flux/error from th scw rate df '''
    full_expo_time=df.EXPO.sum()
    all_pola_dico={'simple':{},'simple_clean':{}, 'pond':{}, 'pond_clean':{}}
    if len(df)==0:
        raise Exception('No data selected! (empty df)')
    
    for b in all_band_names:
        flux, flux_err=[],[]

        for p in range(n_pola_bands):
            name=b+'_'+str(p)
            if spicorr=='auto': # with a time dependent spicorr value
                if compnorm=='auto':
                    compcorr_spec = df_spicorr_rbn['rbn_compcorr_'+b]
                    df[name+'_count']= df.apply(lambda x:compcorr_spec[x.calib_idx] * x[name] * x.EXPO - x['spicorr'] * x[name+'_spurf'] * x['ISGRI_EXPO'], axis=1)
                    df[name+'_count_err']= df.apply(count_error_auto, args=(name, compcorr_spec, alpha_err) ,axis=1)
                    # df[name+'_count_err']= df.apply(lambda x:np.sqrt((compcorr_spec[x.calib_idx]*x[name+'_err']*x.EXPO)**2 + (x['spicorr']*x[name+'_spurf_err']*x['ISGRI_EXPO'])**2), axis=1)

                else:
                    df[name+'_count']= df.apply(lambda x:compnorm*x[name]*x.EXPO-x['spicorr']*x[name+'_spurf']*x['ISGRI_EXPO'], axis=1)
                    df[name+'_count_err']= df.apply(lambda x:np.sqrt((compnorm*x[name+'_err']*x.EXPO)**2 + (x['spicorr']*x[name+'_spurf_err']*x['ISGRI_EXPO'])**2), axis=1)
            else: # with unique spicorr
                df[name+'_count']= df.apply(lambda x:compnorm*x[name]*x.EXPO-spicorr*x[name+'_spurf']*x['ISGRI_EXPO'], axis=1)
                df[name+'_count_err']= df.apply(lambda x:np.sqrt((compnorm*x[name+'_err']*x.EXPO)**2 + (spicorr*x[name+'_spurf_err']*x['ISGRI_EXPO'])**2), axis=1)
            total_count=df[name+'_count'].sum()
            # average over all scw:
            flux.append(total_count/full_expo_time) # divide by full exposure to get count/s
            flux_err.append( np.sqrt((df[name+'_count_err']**2).sum()) / full_expo_time)
        all_pola_dico['simple'][b]=[flux,flux_err]
    return all_pola_dico
