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

def combine_df(src, scw_file_name_list, saved_pola_folder,  save_name=None):
    df_pola_scw=pd.concat([pd.read_csv('{}/{}_{}_pola.csv'.format(saved_pola_folder, src,scw_file_name),
                    dtype={'SCW':'str','REV':'str'}) for scw_file_name in scw_file_name_list]) # charge all df in a list and concat them
    if save_name: df_pola_scw.to_csv('{}/{}_{}_pola.csv'.format(saved_pola_folder,src, save_name))
    return df_pola_scw

def save_pre_2020(src, scw_file_name_list, saved_pola_folder, save_name,year=2020):
    '''useful to remove data with wrong IC'''
    df_pola_scw=combine_df(src,scw_file_name_list, save_name=None)
    df_pola_scw['ISOT']=df_pola_scw.apply(lambda x:np.datetime64(x.ISOT),axis=1)
    df_pola_scw['YEAR']=df_pola_scw.apply(lambda x:x.ISOT.year,axis=1)
    df_pre2020 = df_pola_scw[df_pola_scw.YEAR<year]
    df_pre2020.to_csv('{}/{}_{}_pola.csv'.format(saved_pola_folder,src,save_name))

def charge_df(src, scw_file_name_list, saved_pola_folder='saved_pola_df', resp_dir=None, spicorr_file=None,
              plot=True, verbose=True):
    '''charge csv files of a source into dataframe'''
    df_pola_scw = combine_df(src, scw_file_name_list, saved_pola_folder, save_name=None)
    all_band_names=pd.unique([c.split('_')[0] for c in df_pola_scw.columns if '-' in c])
    # infer number of polarization bin from columns
    df_pola_scw['scw_id']=df_pola_scw.apply(lambda x:x.SCW[:-4],axis=1)
    df_pola_scw['SCW_ORDER']=df_pola_scw.apply(lambda x:int(x.SCW[4:8]),axis=1)
    # df_pola_scw = df_pola_scw.set_index('SCW')
    df_pola_scw['ISOT']=df_pola_scw.apply(lambda x:np.datetime64(x.ISOT),axis=1)
    df_pola_scw['YEAR']=df_pola_scw.apply(lambda x:x.ISOT.year,axis=1)
    df_pola_scw['MONTH']=df_pola_scw.apply(lambda x:x.ISOT.month,axis=1)
    if spicorr_file: # add a time-dependent spicorr and compnorm value in the df according to the year
        df_spicorr = pd.read_csv('{}/{}'.format(resp_dir, spicorr_file))
        df_pola_scw['spicorr']= df_pola_scw.apply(lambda x:df_spicorr.spicorr[(x.YEAR>=df_spicorr.YEAR_START)&(x.YEAR<=df_spicorr.YEAR_END)].iloc[0],axis=1)
        df_pola_scw['compnorm']= df_pola_scw.apply(lambda x:df_spicorr.compnorm[(x.YEAR>=df_spicorr.YEAR_START)&(x.YEAR<=df_spicorr.YEAR_END)].iloc[0],axis=1)
    
    if verbose:
        print('{0} scw found'.format(len(df_pola_scw)))
        print(all_band_names)
    if plot:
        if len(df_pola_scw.YEAR.unique())<3: df_pola_scw.groupby('REV').REV.count().plot(kind="bar") # if only 1 year of data, plots the revs histogram
        else: df_pola_scw.groupby('YEAR').YEAR.count().plot(kind="bar")
        plt.ylabel('Number of scw')
        plt.plot()
    return df_pola_scw, all_band_names

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
        with open('{}/{}'.format(scw_folder, scw_file),'r') as f:
            all_scw+=f.read().splitlines()
    all_scw=[scw.split('.')[0] for scw in all_scw]
    return df[df.scw_id.isin(all_scw)]

def sum_scw_df(df, all_band_names, spicorr, compnorm, n_pola_bands):
    ''' compute the average flux/error from the scw rate df '''
    full_expo_time=df.EXPO.sum()
    all_pola_dico={'simple':{},'simple_clean':{}, 'pond':{}, 'pond_clean':{}}

    for b in all_band_names:
        flux, flux_err=[],[]
        for p in range(n_pola_bands):
            name=b+'_'+str(p)
            if spicorr=='auto': # with a time dependent spicorr value
                # df[name+'_count']= df.apply(lambda x:compnorm*x[name]*x.EXPO-x['spicorr']*x[name+'_spurf']*x['ISGRI_EXPO'], axis=1)
                # df[name+'_count_err']= df.apply(lambda x:np.sqrt((compnorm*x[name+'_err']*x.EXPO)**2 + (x['spicorr']*x[name+'_spurf_err']*x['ISGRI_EXPO'])**2), axis=1)
                df[name+'_count']= df.apply(lambda x:x['compnorm']*x[name]*x.EXPO-x['spicorr']*x[name+'_spurf']*x['ISGRI_EXPO'], axis=1)
                df[name+'_count_err']= df.apply(lambda x:np.sqrt((x['compnorm']*x[name+'_err']*x.EXPO)**2 + (x['spicorr']*x[name+'_spurf_err']*x['ISGRI_EXPO'])**2), axis=1)
            else: # with unique spicorr
                df[name+'_count']= df.apply(lambda x:compnorm*x[name]*x.EXPO-spicorr*x[name+'_spurf']*x['ISGRI_EXPO'], axis=1)
                df[name+'_count_err']= df.apply(lambda x:np.sqrt((compnorm*x[name+'_err']*x.EXPO)**2 + (spicorr*x[name+'_spurf_err']*x['ISGRI_EXPO'])**2), axis=1)
            total_count=df[name+'_count'].sum()
            # average over all scw:
            flux.append(total_count/full_expo_time) # divide by full exposure to get count/s
            flux_err.append( np.sqrt((df[name+'_count_err']**2).sum()) / full_expo_time)
        all_pola_dico['simple'][b]=[flux,flux_err]
    return all_pola_dico
