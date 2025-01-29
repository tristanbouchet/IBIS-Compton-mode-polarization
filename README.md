# Compton-IBIS
python notebooks for analysis of INTEGRAL/IBIS Compton data, obtained from the Compton pipeline (not included here).

Required libraries: numpy, pandas, astropy, tqdm, matplotlib, scipy, lmfit.

# User guide

The saved_pola_df contains raw fluxes for different sources.

The first step is to choose the src name (ex: crab). The scw_file_name_list will contain the name of your runs (ex: ['all_2023']).

The next step allows you to make more selections on the scw you will use (date, angle, etc…).

The last step will build the polarigram and compute the associated Polarization Angle (PA) and Fraction (PF). This can be done for every energy bands and displayed on a plot.
