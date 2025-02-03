This python notebooks was made for polarization analysis of pre-processed INTEGRAL/IBIS Compton data, obtained from the Compton pipeline (not included here).

Required libraries: numpy, pandas, astropy, tqdm, matplotlib, scipy, lmfit.

# User guide

The saved_pola_df contains raw fluxes for different sources.

The first step is to choose src (short-hand of the source name), while the scw_file_name_list will contain the name of your runs.

So far the only available data is for the 2023 outburst of MQ Swift J1727.8-1613 (see Bouchet et al. 2024, https://doi.org/10.1051/0004-6361/202450826). More examples will be added in the future (Crab, Cygnus X-1).

The next step allows you to make more selections on the scw you will use (date, off-axis angle, revolutions, etc…).

The last step will build the polarigram and compute the associated Polarization Angle (PA) and Fraction (PF). This can be done for different energy bands and displayed on a plot.

NB: Some results may vary from reference papers since IBIS calibration files have been updated (especially after year 2020).
