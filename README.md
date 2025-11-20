The COMIBIS python library was made for polarization analysis of processed INTEGRAL/IBIS Compton data obtained from the C++ Compton mode pipeline (not included here). Details can be found in "Inflight calibration of IBIS Compton mode" (Bouchet et al.).

# Installation

The main classes and functions are found in the 'comibis' directory, while the notebook will allow for quick analysis. Optionally, the library can be installed with:
```console
python setup.py bdist_wheel
pip install dist/comibis-1.0-py3-none-any.whl
```
Required libraries: numpy, pandas, astropy, matplotlib, scipy, lmfit, arviz
Optional: iminuit

# User guide

The saved_pola_df contains raw fluxes for different sources, with one file per source.

The first step is to choose src (short-hand of the source name), while the scw_file_name_list will contain the name of the runs.

So far the only available data is for the 2023 outburst of MQ Swift J1727.8-1613 ([Bouchet et al. 2024](https://doi.org/10.1051/0004-6361/202450826)), and the Crab Nebula. More examples will be added in the future.

The next step allows you to make more selections on the scw you will use (date, off-axis angle, revolutions, your own scw list, etc...).

The last step will build the polarigram and compute the associated Polarization Angle (PA) and Fraction (PF). This can be done for different energy bands and displayed on a plot.

The time evolution of polarization parameters is also possible, by groups of rev or years.

NB: Some results may vary from reference papers since IBIS calibration files and methods have been updated throughout the years.
