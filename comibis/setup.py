from setuptools import find_packages, setup

setup(
    name = 'comibis',
    packages = find_packages(include = ['comibis']),
    version = '0.1',
    description = 'Tools to analyze INTEGRAL/IBIS Compton data',
    author = 'Tristan Bouchet',
    install_requires = ['numpy', 'matplotlib', 'pandas', 'scipy','astropy', 'tqdm', 'lmfit']
)