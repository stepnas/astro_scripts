# Astro scripts 
Some related highlights of my research work at Special astrophysical observatory of Russian academy of science (SAO RAS).

### SCIPTS
**gauss_vels.py** – visualization of intensity distribution of accreation disk near the white dwarf in close binary system J1740. 
Here are a few instances of how I've used Nelder-Mead minimization from scipy package and visualization in polar coordinates using matplotlip (examine J1740.png plot).
This code uses doppler tomography technique, developed by Kotze and Potter (https://doi.org/10.1051/0004-6361/201629120).

**JHK_modeling.ipynb** – my set of software scripts for the analysing of stellar spectra (intensity distributions depending on the wavelength of the observed radiation) 
and modelling of the infra red (low energy) spectrum of a close binary system.
Here you can find some examples of how we use interpolation, some astrophysical equations, system of normal equations for estimating stellar parameters.
At the end of the file you can find the result of modeling stellar spectra. 

**spectra_analysis.ipynb** – my set of software scripts for reducing astronomical data and plotting some features of it. Don't go into details, just inspect some
graphs and examples of using the Monte-Carlo method for calculating uncertainties.

**Acccolumn.ipynb** – Plotting of gas pressure, velocity, and density in conjunction with integration of the gas dynamic equations using the fourth-order Runge-Kutta technique 
reliance on a white dwarf's height above its surface as well as integration of the spectrum, or how radiation intensity varies with energy, of gas above a star's surface based 
on the star's mass and other gas medium characteristics.
