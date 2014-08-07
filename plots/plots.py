"""
Copyright (c) 2014, University of Washington
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import norm

data = pd.read_csv('puget_ages_and_site_data_NO_EROS.csv')

lab = {
    'lat': 'Latitude (degrees N)',
    'age': 'Exposure Age (yr)',
    'elev': 'Elevation (m)',
}

def labelit(xlab, ylab):
    plt.xlabel(lab[xlab])
    plt.ylabel(lab[ylab])

# extract the unique groups we will plot in different colors
groups = np.unique(data['group'])    

plt.figure("age vs latitude")

# 1) plot line from terry's paper
#500 m per year start at latest at 16750
# 1 degree = 111.32 km = 111320 m
# degrees per year = 500 / 111320 m 
yr_per_deg = -11320 / 500.0
x = np.linspace(46.75, 47.8, 400)
age = yr_per_deg * (x - x[0]) + 16750.0
plt.plot(x, age, 'k', label='P&S retreat')

# 2) plot samples
for grp in groups:
    is_grp = data['group'] == grp
    samps = data[is_grp]
    fmt = 'D' if (is_grp & data['bedrock']).any() else 'o'
    plt.errorbar(samps['lat'].values, samps['age'].values,
                 yerr=samps['err_int'].values, fmt=fmt, label=grp)

labelit('lat', 'age')
plt.legend(loc='best', numpoints=1)
plt.savefig('age_vs_latitude.pdf')

# see elevation effect
plt.figure("age vs elevation")
for grp in groups:
    samps = data[data['group'] == grp]
    plt.errorbar(samps['elev'].values,
                 samps['age'].values,
                 yerr=samps['err_ext'].values, fmt='o')
plt.legend(groups, loc='best', numpoints=1)
labelit('elev', 'age')
plt.savefig('age_vs_elevation.pdf')


# kernel density estimates
avg_ages = np.zeros(groups.size, dtype=float)
avg_lats = np.zeros(groups.size, dtype=float)
avg_err_int = np.zeros(groups.size, dtype=float)
avg_err_ext = np.zeros(groups.size, dtype=float)
for i, grp in enumerate(groups):
    plt.figure(grp + " kdf")
    samps = data[data['group'] == grp].to_records()
    lower = max(samps['age'].min()
                - 4 * samps['err_int'].max(), 0.0)
    upper = samps['age'].max() + 4 * samps['err_int'].max()
    x = np.linspace(lower, upper, 1000)
    tot_pdf = np.zeros_like(x)
    for smp in samps:
        pdf = norm(smp['age'], smp['err_int']).pdf(x)
        plt.plot(x, pdf, 'b-')
        tot_pdf += pdf
    
    plt.plot(x, tot_pdf, 'k-', linewidth=2)
    int_err = samps['err_int']
    ext_err = samps['err_ext']
    wt_int = 1 / (int_err)**2
    wt_ext = 1 / (ext_err)**2
    avg_ages[i] = np.average(samps['age'], weights=wt_int)
    avg_lats[i] = np.average(samps['lat'], weights=wt_int)
    avg_err_int[i] = 1 / np.sqrt(np.sum(wt_int))
    avg_err_ext[i] = 1 / np.sqrt(np.sum(wt_ext))
    plt.xlabel(lab['age'])
    plt.ylabel('Kernel density function')
    plt.title(grp)
    plt.savefig(grp + "_kdf.pdf")
    
# in last part we found the best ages for each set of samples... plot that data
avg_figs = ('int', 'ext')
avg_errs = (avg_err_int, avg_err_ext)
for i, fig in enumerate(avg_figs):
    plt.figure(fig)
    for j, grp in enumerate(groups):
        plt.errorbar(avg_lats[j], avg_ages[j], avg_errs[i][j], fmt='o')
    labelit('lat', 'age')
    plt.legend(groups, loc='best', numpoints=1)
    plt.title('Best apparent age estimates for each site (' + fig + 'ernal)')
    plt.savefig(fig + ".pdf")
